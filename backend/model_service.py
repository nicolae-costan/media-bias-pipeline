import importlib
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import HTTPException, status
from transformers import AutoTokenizer

from backend.config import Settings
from backend.schemas import AnalyzeResponse, BiasPrediction, BiasedSpan, SentimentPrediction


BIAS_LABELS = ["Non-biased", "Biased"]
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "optimism", "sadness", "neutral"]


@dataclass
class BiasOutput:
    prediction: BiasPrediction
    prob_biased: float


@contextmanager
def _local_module_context(package_dir: Path):
    package_dir = package_dir.resolve()
    package_dir_str = str(package_dir)
    old_path = list(sys.path)
    sys.path.insert(0, package_dir_str)
    sys.modules.pop("dataloader", None)
    try:
        yield
    finally:
        sys.path = old_path
        sys.modules.pop("dataloader", None)


def _resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _clean_text(text: str) -> str:
    return re.sub(
        r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*",
        "LINK",
        text,
        flags=re.MULTILINE,
    )


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    pattern = re.compile(r"[^.!?]+(?:[.!?]+|$)", flags=re.MULTILINE)
    spans: list[tuple[int, int, str]] = []
    for match in pattern.finditer(text):
        sentence = match.group().strip()
        if sentence:
            start = match.start() + len(match.group()) - len(match.group().lstrip())
            end = match.end() - len(match.group()) + len(match.group().rstrip())
            spans.append((start, end, text[start:end]))
    return spans


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = self._select_device(settings.model_device)
        self.bias_model: Any | None = None
        self.bias_tokenizer: Any | None = None
        self.emotion_model: Any | None = None
        self.emotion_tokenizer: Any | None = None
        self.bias_checkpoint_path = _resolve_path(settings.bias_checkpoint_path)
        self.emotion_checkpoint_path = _resolve_path(settings.emotion_checkpoint_path)
        self.emotion_thresholds_path = _resolve_path(settings.emotion_thresholds_path)

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("MODEL_DEVICE=cuda was requested but CUDA is not available")
            return torch.device("cuda")
        if requested == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        self._load_bias_model()
        self._load_emotion_model()

    def status(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "bias_checkpoint_path": str(self.bias_checkpoint_path) if self.bias_checkpoint_path else None,
            "emotion_checkpoint_path": str(self.emotion_checkpoint_path) if self.emotion_checkpoint_path else None,
            "emotion_thresholds_path": str(self.emotion_thresholds_path) if self.emotion_thresholds_path else None,
            "bias_model_loaded": self.bias_model is not None,
            "emotion_model_loaded": self.emotion_model is not None,
            "bias_labels": BIAS_LABELS,
            "emotion_labels": EMOTION_LABELS,
            "max_text_chars": self.settings.max_text_chars,
            "max_batch_size": self.settings.max_batch_size,
        }

    def _load_bias_model(self) -> None:
        if not self.bias_checkpoint_path:
            return
        if not self.bias_checkpoint_path.exists():
            raise RuntimeError(f"Bias checkpoint not found: {self.bias_checkpoint_path}")

        package_dir = Path.cwd() / "BiasTransformer"
        with _local_module_context(package_dir):
            module = importlib.import_module("BiasTransformer.model")
            BiasTransformer = getattr(module, "BiasTransformer")
            self.bias_model = BiasTransformer.load_from_checkpoint(
                str(self.bias_checkpoint_path),
                map_location=str(self.device),
                strict=False,
            )
            self.bias_model.to(self.device)
            self.bias_model.eval()
            self.bias_tokenizer = AutoTokenizer.from_pretrained(self.bias_model.hparams.encoder_model)

    def _load_emotion_model(self) -> None:
        if not self.emotion_checkpoint_path:
            return
        if not self.emotion_checkpoint_path.exists():
            raise RuntimeError(f"Emotion checkpoint not found: {self.emotion_checkpoint_path}")

        package_dir = Path.cwd() / "EmotionModels"
        with _local_module_context(package_dir):
            module = importlib.import_module("EmotionModels.model")
            EmotionModel = getattr(module, "EmotionModel")
            self.emotion_model = EmotionModel.load_from_checkpoint(
                str(self.emotion_checkpoint_path),
                map_location=str(self.device),
                strict=False,
            )
            self.emotion_model.to(self.device)
            self.emotion_model.eval()
            self.emotion_tokenizer = AutoTokenizer.from_pretrained(self.emotion_model.hparams.encoder_model)
            if self.emotion_thresholds_path and self.emotion_thresholds_path.exists():
                self.emotion_model.load_thresholds(str(self.emotion_thresholds_path))

    def analyze_many(self, texts: list[str], include_biased_spans: bool, max_spans: int) -> list[AnalyzeResponse]:
        self._validate_loaded()
        self._validate_texts(texts)
        if not texts:
            return []

        bias_outputs = self._predict_bias(texts)
        emotion_outputs = self._predict_emotions(texts)

        results: list[AnalyzeResponse] = []
        for idx, text in enumerate(texts):
            emotions = emotion_outputs[idx]
            spans = self.explain_bias_sentences(text, bias_outputs[idx].prob_biased, max_spans) if include_biased_spans else []
            results.append(
                AnalyzeResponse(
                    bias=bias_outputs[idx].prediction,
                    sentiment=self._sentiment_from_emotions(emotions),
                    emotions=emotions,
                    biased_spans=spans,
                )
            )
        return results

    def _validate_loaded(self) -> None:
        missing = []
        if self.bias_model is None or self.bias_tokenizer is None:
            missing.append("bias")
        if self.emotion_model is None or self.emotion_tokenizer is None:
            missing.append("emotion")
        if missing:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Required model(s) not loaded: {', '.join(missing)}. Configure checkpoint paths in .env.",
            )

    def _validate_texts(self, texts: list[str]) -> None:
        if len(texts) > self.settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Batch size exceeds max_batch_size={self.settings.max_batch_size}",
            )
        too_long = [idx for idx, text in enumerate(texts) if len(text) > self.settings.max_text_chars]
        if too_long:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Text at index {too_long[0]} exceeds max_text_chars={self.settings.max_text_chars}",
            )

    @torch.inference_mode()
    def _predict_bias(self, texts: list[str]) -> list[BiasOutput]:
        tokenized = self.bias_tokenizer(
            [_clean_text(text) for text in texts],
            padding=True,
            truncation=True,
            max_length=int(self.bias_model.hparams.max_length),
            return_tensors="pt",
        )
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        logits = self.bias_model(**tokenized)
        probs = F.softmax(logits, dim=-1).cpu()

        outputs: list[BiasOutput] = []
        for row in probs:
            prob_unbiased = float(row[0].item())
            prob_biased = float(row[1].item())
            label = "Biased" if prob_biased >= prob_unbiased else "Non-biased"
            confidence = max(prob_biased, prob_unbiased)
            outputs.append(
                BiasOutput(
                    prediction=BiasPrediction(
                        label=label,
                        prob_biased=prob_biased,
                        prob_unbiased=prob_unbiased,
                        confidence=confidence,
                    ),
                    prob_biased=prob_biased,
                )
            )
        return outputs

    @torch.inference_mode()
    def _predict_emotions(self, texts: list[str]) -> list[dict[str, float]]:
        tokenized = self.emotion_tokenizer(
            [_clean_text(text) for text in texts],
            padding=True,
            truncation=True,
            max_length=int(self.emotion_model.hparams.max_length),
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        logits = self.emotion_model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu()
        return [
            {label: float(prob.item()) for label, prob in zip(EMOTION_LABELS, row, strict=True)}
            for row in probs
        ]

    def _sentiment_from_emotions(self, emotions: dict[str, float]) -> SentimentPrediction:
        positive = emotions.get("joy", 0.0) + emotions.get("optimism", 0.0)
        negative = (
            emotions.get("anger", 0.0)
            + emotions.get("disgust", 0.0)
            + emotions.get("fear", 0.0)
            + emotions.get("sadness", 0.0)
        )
        denominator = positive + negative
        score = 0.0 if denominator <= 1e-8 else (positive - negative) / denominator
        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"
        return SentimentPrediction(score=float(score), label=label)

    def explain_bias_sentences(self, text: str, full_prob_biased: float, max_spans: int) -> list[BiasedSpan]:
        if max_spans <= 0:
            return []
        spans = _sentence_spans(text)
        if len(spans) <= 1:
            return []

        masked_texts = [(text[:start] + text[end:]).strip() for start, end, _ in spans]
        masked_probs = [output.prob_biased for output in self._predict_bias(masked_texts)]
        scored = []
        for (start, end, sentence), masked_prob in zip(spans, masked_probs, strict=True):
            delta = full_prob_biased - masked_prob
            if delta > 0:
                scored.append(BiasedSpan(start=start, end=end, text=sentence, bias_delta=float(delta)))
        scored.sort(key=lambda item: item.bias_delta, reverse=True)
        return scored[:max_spans]
