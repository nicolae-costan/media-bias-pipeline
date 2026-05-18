import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Ensure we can import from backend
sys.path.insert(0, str(Path(__cwd__).resolve() if "__cwd__" in globals() else Path.cwd()))

from backend.config import get_settings
from backend.model_service import ModelService, EMOTION_LABELS

class SaliencyPipeline:
    def __init__(self, model_service: ModelService = None):
        self.fallback_mode = False
        self.model_service = model_service
        
        if model_service is None:
            settings = get_settings()
            # Verify checkpoints exist before attempting to load ModelService
            bias_path = Path(settings.bias_checkpoint_path) if settings.bias_checkpoint_path else None
            emo_path = Path(settings.emotion_checkpoint_path) if settings.emotion_checkpoint_path else None
            
            if not bias_path or not bias_path.exists():
                raise FileNotFoundError(f"Custom bias checkpoint not found locally at: {bias_path}")
            if not emo_path or not emo_path.exists():
                raise FileNotFoundError(f"Custom emotion checkpoint not found locally at: {emo_path}")
            
            self.model_service = ModelService(settings)
            self.model_service.load()
            print("Successfully loaded custom fine-tuned checkpoints for Saliency Analysis.")

    @torch.inference_mode()
    def analyze_text_saliency(self, text: str, max_ablation_words: int = 5) -> dict:
        text = text.strip()
        if not text:
            return {}

        # ----------------------------------------------------
        # 1. Base Predictions
        # ----------------------------------------------------
        if not self.fallback_mode:
            # A. Custom Checkpoints Mode
            # Bias base
            bias_tokenized = self.model_service.bias_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            )
            bias_input_ids = bias_tokenized["input_ids"].to(self.model_service.device)
            bias_attention_mask = bias_tokenized["attention_mask"].to(self.model_service.device)
            bias_logits = self.model_service.bias_model(
                input_ids=bias_input_ids, attention_mask=bias_attention_mask
            )
            bias_probs = F.softmax(bias_logits, dim=-1).cpu()[0]
            base_bias_score = float(bias_probs[1].item())

            # Emotion base
            emo_tokenized = self.model_service.emotion_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            )
            emo_input_ids = emo_tokenized["input_ids"].to(self.model_service.device)
            emo_attention_mask = emo_tokenized["attention_mask"].to(self.model_service.device)
            emo_logits = self.model_service.emotion_model(
                input_ids=emo_input_ids, attention_mask=emo_attention_mask
            )
            emo_probs = torch.sigmoid(emo_logits).cpu()[0]
            base_emotions = {
                label: float(emo_probs[i].item())
                for i, label in enumerate(EMOTION_LABELS)
            }
            
            # Attentions
            bias_outputs = self.model_service.bias_model.model(
                input_ids=bias_input_ids, 
                attention_mask=bias_attention_mask,
                output_attentions=True
            )
            device = self.model_service.device
            bias_tokenizer = self.model_service.bias_tokenizer
            bias_model = self.model_service.bias_model
            emotion_tokenizer = self.model_service.emotion_tokenizer
            emotion_model = self.model_service.emotion_model
        else:
            # B. Public Fallback Mode
            # Bias base
            bias_tokenized = self.bias_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            )
            bias_input_ids = bias_tokenized["input_ids"].to(self.device)
            bias_attention_mask = bias_tokenized["attention_mask"].to(self.device)
            bias_outputs = self.bias_model(
                input_ids=bias_input_ids, 
                attention_mask=bias_attention_mask,
                output_attentions=True
            )
            bias_probs = F.softmax(bias_outputs.logits, dim=-1).cpu()[0]
            # Use negative sentiment class (index 0) as proxy for bias/aggression
            base_bias_score = float(bias_probs[0].item())

            # Emotion base
            emo_tokenized = self.emotion_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            )
            emo_input_ids = emo_tokenized["input_ids"].to(self.device)
            emo_attention_mask = emo_tokenized["attention_mask"].to(self.device)
            emo_outputs = self.emotion_model(
                input_ids=emo_input_ids, attention_mask=emo_attention_mask
            )
            emo_probs = F.softmax(emo_outputs.logits, dim=-1).cpu()[0]
            
            # Map j-hartmann model emotions to EMOTION_LABELS
            raw_emotions = {
                label: float(emo_probs[i].item())
                for i, label in enumerate(self.fallback_emo_labels)
            }
            base_emotions = {
                "anger": raw_emotions.get("anger", 0.0),
                "disgust": raw_emotions.get("disgust", 0.0),
                "fear": raw_emotions.get("fear", 0.0),
                "joy": raw_emotions.get("joy", 0.0),
                "optimism": raw_emotions.get("surprise", 0.0), # surprise as optimism proxy
                "sadness": raw_emotions.get("sadness", 0.0),
                "neutral": raw_emotions.get("neutral", 0.0)
            }
            
            device = self.device
            bias_tokenizer = self.bias_tokenizer
            bias_model = self.bias_model
            emotion_tokenizer = self.emotion_tokenizer
            emotion_model = self.emotion_model

        # ----------------------------------------------------
        # 2. Extract Last Layer Self-Attention on [CLS]
        # ----------------------------------------------------
        attentions = getattr(bias_outputs, "attentions", None)
        word_ids = bias_tokenized.word_ids()
        
        if attentions is not None and len(attentions) > 0:
            last_layer_att = attentions[-1] # Shape: (1, num_heads, seq_len, seq_len)
            cls_attention = last_layer_att[0].mean(dim=0)[0].cpu() # Mean across heads, CLS token index 0

            # Map Subword Tokens to Clean Words using word_ids
            word_attentions = {}
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is not None:
                    score = float(cls_attention[token_idx].item())
                    if word_idx not in word_attentions:
                        word_attentions[word_idx] = []
                    word_attentions[word_idx].append(score)

            # Average sub-token attention per word ID
            word_level_attention = {
                word_idx: sum(scores) / len(scores)
                for word_idx, scores in word_attentions.items()
            }
        else:
            # Safe Fallback when SDPA kernel is locked: uniform placeholder weight
            # This causes the pipeline to ablate all candidate words equally
            word_level_attention = {}
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is not None and word_idx not in word_level_attention:
                    word_level_attention[word_idx] = 1.0

        # Build clean word dictionaries and char boundaries
        word_saliency = {}
        for word_idx, att_score in word_level_attention.items():
            span = bias_tokenized.word_to_chars(word_idx)
            if span is not None:
                start, end = span
                word_text = text[start:end].strip().lower().strip(".,!?\"'()[]{}*&-;:")
                if word_text and len(word_text) > 1: # Ignore single letters
                    word_saliency[word_idx] = {
                        "word": word_text,
                        "attention": att_score,
                        "span": (start, end)
                    }

        # ----------------------------------------------------
        # 4. Ranked Targeted Ablation
        # ----------------------------------------------------
        ranked_candidates = sorted(
            word_saliency.items(), key=lambda x: x[1]["attention"], reverse=True
        )
        ablation_targets = ranked_candidates[:max_ablation_words]

        mask_token = bias_tokenizer.mask_token or "[MASK]"
        emo_mask_token = emotion_tokenizer.mask_token or "[MASK]"

        ablation_results = []
        for word_idx, data in ablation_targets:
            start, end = data["span"]
            masked_text_bias = text[:start] + mask_token + text[end:]
            masked_text_emo = text[:start] + emo_mask_token + text[end:]

            # A. Masked Bias prediction
            m_bias_tok = bias_tokenizer(
                masked_text_bias, return_tensors="pt", truncation=True, max_length=256
            )
            m_bias_input = {k: v.to(device) for k, v in m_bias_tok.items()}
            m_bias_outs = bias_model(**m_bias_input)
            if self.fallback_mode:
                m_bias_probs = F.softmax(m_bias_outs.logits, dim=-1).cpu()[0]
                masked_bias_score = float(m_bias_probs[0].item())
            else:
                m_bias_probs = F.softmax(m_bias_outs, dim=-1).cpu()[0]
                masked_bias_score = float(m_bias_probs[1].item())
            bias_delta = base_bias_score - masked_bias_score

            # B. Masked Emotion prediction
            m_emo_tok = emotion_tokenizer(
                masked_text_emo, return_tensors="pt", truncation=True, max_length=256
            )
            m_emo_input = {k: v.to(device) for k, v in m_emo_tok.items()}
            m_emo_outs = emotion_model(**m_emo_input)
            if self.fallback_mode:
                m_emo_probs = F.softmax(m_emo_outs.logits, dim=-1).cpu()[0]
            else:
                m_emo_probs = torch.sigmoid(m_emo_outs).cpu()[0]
            
            emotion_deltas = {}
            if self.fallback_mode:
                raw_m_emotions = {
                    label: float(m_emo_probs[i].item())
                    for i, label in enumerate(self.fallback_emo_labels)
                }
                masked_emotions = {
                    "anger": raw_m_emotions.get("anger", 0.0),
                    "disgust": raw_m_emotions.get("disgust", 0.0),
                    "fear": raw_m_emotions.get("fear", 0.0),
                    "joy": raw_m_emotions.get("joy", 0.0),
                    "optimism": raw_m_emotions.get("surprise", 0.0),
                    "sadness": raw_m_emotions.get("sadness", 0.0),
                    "neutral": raw_m_emotions.get("neutral", 0.0)
                }
                for label in EMOTION_LABELS:
                    emotion_deltas[label] = base_emotions[label] - masked_emotions[label]
            else:
                for i, label in enumerate(EMOTION_LABELS):
                    masked_score = float(m_emo_probs[i].item())
                    emotion_deltas[label] = base_emotions[label] - masked_score

            ablation_results.append({
                "word": data["word"],
                "attention": data["attention"],
                "bias_drop": bias_delta,
                "emotion_drops": emotion_deltas
            })

        return {
            "text": text,
            "base_bias": base_bias_score,
            "base_emotions": base_emotions,
            "word_saliency": ablation_results
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="The corrupt and tyrannical media outlet spread massive state propaganda.")
    args = parser.parse_args()

    print("Initializing Saliency Pipeline...")
    pipeline = SaliencyPipeline()
    print("Analyzing Sentence:", f"'{args.test}'")
    res = pipeline.analyze_text_saliency(args.test, max_ablation_words=3)
    
    print("\n--- Saliency Results ---")
    print(f"Base Bias Score: {res['base_bias']:.4f}")
    print("Base Emotions:", {k: f"{v:.4f}" for k, v in res['base_emotions'].items()})
    print("\nTop Saliency Words:")
    for item in res["word_saliency"]:
        print(f" - Word: '{item['word']}'")
        print(f"   Attention Score : {item['attention']:.4f}")
        print(f"   Bias Drop Delta : {item['bias_drop']:.4f}")
        print(f"   Emotion Drops   : " + ", ".join(f"{k}: {v:.4f}" for k, v in item['emotion_drops'].items() if abs(v) > 0.01))
