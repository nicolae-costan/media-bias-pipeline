"""
predict_server.py
=================


Start:
    uvicorn predict_server:app --host 0.0.0.0 --port 8000
"""

import os
import re
import sys
import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Optional

import torch
import numpy as np
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMOTION_CHECKPOINT  = os.getenv("EMOTION_CHECKPOINT",  "checkpoints/emotion_model.ckpt")
EMOTION_THRESHOLDS  = os.getenv("EMOTION_THRESHOLDS",  "checkpoints/thresholds.json")
BIAS_MODEL_PATH     = os.getenv("BIAS_MODEL_PATH",     "checkpoints/bias_model")
EMOTION_MAX_LENGTH  = int(os.getenv("EMOTION_MAX_LENGTH", 512))
BIAS_MAX_LENGTH     = int(os.getenv("BIAS_MAX_LENGTH",    512))

DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 5433))
DB_NAME     = os.getenv("DB_NAME",     "media_bias")
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

EMOTION_LABELS = [
    "Anger", "Contempt", "Disgust", "Fear", "Gratitude",
    "Guilt", "Happiness", "Hope", "Pride", "Relief",
    "Sadness", "Sympathy", "Emotions_Neutral",
]

# ---------------------------------------------------------------------------
# Inference classes
# ---------------------------------------------------------------------------

class EmotionInference:
    """
    Thin wrapper — delegates entirely to EmotionModel.predict(),
    which owns the sliding window logic and the tuned thresholds.
    """

    def __init__(self, checkpoint_path: str, max_length: int, thresholds_path: str = None):
        log.info(f"Loading EmotionModel from {checkpoint_path} ...")

        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, parent_dir)

        from EmotionModels.model import EmotionModel
        import sklearn.preprocessing

        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([sklearn.preprocessing.LabelEncoder])

        self._model = EmotionModel.load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
        self._model.eval()

        if thresholds_path:
            self._model.load_thresholds(thresholds_path)

        self._tokenizer  = AutoTokenizer.from_pretrained(self._model.hparams.encoder_model)
        self._max_length = max_length
        log.info(f"EmotionModel ready. Thresholds: {self._model.thresholds.tolist()}")

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Delegates to EmotionModel.predict() — threshold and sliding
        window logic lives in the model, not here.

        Returns list of N dicts:
        {
            "probs":       { "Anger": 0.82, ... },
            "predictions": { "Anger": 1, ... },
            "active":      ["Anger", "Disgust"],
            "chunks":      [ { "chunk_index": 0, "probs": ..., "active": ... }, ... ]
        }
        """
        return self._model.predict(
            texts,
            tokenizer  = self._tokenizer,
            max_length = self._max_length,
            stride     = 50,
        )


class BiasInference:
    """
    Wraps a standard HuggingFace fine-tuned BERT/RoBERTa bias classifier.
    Handles a batch of N texts in a single forward pass.

    Save your trained model with:
        model.save_pretrained("checkpoints/bias_model")
        tokenizer.save_pretrained("checkpoints/bias_model")

    The config.json must contain:
        "id2label": {"0": "Non-Biased", "1": "Biased"}
    """

    def __init__(self, model_path: str, max_length: int):
        log.info(f"Loading BiasModel from {model_path} ...")

        self._tokenizer  = AutoTokenizer.from_pretrained(model_path)
        self._model      = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.eval()
        self._max_length = max_length

        # Read label mapping from config.json — int keys
        raw = self._model.config.id2label or {0: "Non-Biased", 1: "Biased"}
        self._id2label = {int(k): v for k, v in raw.items()}
        self._biased_idx = next(k for k, v in self._id2label.items() if v == "Biased")

        log.info(f"BiasModel ready. Labels: {self._id2label}")

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Returns list of N dicts: { "label": "Biased", "p_biased": 0.74 }
        """
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        outputs = self._model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        # softmax — single-label, Biased XOR Non-Biased
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()  # [N, 2]

        return [
            {
                "label":    self._id2label[int(probs[i].argmax())],
                "p_biased": round(float(probs[i, self._biased_idx]), 4),
            }
            for i in range(len(texts))
        ]


class ModelRegistry:
    """
    Owns both models. Instantiated once at startup.
    Injected into every endpoint that needs it via FastAPI DI.
    """

    def __init__(self):
        self.emotion = EmotionInference(EMOTION_CHECKPOINT, EMOTION_MAX_LENGTH, EMOTION_THRESHOLDS)
        self.bias    = BiasInference(BIAS_MODEL_PATH, BIAS_MAX_LENGTH)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class Database:
    """Thin Postgres wrapper. Injected via FastAPI DI."""

    def __init__(self):
        self._params = dict(
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        self._ensure_table()

    def _connect(self):
        return psycopg2.connect(**self._params)

    def _ensure_table(self):
        conn = self._connect()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS article_predictions (
                article_id   TEXT PRIMARY KEY,
                title        TEXT,
                emotions     JSONB,
                bias_label   TEXT,
                p_biased     FLOAT,
                predicted_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        log.info("DB table ready.")

    def save_batch(self, records: list[dict]):
        if not records:
            return
        conn = self._connect()
        cur  = conn.cursor()
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO article_predictions
                (article_id, title, emotions, bias_label, p_biased)
            VALUES %s
            ON CONFLICT (article_id) DO UPDATE
                SET emotions     = EXCLUDED.emotions,
                    bias_label   = EXCLUDED.bias_label,
                    p_biased     = EXCLUDED.p_biased,
                    predicted_at = NOW()
            """,
            [
                (r["article_id"], r["title"],
                 json.dumps(r["emotions"]), r["bias_label"], r["p_biased"])
                for r in records
            ],
        )
        conn.commit()
        cur.close()
        conn.close()


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup, keeps models alive for entire server life
# ---------------------------------------------------------------------------

_registry: ModelRegistry | None = None
_database: Database | None      = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry, _database
    _registry = ModelRegistry()  # both models loaded here, once
    _database = Database()
    log.info("Server ready.")
    yield                        # server lives here — models stay in RAM
    log.info("Shutting down.")


app = FastAPI(title="Media Bias API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Dependency functions
# ---------------------------------------------------------------------------

def get_registry() -> ModelRegistry:
    """
    FastAPI calls this for every request that declares Depends(get_registry).
    It just returns the already-loaded registry — zero overhead,
    no loading, no instantiation on each call.
    """
    if _registry is None:
        raise HTTPException(503, "Models not loaded yet.")
    return _registry


def get_db() -> Database:
    if _database is None:
        raise HTTPException(503, "Database not ready.")
    return _database


# Clean type aliases so endpoint signatures stay readable
Models = Annotated[ModelRegistry, Depends(get_registry)]
DB     = Annotated[Database,      Depends(get_db)]

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ArticleRequest(BaseModel):
    article_id: str
    title:      Optional[str] = ""
    text:       str

class ChunkResult(BaseModel):
    chunk_index: int
    probs:       dict   # raw sigmoid score per emotion for this chunk
    predictions: dict   # 0/1 per emotion after thresholds
    active:      list   # emotions that fired in this chunk

class EmotionResult(BaseModel):
    probs:       dict         # mean sigmoid score over all chunks
    predictions: dict         # 0/1 per emotion (max over chunks)
    active:      list         # emotions that fired anywhere in article
    chunks:      list[ChunkResult]  # per-section breakdown

class PredictionResult(BaseModel):
    article_id: str
    emotions:   EmotionResult
    bias_label: str
    p_biased:   float

class BatchResponse(BaseModel):
    results: list[PredictionResult]
    saved:   bool

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',
        'LINK', str(text), flags=re.MULTILINE
    )


@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(
    articles: list[ArticleRequest],
    models:   Models,                  # injected — already in RAM
    db:       DB,                      # injected — already connected
):
    if not articles:
        raise HTTPException(422, "Empty batch.")
    if len(articles) > 100:
        raise HTTPException(422, "Max batch size is 100.")

    texts = [_clean(a.text) for a in articles]

    # Two forward passes total — one per model, each covering all N articles
    emotion_results = models.emotion.predict_batch(texts)   # list of dicts with chunks
    bias_results    = models.bias.predict_batch(texts)      # list of {label, p_biased}

    predictions = [
        PredictionResult(
            article_id = articles[i].article_id,
            emotions   = EmotionResult(
                probs       = emotion_results[i]["probs"],
                predictions = emotion_results[i]["predictions"],
                active      = emotion_results[i]["active"],
                chunks      = [ChunkResult(**c) for c in emotion_results[i]["chunks"]],
            ),
            bias_label = bias_results[i]["label"],
            p_biased   = bias_results[i]["p_biased"],
        )
        for i in range(len(articles))
    ]

    db.save_batch([
        {
            "article_id": p.article_id,
            "title":      articles[i].title,
            "emotions":   p.emotions.probs,
            "bias_label": p.bias_label,
            "p_biased":   p.p_biased,
        }
        for i, p in enumerate(predictions)
    ])

    return BatchResponse(results=predictions, saved=True)


@app.get("/health")
async def health(models: Models):
    return {
        "status":        "ok",
        "emotion_model": models.emotion is not None,
        "bias_model":    models.bias    is not None,
    }