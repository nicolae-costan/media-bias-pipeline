# Backend API

FastAPI backend for the final app. It loads the trained bias and emotion checkpoints once at startup, exposes text analysis endpoints, and wraps NewsAPI keyword search.

## Configuration

Set these values in `.env` or the shell:

```bash
BIAS_CHECKPOINT_PATH=BiasTransformer/tb_logs/bias_transformer/version_0/checkpoints/<best>.ckpt
EMOTION_CHECKPOINT_PATH=EmotionModels/tb_logs/emotion_classification/version_0/checkpoints/<best>.ckpt
EMOTION_THRESHOLDS_PATH=EmotionModels/thresholds.json
NEWS_API_KEY=<your-newsapi-key>
MODEL_DEVICE=auto
BACKEND_ALLOWED_ORIGINS='["http://localhost:3000", "http://localhost:5173"]'
```

## Run

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /health`
- `GET /api/v1/models/status`
- `POST /api/v1/analyze`
- `POST /api/v1/analyze/batch`
- `POST /api/v1/news/search`
- `POST /api/v1/news/analyze`
