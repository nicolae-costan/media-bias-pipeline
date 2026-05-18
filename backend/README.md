# Backend API

FastAPI backend for the final app. It loads the trained bias and emotion checkpoints once at startup, exposes text analysis endpoints, and wraps NewsAPI keyword search.

## Configuration

Set these values in `.env` or the shell:

```bash
BIAS_CHECKPOINT_PATH=BiasTransformer/tb_logs/bias_transformer/version_0/checkpoints/<best>.ckpt
EMOTION_CHECKPOINT_PATH=EmotionModels/tb_logs/emotion_classification/version_0/checkpoints/<best>.ckpt
EMOTION_THRESHOLDS_PATH=EmotionModels/thresholds.json
NEWS_API_KEY=<your-newsapi-key>
NEWS_FETCH_FULL_CONTENT=true
NEWS_ARTICLE_MAX_CHARS=0
LONG_TEXT_CHUNK_CHARS=2500
MODEL_DEVICE=auto
BACKEND_ALLOWED_ORIGINS='["http://localhost:3000", "http://localhost:5173"]'
```

NewsAPI only returns truncated article snippets in `content`. When `NEWS_FETCH_FULL_CONTENT` is enabled, the backend uses NewsAPI for search results, then fetches each article URL and stores extracted page text in `article.content` when available.

`POST /api/v1/news/analyze` analyzes long fetched articles in chunks and aggregates the results, so full articles can contribute without raising the direct text-size limit used by `/api/v1/analyze`.
Set `NEWS_ARTICLE_MAX_CHARS` above zero only if you want to cap extracted article text before analysis.

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
