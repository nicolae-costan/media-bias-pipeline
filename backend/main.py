from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.model_service import ModelService
from backend.news_service import NewsService, article_analysis_text
from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    HealthResponse,
    ModelStatusResponse,
    NewsAnalyzeRequest,
    NewsAnalyzeResponse,
    NewsSearchRequest,
    NewsSearchResponse,
    AnalyzedNewsArticle,
)


settings = get_settings()
model_service = ModelService(settings)
news_service = NewsService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = model_service.status()
    return HealthResponse(
        status="ok",
        device=status["device"],
        bias_model_loaded=status["bias_model_loaded"],
        emotion_model_loaded=status["emotion_model_loaded"],
        news_api_configured=bool(settings.news_api_key),
    )


@app.get(f"{settings.api_prefix}/models/status", response_model=ModelStatusResponse)
def models_status() -> ModelStatusResponse:
    return ModelStatusResponse(**model_service.status())


@app.post(f"{settings.api_prefix}/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return model_service.analyze_many(
        [request.text],
        include_biased_spans=request.include_biased_spans,
        max_spans=request.max_spans,
    )[0]


@app.post(f"{settings.api_prefix}/analyze/batch", response_model=BatchAnalyzeResponse)
def analyze_batch(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    return BatchAnalyzeResponse(
        results=model_service.analyze_many(
            request.texts,
            include_biased_spans=request.include_biased_spans,
            max_spans=request.max_spans,
        )
    )


@app.post(f"{settings.api_prefix}/news/search", response_model=NewsSearchResponse)
async def news_search(request: NewsSearchRequest) -> NewsSearchResponse:
    return await news_service.search(request)


@app.post(f"{settings.api_prefix}/news/analyze", response_model=NewsAnalyzeResponse)
async def news_analyze(request: NewsAnalyzeRequest) -> NewsAnalyzeResponse:
    news = await news_service.search(request)
    article_text_pairs = [
        (article, text)
        for article in news.articles
        if (text := article_analysis_text(article).strip())
    ]
    articles = [article for article, _ in article_text_pairs]
    analyses = [
        model_service.analyze_long_text(
            text,
            include_biased_spans=request.include_biased_spans,
            max_spans=request.max_spans,
        )
        for _, text in article_text_pairs
    ]
    return NewsAnalyzeResponse(
        total_results=news.total_results,
        articles=[
            AnalyzedNewsArticle(article=article, analysis=analysis)
            for article, analysis in zip(articles, analyses, strict=True)
        ],
    )
