from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class HealthResponse(BaseModel):
    status: Literal["ok"]
    device: str
    bias_model_loaded: bool
    emotion_model_loaded: bool
    news_api_configured: bool


class ModelStatusResponse(BaseModel):
    device: str
    bias_checkpoint_path: str | None
    emotion_checkpoint_path: str | None
    emotion_thresholds_path: str | None
    bias_model_loaded: bool
    emotion_model_loaded: bool
    bias_labels: list[str]
    emotion_labels: list[str]
    max_text_chars: int
    max_batch_size: int


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    include_biased_spans: bool = True
    max_spans: int = Field(default=5, ge=0, le=20)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=10)
    include_biased_spans: bool = False
    max_spans: int = Field(default=3, ge=0, le=20)

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_blank(cls, value: list[str]) -> list[str]:
        if any(not text.strip() for text in value):
            raise ValueError("texts must not contain blank values")
        return value


class BiasPrediction(BaseModel):
    label: Literal["Biased", "Non-biased"]
    prob_biased: float
    prob_unbiased: float
    confidence: float


class SentimentPrediction(BaseModel):
    score: float
    label: Literal["negative", "neutral", "positive"]


class BiasedSpan(BaseModel):
    start: int
    end: int
    text: str
    bias_delta: float


class AnalyzeResponse(BaseModel):
    bias: BiasPrediction
    sentiment: SentimentPrediction
    emotions: dict[str, float]
    biased_spans: list[BiasedSpan] = Field(default_factory=list)


class BatchAnalyzeResponse(BaseModel):
    results: list[AnalyzeResponse]


class NewsSearchRequest(BaseModel):
    keywords: str = Field(..., min_length=1, max_length=500)
    language: str = Field(default="en", min_length=2, max_length=2)
    sort_by: Literal["relevancy", "popularity", "publishedAt"] = "publishedAt"
    from_date: date | None = None
    to_date: date | None = None
    page: int = Field(default=1, ge=1)

    @field_validator("keywords")
    @classmethod
    def keywords_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("keywords must not be blank")
        return value


class NewsAnalyzeRequest(NewsSearchRequest):
    include_biased_spans: bool = False
    max_spans: int = Field(default=3, ge=0, le=20)


class NewsArticle(BaseModel):
    source_id: str | None
    source_name: str | None
    author: str | None
    title: str | None
    description: str | None
    url: HttpUrl
    url_to_image: HttpUrl | None
    published_at: datetime | None
    content: str | None


class NewsSearchResponse(BaseModel):
    total_results: int
    articles: list[NewsArticle]


class AnalyzedNewsArticle(BaseModel):
    article: NewsArticle
    analysis: AnalyzeResponse


class NewsAnalyzeResponse(BaseModel):
    total_results: int
    articles: list[AnalyzedNewsArticle]
