from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Media Bias API"
    api_prefix: str = "/api/v1"
    backend_allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"])

    bias_checkpoint_path: str | None = None
    emotion_checkpoint_path: str | None = None
    emotion_thresholds_path: str = "EmotionModels/thresholds.json"
    model_device: Literal["auto", "cpu", "cuda"] = "auto"
    max_text_chars: int = 20_000
    max_batch_size: int = 10
    long_text_chunk_chars: int = 2_500

    news_api_key: str | None = None
    news_api_base_url: str = "https://newsapi.org/v2"
    news_api_timeout_seconds: float = 10.0
    news_fetch_full_content: bool = False
    news_article_timeout_seconds: float = 8.0
    news_article_max_chars: int = 0
    news_article_user_agent: str = (
        "Mozilla/5.0 (compatible; MediaBiasPipeline/1.0; +https://localhost)"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
