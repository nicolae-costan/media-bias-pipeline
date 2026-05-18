import httpx
from fastapi import HTTPException, status

from backend.config import Settings
from backend.schemas import NewsArticle, NewsSearchRequest, NewsSearchResponse


class NewsService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def search(self, request: NewsSearchRequest) -> NewsSearchResponse:
        if not self.settings.news_api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEWS_API_KEY is not configured.",
            )

        params = {
            "q": request.keywords,
            "language": request.language,
            "sortBy": request.sort_by,
            "pageSize": 10,
            "page": request.page,
        }
        if request.from_date:
            params["from"] = request.from_date.isoformat()
        if request.to_date:
            params["to"] = request.to_date.isoformat()

        headers = {"X-Api-Key": self.settings.news_api_key}
        url = f"{self.settings.news_api_base_url.rstrip('/')}/everything"
        try:
            async with httpx.AsyncClient(timeout=self.settings.news_api_timeout_seconds) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._news_error_detail(exc.response)
            raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"NewsAPI request failed: {exc}",
            ) from exc

        payload = response.json()
        if payload.get("status") != "ok":
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=payload.get("message", "NewsAPI returned an error response."),
            )

        articles = [
            self._normalize_article(article)
            for article in payload.get("articles", [])
            if article.get("url")
        ]
        return NewsSearchResponse(total_results=int(payload.get("totalResults", 0)), articles=articles[:10])

    @staticmethod
    def _news_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text or "NewsAPI returned an HTTP error."
        return payload.get("message") or payload.get("code") or "NewsAPI returned an HTTP error."

    @staticmethod
    def _normalize_article(article: dict) -> NewsArticle:
        source = article.get("source") or {}
        return NewsArticle(
            source_id=source.get("id"),
            source_name=source.get("name"),
            author=article.get("author"),
            title=article.get("title"),
            description=article.get("description"),
            url=article.get("url"),
            url_to_image=article.get("urlToImage"),
            published_at=article.get("publishedAt"),
            content=article.get("content"),
        )


def article_analysis_text(article: NewsArticle) -> str:
    parts = [article.title, article.description, article.content]
    return "\n\n".join(part for part in parts if part)

