import asyncio
import re
from html.parser import HTMLParser

import httpx
from fastapi import HTTPException, status

from backend.config import Settings
from backend.schemas import NewsArticle, NewsSearchRequest, NewsSearchResponse


TRUNCATION_RE = re.compile(r"\s*\[\+\d+\s+chars\]\s*$")
WHITESPACE_RE = re.compile(r"\s+")


class ArticleTextExtractor(HTMLParser):
    TEXT_TAGS = {"article", "main", "p", "h1", "h2", "h3", "li", "blockquote"}
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "nav", "footer", "form"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._text_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag in self.TEXT_TAGS:
            self._text_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.TEXT_TAGS and self._text_depth:
            self._text_depth -= 1
            self._chunks.append("\n")
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._text_depth:
            return
        text = WHITESPACE_RE.sub(" ", data).strip()
        if text:
            self._chunks.append(f"{text} ")

    def text(self) -> str:
        lines = []
        for line in "".join(self._chunks).splitlines():
            clean_line = WHITESPACE_RE.sub(" ", line).strip()
            if clean_line:
                lines.append(clean_line)
        return "\n\n".join(lines)


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

        raw_articles = [article for article in payload.get("articles", []) if article.get("url")]
        articles = await self._normalize_articles(raw_articles[:10])
        return NewsSearchResponse(total_results=int(payload.get("totalResults", 0)), articles=articles[:10])

    async def _normalize_articles(self, articles: list[dict]) -> list[NewsArticle]:
        if not self.settings.news_fetch_full_content:
            return [self._normalize_article(article) for article in articles]

        headers = {"User-Agent": self.settings.news_article_user_agent}
        timeout = httpx.Timeout(self.settings.news_article_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            full_texts = await asyncio.gather(
                *(self._fetch_article_text(client, article["url"]) for article in articles),
                return_exceptions=True,
            )

        normalized = []
        for article, full_text in zip(articles, full_texts, strict=True):
            content = full_text if isinstance(full_text, str) and full_text else None
            normalized.append(self._normalize_article(article, content_override=content))
        return normalized

    async def _fetch_article_text(self, client: httpx.AsyncClient, url: str) -> str | None:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError:
            return None

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            return None

        extractor = ArticleTextExtractor()
        try:
            extractor.feed(response.text)
        except Exception:
            return None

        text = clean_article_content(extractor.text())
        if len(text) < 500:
            return None
        if self.settings.news_article_max_chars > 0:
            return text[: self.settings.news_article_max_chars]
        return text

    @staticmethod
    def _news_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text or "NewsAPI returned an HTTP error."
        return payload.get("message") or payload.get("code") or "NewsAPI returned an HTTP error."

    @staticmethod
    def _normalize_article(article: dict, content_override: str | None = None) -> NewsArticle:
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
            content=content_override or clean_article_content(article.get("content")),
        )


def article_analysis_text(article: NewsArticle) -> str:
    parts = [article.title, article.description, article.content]
    return "\n\n".join(part for part in parts if part)


def clean_article_content(content: str | None) -> str | None:
    if not content:
        return None
    content = TRUNCATION_RE.sub("", content)
    content = WHITESPACE_RE.sub(" ", content).strip()
    return content or None
