// Services.cs — NewsAPI + PredictionAPI clients

using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using MediaBias.Maui.Models;

namespace MediaBias.Maui.Services;

// ---------------------------------------------------------------------------
// Shared JSON options — matches FastAPI's snake_case naming
// ---------------------------------------------------------------------------

internal static class Json
{
    public static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };
}

// ---------------------------------------------------------------------------
// NewsAPI service
// ---------------------------------------------------------------------------

public class NewsService
{
    private readonly HttpClient _http;
    private readonly string _apiKey;

    // Known outlets you want to support in the source filter
    public static readonly List<string> KnownSources = new()
    {
        "bbc-news", "cnn", "fox-news", "the-guardian-uk",
        "reuters", "associated-press", "al-jazeera-english",
        "the-new-york-times", "the-washington-post", "nbc-news",
    };

    public NewsService(HttpClient http, string apiKey)
    {
        _http = http;
        _apiKey = apiKey;
    }

    /// <summary>
    /// Search articles by keyword. Returns up to <paramref name="pageSize"/> articles.
    /// </summary>
    public async Task<List<Article>> SearchByKeywordAsync(
        string keyword,
        int pageSize = 30,
        CancellationToken ct = default)
    {
        var url = $"https://newsapi.org/v2/everything"
                + $"?q={Uri.EscapeDataString(keyword)}"
                + $"&language=en"
                + $"&sortBy=publishedAt"
                + $"&pageSize={pageSize}"
                + $"&apiKey={_apiKey}";

        var response = await _http.GetFromJsonAsync<NewsApiResponse>(url, Json.Options, ct)
                       ?? new NewsApiResponse();

        return FilterUsable(response.Articles);
    }

    /// <summary>
    /// Fetch latest headlines from a specific source (e.g. "bbc-news").
    /// </summary>
    public async Task<List<Article>> SearchBySourceAsync(
        string source,
        int pageSize = 30,
        CancellationToken ct = default)
    {
        var url = $"https://newsapi.org/v2/top-headlines"
                + $"?sources={Uri.EscapeDataString(source)}"
                + $"&pageSize={pageSize}"
                + $"&apiKey={_apiKey}";

        var response = await _http.GetFromJsonAsync<NewsApiResponse>(url, Json.Options, ct)
                       ?? new NewsApiResponse();

        return FilterUsable(response.Articles);
    }

    /// <summary>
    /// Search by both keyword and source simultaneously.
    /// NewsAPI doesn't support combined filtering on /everything for sources,
    /// so we fetch both and merge, deduplicating by URL.
    /// </summary>
    public async Task<List<Article>> SearchAsync(
        string? keyword = null,
        string? source = null,
        int pageSize = 30,
        CancellationToken ct = default)
    {
        if (!string.IsNullOrWhiteSpace(keyword) && !string.IsNullOrWhiteSpace(source))
        {
            // Fetch from source, then client-side filter by keyword in title/description
            var bySource = await SearchBySourceAsync(source, pageSize, ct);
            return bySource
                .Where(a => (a.Title ?? "").Contains(keyword, StringComparison.OrdinalIgnoreCase)
                         || (a.Description ?? "").Contains(keyword, StringComparison.OrdinalIgnoreCase))
                .ToList();
        }

        if (!string.IsNullOrWhiteSpace(source))
            return await SearchBySourceAsync(source, pageSize, ct);

        if (!string.IsNullOrWhiteSpace(keyword))
            return await SearchByKeywordAsync(keyword, pageSize, ct);

        // Neither — return top headlines
        var url = $"https://newsapi.org/v2/top-headlines"
                + $"?language=en&pageSize={pageSize}&apiKey={_apiKey}";
        var response = await _http.GetFromJsonAsync<NewsApiResponse>(url, Json.Options, ct)
                       ?? new NewsApiResponse();
        return FilterUsable(response.Articles);
    }

    private static List<Article> FilterUsable(List<Article> articles) =>
        articles
            .Where(a => !string.IsNullOrWhiteSpace(a.FullText) && a.FullText.Length >= 50)
            .DistinctBy(a => a.Url)
            .ToList();
}

// ---------------------------------------------------------------------------
// Prediction API service
// ---------------------------------------------------------------------------

public class PredictionService
{
    private readonly HttpClient _http;
    private readonly string _baseUrl;

    public PredictionService(HttpClient http, string serverUrl)
    {
        _http = http;
        _baseUrl = serverUrl.TrimEnd('/');
    }

    /// <summary>
    /// Sends a batch of articles to the FastAPI server.
    /// Returns a dictionary keyed by article_id for easy lookup.
    /// </summary>
    public async Task<Dictionary<string, PredictionResult>> PredictBatchAsync(
        IEnumerable<ArticleViewModel> articles,
        CancellationToken ct = default)
    {
        var payloads = articles.Select(a => new ArticlePayload
        {
            ArticleId = a.Article.ArticleId,
            Title = a.Article.Title ?? "",
            Text = a.Article.FullText ?? "",
        }).ToList();

        if (!payloads.Any())
            return new();

        var body = JsonSerializer.Serialize(payloads, Json.Options);
        var content = new StringContent(body, Encoding.UTF8, "application/json");

        var response = await _http.PostAsync($"{_baseUrl}/predict_batch", content, ct);
        response.EnsureSuccessStatusCode();

        var result = await response.Content
            .ReadFromJsonAsync<BatchResponse>(Json.Options, ct)
            ?? new BatchResponse();

        return result.Results.ToDictionary(r => r.ArticleId);
    }

    /// <summary>
    /// Checks whether the Python server is alive and models are loaded.
    /// </summary>
    public async Task<bool> IsHealthyAsync(CancellationToken ct = default)
    {
        try
        {
            var response = await _http.GetAsync($"{_baseUrl}/health", ct);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
}