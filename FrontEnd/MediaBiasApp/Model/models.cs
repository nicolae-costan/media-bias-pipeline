

namespace MediaBias.Maui.Models;

// ---------------------------------------------------------------------------
// NewsAPI
// ---------------------------------------------------------------------------

public class NewsApiResponse
{
    public string Status { get; set; } = "";
    public int TotalResults { get; set; }
    public List<Article> Articles { get; set; } = new();
}

public class Article
{
    public NewsSource Source { get; set; } = new();
    public string Author { get; set; } = "";
    public string Title { get; set; } = "";
    public string Description { get; set; } = "";
    public string Url { get; set; } = "";
    public string UrlToImage { get; set; } = "";
    public DateTime? PublishedAt { get; set; }
    public string Content { get; set; } = "";

    // Stable ID derived from URL
    public string ArticleId => Convert.ToBase64String(
        System.Security.Cryptography.MD5.HashData(
            System.Text.Encoding.UTF8.GetBytes(Url ?? Title ?? Guid.NewGuid().ToString())
        )
    ).Replace("/", "_").Replace("+", "-")[..16];

    public string FullText => string.IsNullOrWhiteSpace(Content)
        ? Description ?? ""
        : System.Text.RegularExpressions.Regex
            .Replace(Content, @"\[\+\d+ chars\]", "").Trim();
}

public class NewsSource
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
}

// ---------------------------------------------------------------------------
// Prediction API — mirrors FastAPI response schema
// ---------------------------------------------------------------------------

public class PredictRequest
{
    public List<ArticlePayload> Articles { get; set; } = new();
}

public class ArticlePayload
{
    public string ArticleId { get; set; } = "";
    public string Title { get; set; } = "";
    public string Text { get; set; } = "";
}

public class BatchResponse
{
    public List<PredictionResult> Results { get; set; } = new();
    public bool Saved { get; set; }
}

public class PredictionResult
{
    public string ArticleId { get; set; } = "";
    public EmotionResult Emotions { get; set; } = new();
    public string BiasLabel { get; set; } = "";
    public float PBiased { get; set; }
}

public class EmotionResult
{
    public Dictionary<string, float> Probs { get; set; } = new();
    public Dictionary<string, int> Predictions { get; set; } = new();
    public List<string> Active { get; set; } = new();
    public List<ChunkResult> Chunks { get; set; } = new();
}

public class ChunkResult
{
    public int ChunkIndex { get; set; }
    public Dictionary<string, float> Probs { get; set; } = new();
    public Dictionary<string, int> Predictions { get; set; } = new();
    public List<string> Active { get; set; } = new();
}


public class ArticleViewModel
{
    public Article Article { get; init; } = new();
    public PredictionResult Prediction { get; set; }  // null until analysed

    public string Title => Article.Title;
    public string Source => Article.Source.Name;
    public string PublishedAt => Article.PublishedAt?.ToString("MMM dd, HH:mm") ?? "";
    public string ImageUrl => Article.UrlToImage;

    // Bias display
    public bool HasPrediction => Prediction != null;
    public string BiasLabel => Prediction?.BiasLabel ?? "Pending...";
    public float PBiased => Prediction?.PBiased ?? 0f;
    public Color BiasColor => BiasLabel switch
    {
        "Biased" => Color.FromArgb("#E53935"),
        "Non-Biased" => Color.FromArgb("#43A047"),
        _ => Color.FromArgb("#9E9E9E"),
    };

    // Top 3 active emotions for the list card
    public string TopEmotions => Prediction == null ? ""
        : string.Join(" · ", (Prediction.Emotions.Active ?? new()).Take(3));

    // Bias confidence bar (0–1)
    public double BiasConfidence => Prediction?.PBiased ?? 0;
}