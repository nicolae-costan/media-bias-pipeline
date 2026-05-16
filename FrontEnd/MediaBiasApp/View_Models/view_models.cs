// ViewModels.cs — MVVM ViewModels for both pages

using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using MediaBias.Maui.Models;
using MediaBias.Maui.Services;

namespace MediaBias.Maui.ViewModels;

// ---------------------------------------------------------------------------
// Base ViewModel — INotifyPropertyChanged boilerplate
// ---------------------------------------------------------------------------

public abstract class BaseViewModel : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;

    protected void Set<T>(ref T field, T value, [CallerMemberName] string? prop = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return;
        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(prop));
    }

    protected void Notify([CallerMemberName] string? prop = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(prop));
}

// ---------------------------------------------------------------------------
// News list page ViewModel
// ---------------------------------------------------------------------------

public class NewsListViewModel : BaseViewModel
{
    private readonly NewsService _news;
    private readonly PredictionService _prediction;

    // ── State ────────────────────────────────────────────────────────────────

    private string _keyword = "";
    private string _selectedSource = "";
    private bool _isLoading;
    private bool _isAnalysing;
    private string _statusMessage = "Search for news or pick a source to begin.";
    private bool _serverHealthy = true;

    public string Keyword
    {
        get => _keyword;
        set => Set(ref _keyword, value);
    }

    public string SelectedSource
    {
        get => _selectedSource;
        set => Set(ref _selectedSource, value);
    }

    public bool IsLoading
    {
        get => _isLoading;
        set { Set(ref _isLoading, value); Notify(nameof(IsIdle)); }
    }

    public bool IsAnalysing
    {
        get => _isAnalysing;
        set { Set(ref _isAnalysing, value); Notify(nameof(IsIdle)); }
    }

    public bool IsIdle => !IsLoading && !IsAnalysing;

    public string StatusMessage
    {
        get => _statusMessage;
        set => Set(ref _statusMessage, value);
    }

    public bool ServerHealthy
    {
        get => _serverHealthy;
        set => Set(ref _serverHealthy, value);
    }

    // Articles shown in the list
    public ObservableCollection<ArticleViewModel> Articles { get; } = new();

    // Source picker options
    public List<string> Sources { get; } = new List<string> { "" }
    .Concat(NewsService.KnownSources)
    .ToList();

    // ── Commands ──────────────────────────────────────────────────────────────

    public ICommand SearchCommand { get; }
    public ICommand AnalyseCommand { get; }
    public ICommand RefreshCommand { get; }

    // Navigation callback — set by the page so the VM doesn't depend on Navigation
    public Func<ArticleViewModel, Task>? NavigateToDetail { get; set; }
    public ICommand OpenDetailCommand { get; }

    // ── Constructor ───────────────────────────────────────────────────────────

    public NewsListViewModel(NewsService news, PredictionService prediction)
    {
        _news = news;
        _prediction = prediction;

        SearchCommand = new AsyncCommand(SearchAsync, () => IsIdle);
        AnalyseCommand = new AsyncCommand(AnalyseAsync, () => IsIdle && Articles.Any());
        RefreshCommand = new AsyncCommand(CheckHealthAsync);

        OpenDetailCommand = new AsyncCommand<ArticleViewModel>(async vm =>
        {
            if (vm != null && NavigateToDetail != null)
                await NavigateToDetail(vm);
        });

        // Check server health on startup
        Task.Run(CheckHealthAsync);
    }

    // ── Search ────────────────────────────────────────────────────────────────

    private async Task SearchAsync()
    {
        IsLoading = true;
        StatusMessage = "Fetching articles...";
        Articles.Clear();

        try
        {
            var articles = await _news.SearchAsync(
                keyword: string.IsNullOrWhiteSpace(Keyword) ? null : Keyword,
                source: string.IsNullOrWhiteSpace(SelectedSource) ? null : SelectedSource,
                pageSize: 30
            );

            foreach (var article in articles)
                Articles.Add(new ArticleViewModel { Article = article });

            StatusMessage = articles.Count == 0
                ? "No articles found. Try a different search."
                : $"{articles.Count} articles fetched. Tap 'Analyse' to run predictions.";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Failed to fetch news: {ex.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }

    // ── Analyse ───────────────────────────────────────────────────────────────

    private async Task AnalyseAsync()
    {
        if (!Articles.Any()) return;

        IsAnalysing = true;
        StatusMessage = $"Analysing {Articles.Count} articles...";

        try
        {
            // Send all articles to FastAPI in one batch call
            var results = await _prediction.PredictBatchAsync(Articles);

            int found = 0;
            foreach (var vm in Articles)
            {
                if (results.TryGetValue(vm.Article.ArticleId, out var result))
                {
                    vm.Prediction = result;
                    found++;
                    // Notify the list that this item changed
                    Notify(nameof(Articles));
                }
            }

            var biasedCount = Articles.Count(a => a.BiasLabel == "Biased");
            StatusMessage = $"Done. {biasedCount}/{Articles.Count} articles flagged as biased.";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Analysis failed: {ex.Message}";
        }
        finally
        {
            IsAnalysing = false;
        }
    }

    // ── Health check ──────────────────────────────────────────────────────────

    private async Task CheckHealthAsync()
    {
        ServerHealthy = await _prediction.IsHealthyAsync();
        if (!ServerHealthy)
            StatusMessage = " Prediction server unreachable. Check your VM address.";
    }
}

// ---------------------------------------------------------------------------
// Article detail page ViewModel
// ---------------------------------------------------------------------------

public class ArticleDetailViewModel : BaseViewModel
{
    private ArticleViewModel _article = new();

    public ArticleViewModel Article
    {
        get => _article;
        set
        {
            Set(ref _article, value);
            Notify(nameof(Title));
            Notify(nameof(Source));
            Notify(nameof(BiasLabel));
            Notify(nameof(BiasColor));
            Notify(nameof(PBiasedPercent));
            Notify(nameof(ActiveEmotions));
            Notify(nameof(Chunks));
            Notify(nameof(HasChunks));
            Notify(nameof(AllEmotions));
        }
    }

    public string Title => Article.Title;
    public string Source => Article.Source;
    public string BiasLabel => Article.BiasLabel;
    public Color BiasColor => Article.BiasColor;
    public string PBiasedPercent => $"{Article.PBiased * 100:F0}%";

    // Active emotions at article level
    public List<EmotionBadge> ActiveEmotions =>
        (Article.Prediction?.Emotions.Active ?? new())
            .Select(e => new EmotionBadge(e, Article.Prediction!.Emotions.Probs.GetValueOrDefault(e)))
            .ToList();

    // All emotions with their raw probabilities (for a full breakdown bar chart)
    public List<EmotionBadge> AllEmotions =>
        (Article.Prediction?.Emotions.Probs ?? new())
            .OrderByDescending(kv => kv.Value)
            .Select(kv => new EmotionBadge(kv.Key, kv.Value))
            .ToList();

    // Per-chunk breakdown
    public List<ChunkDisplay> Chunks =>
        (Article.Prediction?.Emotions.Chunks ?? new())
            .Select(c => new ChunkDisplay(c))
            .ToList();

    public bool HasChunks => Chunks.Any();

    public ICommand OpenUrlCommand { get; } = new AsyncCommand<string>(async url =>
    {
        if (!string.IsNullOrWhiteSpace(url))
            await Launcher.OpenAsync(url);
    });
}

// ── Display helpers ────────────────────────────────────────────────────────

public record EmotionBadge(string Name, float Score)
{
    public string ScoreText => $"{Score * 100:F0}%";
    public double ScoreWidth => Score;   // bind to ProgressBar.Progress
    public Color BadgeColor => Score > 0.6f
        ? Color.FromArgb("#E53935")
        : Score > 0.35f
            ? Color.FromArgb("#FB8C00")
            : Color.FromArgb("#43A047");
}

public class ChunkDisplay
{
    public string Title { get; }
    public List<EmotionBadge> ActiveEmotions { get; }
    public string ActiveText { get; }

    public ChunkDisplay(ChunkResult chunk)
    {
        Title = $"Section {chunk.ChunkIndex + 1}";
        ActiveEmotions = chunk.Active
            .Select(e => new EmotionBadge(e, chunk.Probs.GetValueOrDefault(e)))
            .ToList();
        ActiveText = chunk.Active.Any()
            ? string.Join(", ", chunk.Active)
            : "No strong emotions";
    }
}

// ---------------------------------------------------------------------------
// AsyncCommand helpers — keeps ViewModels free of ICommand boilerplate
// ---------------------------------------------------------------------------

public class AsyncCommand : ICommand
{
    private readonly Func<Task> _execute;
    private readonly Func<bool>? _canExecute;
    private bool _running;

    public AsyncCommand(Func<Task> execute, Func<bool>? canExecute = null)
    {
        _execute = execute;
        _canExecute = canExecute;
    }

    public event EventHandler? CanExecuteChanged;

    public bool CanExecute(object? _) => !_running && (_canExecute?.Invoke() ?? true);

    public async void Execute(object? _)
    {
        if (!CanExecute(null)) return;
        _running = true;
        CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        try { await _execute(); }
        finally
        {
            _running = false;
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}

public class AsyncCommand<T> : ICommand
{
    private readonly Func<T?, Task> _execute;
    public AsyncCommand(Func<T?, Task> execute) => _execute = execute;
    public event EventHandler? CanExecuteChanged;
    public bool CanExecute(object? _) => true;
    public async void Execute(object? p) => await _execute(p is T t ? t : default);
}