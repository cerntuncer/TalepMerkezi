// TalepMerkezi/Services/AI/AiClassifier.cs
using System.Net.Http.Json;
using Microsoft.Extensions.Configuration;

namespace TalepMerkezi.Services.AI;

public sealed class AiClassifier : IAIClassifier
{
    private readonly IHttpClientFactory _httpFactory;
    private readonly string _baseUrl;

    private record Req(string text);
    private record Resp(bool ok, string label, double confidence, string model);

    public AiClassifier(IHttpClientFactory httpFactory, IConfiguration cfg)
    {
        _httpFactory = httpFactory;
        _baseUrl = cfg["AI:BaseUrl"] ?? "http://localhost:8000";
    }

    public async Task<(string label, double confidence)> ClassifyAsync(string text, CancellationToken ct = default)
    {
        var client = _httpFactory.CreateClient("ai");
        var url = $"{_baseUrl.TrimEnd('/')}/classify-support2";

        var resp = await client.PostAsJsonAsync(url, new Req(text), ct);
        resp.EnsureSuccessStatusCode();

        var data = await resp.Content.ReadFromJsonAsync<Resp>(cancellationToken: ct)
                   ?? throw new InvalidOperationException("AI response is null");
        return (data.label, data.confidence);
    }
}
