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
        _baseUrl = cfg["AI:BaseUrl"] ?? "http://localhost:8000/";
    }

    public async Task<(string label, double confidence)> ClassifyAsync(string text, CancellationToken ct = default)
    {
        var client = _httpFactory.CreateClient("ai");
        var resp = await client.PostAsJsonAsync("classify", new Req(text), ct); // sadece relative path
        resp.EnsureSuccessStatusCode();
        var data = await resp.Content.ReadFromJsonAsync<Resp>(cancellationToken: ct)
                   ?? throw new InvalidOperationException("AI response null");
        return (data.label, data.confidence);
    }
}
