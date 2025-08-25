using System.Net.Http.Json;
using Microsoft.Extensions.Configuration;

namespace TalepMerkezi.Services.AI
{
    public sealed class HttpAIClassifier : IAIClassifier
    {
        private readonly HttpClient _http;
        private readonly string _baseUrl;

        private record Req(string text);
        private record Resp(bool ok, string label, double confidence, string model);

        public HttpAIClassifier(HttpClient http, IConfiguration config)
        {
            _http = http;
            _baseUrl = config["AI:BaseUrl"] ?? "http://localhost:8000";
        }

        public async Task<(string label, double confidence)> ClassifyAsync(string text, CancellationToken ct = default)
        {
            var resp = await _http.PostAsJsonAsync($"{_baseUrl}/classify", new Req(text), ct);
            resp.EnsureSuccessStatusCode();

            var data = await resp.Content.ReadFromJsonAsync<Resp>(cancellationToken: ct)
                       ?? throw new InvalidOperationException("AI response is null");

            return (data.label, data.confidence);
        }
    }
}
