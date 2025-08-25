using System.Net.Http.Json;

namespace TalepMerkezi.Services.AI
{
    public class HttpAIClassifier : IAIClassifier
    {
        private readonly HttpClient _http;
        public HttpAIClassifier(HttpClient http) => _http = http;

        private sealed record Req(string text);
        private sealed record Resp(string category, float confidence);

        public async Task<(string category, float confidence)> ClassifyAsync(string text, CancellationToken ct = default)
        {
            var res = await _http.PostAsJsonAsync("/classify", new Req(text ?? string.Empty), ct);
            res.EnsureSuccessStatusCode();
            var dto = await res.Content.ReadFromJsonAsync<Resp>(cancellationToken: ct);
            return (dto!.category, dto.confidence);
        }
    }
}
