using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Extensions.Configuration;

namespace TalepMerkezi.Services.AI
{
    public sealed class HttpAIClassifier : IAIClassifier
    {
        private readonly HttpClient _http;
        private readonly string _baseUrl;

        private record Req(string text);
        private record Resp(bool ok, string label, double confidence, string model);

        private static readonly JsonSerializerOptions JsonOpts = new(JsonSerializerDefaults.Web)
        {
            PropertyNameCaseInsensitive = true
        };

        public HttpAIClassifier(HttpClient http, IConfiguration config)
        {
            _http = http;
            _baseUrl = (config["AI:BaseUrl"] ?? "http://localhost:8000").TrimEnd('/');
        }

        public async Task<(string label, double confidence)> ClassifyAsync(string text, CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Sınıflandırılacak metin boş olamaz.", nameof(text));

            var url = $"{_baseUrl}/classify-support2"; // ML endpoint

            try
            {
                var resp = await _http.PostAsJsonAsync(url, new Req(text), JsonOpts, ct);
                resp.EnsureSuccessStatusCode();

                var data = await resp.Content.ReadFromJsonAsync<Resp>(JsonOpts, ct)
                           ?? throw new InvalidOperationException("AI response is null");

                return (data.label, data.confidence);
            }
            catch (Exception ex)
            {
                // İstersen burada throw; yapıp controller'ın try/catch'ine düşürebilirsin.
                // Controller'ın "boş etiket → Canceled" akışına uyum için aşağıyı bırakıyorum:
                Console.WriteLine($"[AI ERROR] {ex.Message}");
                return ("", 0.0);
            }
        }
    }
}
