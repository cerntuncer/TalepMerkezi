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

            var url = $"{_baseUrl}/classify"; // 👈 doğru endpoint olduğunu varsayıyoruz

            try
            {
                Console.WriteLine($"[AI Request] URL: {url}");
                Console.WriteLine($"[AI Request] Text: {text}");

                var resp = await _http.PostAsJsonAsync(url, new Req(text), JsonOpts, ct);
                resp.EnsureSuccessStatusCode();

                var data = await resp.Content.ReadFromJsonAsync<Resp>(JsonOpts, ct)
                           ?? throw new InvalidOperationException("AI response is null");

                Console.WriteLine($"[AI Response] Label: {data.label}, Confidence: {data.confidence}, Model: {data.model}");

                return (data.label, data.confidence);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AI ERROR] AI servisine ulaşılamadı veya hata oluştu.");
                Console.WriteLine($"[AI ERROR] Hata mesajı: {ex.Message}");
                return ("", 0.0);  // Controller bu durumda iptal olarak işaretleyecek
            }
        }
    }
}
