namespace TalepMerkezi.Services.AI
{
    public interface IAIClassifier
    {
        Task<(string category, float confidence)> ClassifyAsync(string text, CancellationToken ct = default);
    }
}