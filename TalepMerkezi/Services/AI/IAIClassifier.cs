namespace TalepMerkezi.Services.AI;

public interface IAIClassifier
{
    Task<(string label, double confidence)> ClassifyAsync(string text, CancellationToken ct = default);
}
