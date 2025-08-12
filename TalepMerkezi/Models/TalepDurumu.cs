using System.ComponentModel.DataAnnotations;

namespace TalepMerkezi.Models
{
    public enum TalepDurumu
    {
        [Display(Name = "Yeni")] New = 0,
        [Display(Name = "İşlemde")] InProgress = 1,
        [Display(Name = "Tamamlandı")] Done = 2,
        [Display(Name = "İptal")] Canceled = 3
    }
}
