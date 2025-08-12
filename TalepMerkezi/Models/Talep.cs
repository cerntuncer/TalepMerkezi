using System;
using System.ComponentModel.DataAnnotations;

namespace TalepMerkezi.Models
{
    public class Talep
    {
        public int Id { get; set; }

        [Required, StringLength(80)]
        public string Name { get; set; } = default!;

        [Required, StringLength(80)]
        public string Surname { get; set; } = default!;

        [EmailAddress, StringLength(120)]
        public string? Email { get; set; }

        [Required, MinLength(5)]
        public string Text { get; set; } = default!;

        [Required]
        public TalepDurumu Status { get; set; } = TalepDurumu.New;

        [StringLength(50)]
        public string? PredictedLabel { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}
