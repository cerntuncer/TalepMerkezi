using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using TalepMerkezi.Data;
using TalepMerkezi.Models;
using TalepMerkezi.Services.AI;

namespace TalepMerkezi.Controllers
{
    public class TaleplerController : Controller
    {
        private readonly AppDbContext _db;
        private readonly IAIClassifier _ai;

        public TaleplerController(AppDbContext db, IAIClassifier ai)
        {
            _db = db;
            _ai = ai;
        }

        [HttpGet("/")]
        public async Task<IActionResult> Index()
        {
            var list = await _db.Talepler
                .AsNoTracking()
                .OrderByDescending(x => x.Id)
                .Take(50)
                .ToListAsync();
            return View(list);
        }

        [HttpGet]
        public IActionResult Create() => View();

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Create([Bind("Name,Surname,Email,Text")] Talep input)
        {
            if (!ModelState.IsValid) return View(input);

            var t = new Talep
            {
                Name = input.Name.Trim(),
                Surname = input.Surname.Trim(),
                Email = string.IsNullOrWhiteSpace(input.Email) ? null : input.Email!.Trim(),
                Text = input.Text.Trim(),
                Status = TalepDurumu.New
            };

            _db.Talepler.Add(t);
            await _db.SaveChangesAsync();
            return RedirectToAction(nameof(Index));
        }

        // === AI sınıflandırma ===
        [HttpPost("/talepler/{id:int}/classify")]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Classify(int id)
        {
            var t = await _db.Talepler.FirstOrDefaultAsync(x => x.Id == id);
            if (t == null) return NotFound();

            // Daha önce tamamlandıysa tekrar işlem yapma (idempotent)
            if (t.Status == TalepDurumu.Done)
                return RedirectToAction(nameof(Index));

            // 1) "İşlemde" olarak işaretle
            t.Status = TalepDurumu.InProgress;
            await _db.SaveChangesAsync();

            try
            {
                // 2) Python AI servisine gönder
                var (cat, conf) = await _ai.ClassifyAsync(t.Text);

                // 3) Sonuçları kaydet
                t.PredictedLabel = cat;
                t.Status = TalepDurumu.Done;
                await _db.SaveChangesAsync();
            }
           
            catch (Exception ex)
            {
                Console.WriteLine($"[Sınıflandırma Hatası] TalepId: {t.Id}, Hata: {ex.Message}");
                t.Status = TalepDurumu.Canceled;
                await _db.SaveChangesAsync();
            }

            

            return RedirectToAction(nameof(Index));
        }
    }
}
