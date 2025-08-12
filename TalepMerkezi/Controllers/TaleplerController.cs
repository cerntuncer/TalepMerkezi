using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using TalepMerkezi.Data;
using TalepMerkezi.Models;

namespace TalepMerkezi.Controllers
{
    public class TaleplerController : Controller
    {
        private readonly AppDbContext _db;
        public TaleplerController(AppDbContext db) => _db = db;

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
    }
}
