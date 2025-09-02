using Microsoft.EntityFrameworkCore;
using TalepMerkezi.Data;
using TalepMerkezi.Services.AI;

var builder = WebApplication.CreateBuilder(args);

// MVC
builder.Services.AddControllersWithViews();

// DB
builder.Services.AddDbContext<AppDbContext>(opt =>
    opt.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

// === AI Servisi ===
// Development: Hata fırlatsın, debug kolay olsun.
// Production: Daha toleranslı olsun, boş sonuç dönsün.
// HttpClientFactory (hem Dev hem Prod için hazır olsun)
builder.Services.AddHttpClient("ai", c =>
{
    c.Timeout = TimeSpan.FromSeconds(15);
});

if (builder.Environment.IsDevelopment())
{
    builder.Services.AddScoped<IAIClassifier, AiClassifier>();
}
else
{
    builder.Services.AddHttpClient<IAIClassifier, HttpAIClassifier>(c =>
    {
        c.Timeout = TimeSpan.FromSeconds(15); // ML için güvenli timeout
    });
}

// === App ===
var app = builder.Build();

// --- DB migrate on startup (SQLite) ---
using (var scope = app.Services.CreateScope())
{
    try
    {
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        db.Database.Migrate();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[DB] Migration failed: {ex.Message}");
        // Uygulama yine de ayağa kalksın; log yeterli.
    }
}

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();
app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Talepler}/{action=Index}/{id?}");

app.Run();