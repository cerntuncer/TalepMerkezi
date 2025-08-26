using Microsoft.EntityFrameworkCore;
using TalepMerkezi.Data;
using TalepMerkezi.Services.AI;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllersWithViews();

builder.Services.AddDbContext<AppDbContext>(opt =>
    opt.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

builder.Services.AddHttpClient("ai", c =>
{
    var baseUrl = builder.Configuration["AI:BaseUrl"] ?? "http://localhost:8000/";
    c.BaseAddress = new Uri(baseUrl);
    c.Timeout = TimeSpan.FromSeconds(5);
});
builder.Services.AddScoped<IAIClassifier, AiClassifier>();

var app = builder.Build();

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
