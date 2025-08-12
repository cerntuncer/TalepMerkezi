using Microsoft.EntityFrameworkCore;
using TalepMerkezi.Models;

namespace TalepMerkezi.Data
{
    public class AppDbContext : DbContext
    {
        public DbSet<Talep> Talepler => Set<Talep>();
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<Talep>()
                .Property(t => t.Status)
                .HasConversion<string>()   // DB'de string sakla
                .HasMaxLength(20);
        }
    }
}
