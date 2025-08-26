# Build stage
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY TalepMerkezi.csproj ./
RUN dotnet restore "TalepMerkezi.csproj"

COPY . .
RUN dotnet publish "TalepMerkezi.csproj" -c Release -o /app/publish /p:UseAppHost=false

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app
COPY --from=build /app/publish .

# Kestrel listen on 8080 inside container
ENV ASPNETCORE_URLS=http://0.0.0.0:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "TalepMerkezi.dll"]