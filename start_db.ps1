# start_db.ps1
# Simple script to ensure Docker is running and start the database

Write-Host "Checking Docker status..." -ForegroundColor Cyan

# Check if Docker is running
docker ps >$null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker Desktop is not running. Starting it now..." -ForegroundColor Yellow
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    Write-Host "Waiting for Docker to be ready (this may take up to 60 seconds)..." -ForegroundColor Gray
    
    $timeout = 60
    $elapsed = 0
    while ($elapsed -lt $timeout) {
        docker ps >$null 2>&1
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep -Seconds 5
        $elapsed += 5
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker could not be started. Please start Docker Desktop manually." -ForegroundColor Red
    exit 1
}

Write-Host "Starting PostgreSQL + pgvector via Docker Compose..." -ForegroundColor Green
docker-compose up -d

Write-Host "`nDatabase is ready on localhost:5433" -ForegroundColor Green
Write-Host "User: postgres | Password: mediabias123 | DB: media_bias" -ForegroundColor Gray
