# scripts/dev-up.ps1
# PowerShell script to spin up the dev environment on Windows

Write-Host "=== RAG Stack Dev Up (Windows) ==="

# Check if Docker is running
if (-not (Get-Process -Name "com.docker" -ErrorAction SilentlyContinue)) {
    Write-Warning "Docker Desktop does not appear to be running. Start Docker Desktop first."
    exit 1
}

# Copy .env.example to .env if missing
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
} else {
    Write-Host ".env already exists, skipping copy."
}

# Run docker compose
Write-Host "Starting docker compose..."
docker compose up --build
