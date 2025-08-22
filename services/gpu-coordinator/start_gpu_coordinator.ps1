# Simple GPU Coordinator Starter - Fixed Version
Write-Host "Starting GPU Coordinator..." -ForegroundColor Green

# Check Docker
try {
    docker version | Out-Null
    Write-Host "Docker is running" -ForegroundColor Green
} catch {
    Write-Host "Docker not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Cleanup existing containers
Write-Host "Cleaning up..." -ForegroundColor Yellow
docker stop agentic-redis agentic-gpu-coordinator 2>$null
docker rm agentic-redis agentic-gpu-coordinator 2>$null

try {
    # Start Redis
    Write-Host "Starting Redis..." -ForegroundColor Cyan
    docker run -d --name agentic-redis -p 6379:6379 redis:7-alpine
    
    # Build GPU Coordinator (simple version)
    Write-Host "Building GPU Coordinator..." -ForegroundColor Cyan
    
    # Create simple Dockerfile
    $dockerfileContent = @"
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install fastapi==0.104.1 uvicorn==0.24.0 httpx==0.25.2 pydantic==2.5.0
COPY . .
RUN groupadd -r appuser && useradd -r -g appuser appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8080
CMD ["python", "main.py"]
"@
    
    $dockerfileContent | Out-File -FilePath "Dockerfile.temp" -Encoding ASCII
    docker build -f Dockerfile.temp -t gpu-coord:test .
    Remove-Item "Dockerfile.temp"
    
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed"
    }
    
    # Start GPU Coordinator
    Write-Host "Starting GPU Coordinator..." -ForegroundColor Cyan
    docker run -d --name agentic-gpu-coordinator -p 8080:8080 gpu-coord:test
    
    # Wait for startup
    Write-Host "Waiting for services..." -ForegroundColor Yellow
    Start-Sleep -Seconds 15
    
    # Test
    Write-Host "Testing..." -ForegroundColor Cyan
    $response = $null
    try { 
        $response = Invoke-RestMethod -Uri "http://localhost:8080/health" -TimeoutSec 10 
    } catch { 
        Write-Host "Initial test failed, waiting longer..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8080/health" -TimeoutSec 10
        } catch {
            $response = $null
        }
    }
    
    if ($response -and $response.status -eq "healthy") {
        Write-Host "GPU Coordinator is ready!" -ForegroundColor Green
        Write-Host "URL: http://localhost:8080" -ForegroundColor Cyan
        Write-Host "Health: http://localhost:8080/health" -ForegroundColor Cyan
        Write-Host "Status: http://localhost:8080/status" -ForegroundColor Cyan
        Write-Host "" 
        Write-Host "Run tests with: python test_gpu.py" -ForegroundColor Yellow
        
        # Quick test
        Write-Host ""
        Write-Host "Quick test results:" -ForegroundColor Yellow
        Write-Host "Health Status: $($response.status)" -ForegroundColor Green
        Write-Host "GPU Available: $($response.gpu_available)" -ForegroundColor Green
        Write-Host "Total GPUs: $($response.total_gpus)" -ForegroundColor Green
        
    } else {
        Write-Host "GPU Coordinator failed to start" -ForegroundColor Red
        Write-Host "Container logs:" -ForegroundColor Yellow
        docker logs agentic-gpu-coordinator
    }
    
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Container logs:" -ForegroundColor Yellow
    docker logs agentic-gpu-coordinator 2>$null
}

Write-Host ""
Write-Host "Commands to manage containers:" -ForegroundColor Gray
Write-Host "  Stop: docker stop agentic-gpu-coordinator agentic-redis" -ForegroundColor Gray
Write-Host "  Remove: docker rm agentic-gpu-coordinator agentic-redis" -ForegroundColor Gray
Write-Host "  Logs: docker logs agentic-gpu-coordinator" -ForegroundColor Gray