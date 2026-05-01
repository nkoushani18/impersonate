# Start Ollama with GPU-only configuration
# Force use of GPU 0 (first GPU)

Write-Host "🎮 Starting Ollama with GPU-only mode..." -ForegroundColor Green
Write-Host ""

# Set environment variables for GPU
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_NUM_GPU = "1"

# Start Ollama
Write-Host "GPU Device: $env:CUDA_VISIBLE_DEVICES" -ForegroundColor Cyan
Write-Host "Starting Ollama server..." -ForegroundColor Yellow
Write-Host ""

ollama serve
