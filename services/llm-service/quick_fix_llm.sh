#!/bin/bash
# LLM Service Debug Script
# اسکریپت تشخیص عیب سرویس LLM

echo "🔍 LLM Service Diagnostic Script"
echo "=================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    case $2 in
        "SUCCESS") echo -e "${GREEN}✅ $1${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $1${NC}" ;;
        "ERROR") echo -e "${RED}❌ $1${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $1${NC}" ;;
        *) echo "$1" ;;
    esac
}

# Step 1: Check container status
print_status "Checking LLM Service container..." "INFO"
if docker ps | grep -q llm-service; then
    print_status "LLM Service container is running" "SUCCESS"
    
    # Get container stats
    echo "📊 Container Resource Usage:"
    docker stats llm-service --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
else
    print_status "LLM Service container is not running!" "ERROR"
    echo "Available containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}"
    exit 1
fi

echo ""

# Step 2: Check recent logs
print_status "Checking recent logs..." "INFO"
echo "📝 Last 20 log entries:"
docker logs llm-service --tail 20

echo ""

# Step 3: Check model files
print_status "Checking model files..." "INFO"
echo "📁 Model directory contents:"
docker exec llm-service ls -la /app/models/ 2>/dev/null || echo "Cannot access model directory"

echo "🔍 Looking for model files:"
docker exec llm-service find /app/models -name "*.json" -o -name "*.bin" -o -name "*.safetensors" 2>/dev/null | head -10

echo ""

# Step 4: Check environment variables
print_status "Checking environment variables..." "INFO"
echo "🌍 Relevant environment variables:"
docker exec llm-service env | grep -E "(MODEL|GPU|CUDA)" || echo "No relevant env vars found"

echo ""

# Step 5: Test Python imports
print_status "Testing Python imports..." "INFO"
docker exec llm-service python3 -c "
try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'🔥 CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'🎮 GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
except Exception as e:
    print(f'❌ PyTorch error: {e}')

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('✅ Transformers imported successfully')
except Exception as e:
    print(f'❌ Transformers error: {e}')

try:
    import sys
    sys.path.append('/app')
    from models.loader import ModelLoader
    print('✅ ModelLoader imported successfully')
    
    loader = ModelLoader()
    print(f'📍 Model path: {loader.model_path}')
    print(f'🏷️ Model name: {loader.model_name}')
    print(f'🔧 Device: {loader.device}')
    print(f'📦 Is loaded: {loader.is_loaded}')
    
except Exception as e:
    print(f'❌ ModelLoader error: {e}')
" 2>/dev/null

echo ""

# Step 6: Test Health Endpoint
print_status "Testing health endpoint..." "INFO"
health_response=$(curl -s http://localhost:8002/health)
if [ $? -eq 0 ]; then
    print_status "Health endpoint accessible" "SUCCESS"
    echo "📊 Health Response:"
    echo "$health_response" | python3 -m json.tool 2>/dev/null || echo "$health_response"
else
    print_status "Health endpoint not accessible" "ERROR"
fi

echo ""

# Step 7: Test model loading manually
print_status "Testing manual model loading..." "INFO"
docker exec llm-service python3 -c "
import asyncio
import sys
sys.path.append('/app')
from models.loader import ModelLoader

async def test_loading():
    try:
        loader = ModelLoader()
        print(f'🔄 Attempting to load model from: {loader.model_path}/{loader.model_name}')
        await loader.initialize()
        print('✅ Model loaded successfully!')
        
        model_info = loader.get_model_info()
        print(f'📊 Model info: {model_info}')
        
    except Exception as e:
        print(f'❌ Model loading failed: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(test_loading())
" 2>/dev/null

echo ""

# Step 8: Test API with correct payload
print_status "Testing generation API with correct payload..." "INFO"
api_response=$(curl -s -X POST https://localhost:8002/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "سلام",
        "max_length": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": true,
        "num_return_sequences": 1
    }')

if [ $? -eq 0 ]; then
    print_status "Generation API accessible" "SUCCESS"
    echo "🤖 API Response:"
    echo "$api_response" | python3 -m json.tool 2>/dev/null || echo "$api_response"
else
    print_status "Generation API not accessible" "ERROR"
fi

echo ""

# Step 9: Check Docker image layers
print_status "Checking Docker image info..." "INFO"
echo "🐳 Image information:"
docker image inspect llm-service:latest | grep -A 10 -B 5 "Config" 2>/dev/null | head -15

echo ""

# Step 10: Network connectivity test
print_status "Testing network connectivity..." "INFO"
echo "🌐 Port connectivity:"
netstat -tlnp | grep :8002 || echo "Port 8002 not listening"

echo ""

# Summary
echo "=================================================="
print_status "Diagnostic Summary" "INFO"
echo "=================================================="

# Check if model is loaded
if echo "$health_response" | grep -q '"model_loaded":true'; then
    print_status "Model Status: LOADED" "SUCCESS"
elif echo "$health_response" | grep -q '"model_loaded":false'; then
    print_status "Model Status: NOT LOADED" "WARNING"
else
    print_status "Model Status: UNKNOWN" "ERROR"
fi

# Check GPU availability
if echo "$health_response" | grep -q '"gpu_available":true'; then
    print_status "GPU Status: AVAILABLE" "SUCCESS"
else
    print_status "GPU Status: NOT AVAILABLE" "WARNING"
fi

# Check API accessibility
if [ -n "$api_response" ] && ! echo "$api_response" | grep -q "error"; then
    print_status "API Status: WORKING" "SUCCESS"
else
    print_status "API Status: ISSUES DETECTED" "WARNING"
fi

echo ""
echo "🔧 Next steps:"
echo "1. If model not loaded: Check model files and paths"
echo "2. If API issues: Check payload format and endpoints"
echo "3. If GPU issues: Check CUDA installation and drivers"
echo "4. Check container logs for detailed error messages"

echo ""
echo "📋 Quick fixes:"
echo "# Restart container:"
echo "docker restart llm-service"
echo ""
echo "# Check detailed logs:"
echo "docker logs llm-service -f"
echo ""
echo "# Test API directly:"
echo "curl -X POST https://localhost:8002/generate -H 'Content-Type: application/json' -d '{\"prompt\":\"سلام\",\"max_length\":50}'"