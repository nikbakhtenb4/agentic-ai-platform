#!/bin/bash

# Quick Fix Script for LLM Service Issues
# This script addresses the immediate problems found in your test results

set -e  # Exit on any error

echo "🔧 LLM Service Quick Fix Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Check if services are running
print_step "Step 1: Checking service status"

if curl -s http://localhost:8002/health > /dev/null; then
    print_success "LLM Service is responding"
else
    print_error "LLM Service is not responding on port 8002"
    echo "Try: docker ps | grep llm-service"
    echo "If not running: docker-compose up llm-service -d"
fi

if curl -s http://localhost:8080/health > /dev/null; then
    print_success "GPU Coordinator is responding"
else
    print_warning "GPU Coordinator is not responding on port 8080"
    echo "This is optional, but if needed: docker-compose up gpu-coordinator -d"
fi

# Step 2: Run diagnostic script
print_step "Step 2: Running model path diagnostic"

# Create the diagnostic script
cat > /tmp/model_diagnostic.py << 'EOF'
# Insert the diagnostic script content here
import os
import sys
from pathlib import Path

def check_model_paths():
    print("🔍 Quick Model Path Check")
    print("=" * 30)
    
    model_path = os.getenv("MODEL_PATH", "/app/models/llm")
    model_name = os.getenv("MODEL_NAME", "gpt2-fa")
    
    possible_paths = [
        Path(model_path),
        Path(model_path) / model_name,
        Path("/app/models/llm/gpt2-fa"),
        Path("/app/models/llm"),
        Path("/app/models"),
    ]
    
    print(f"Looking for model: {model_name}")
    print(f"Base path: {model_path}")
    print()
    
    found_any = False
    
    for path in possible_paths:
        print(f"📁 Checking: {path}")
        if path.exists():
            if path.is_dir():
                try:
                    contents = [f.name for f in path.iterdir()]
                    print(f"   ✅ Directory exists ({len(contents)} items)")
                    
                    # Check for key files
                    has_config = (path / "config.json").exists()
                    has_model = any((path / f).exists() for f in ["pytorch_model.bin", "model.safetensors", "model.bin"])
                    has_tokenizer = any((path / f).exists() for f in ["tokenizer.json", "tokenizer_config.json", "vocab.txt"])
                    
                    if has_config:
                        print(f"   ✅ Has config.json")
                        found_any = True
                    if has_model:
                        print(f"   ✅ Has model files")
                    if has_tokenizer:
                        print(f"   ✅ Has tokenizer files")
                    
                    if has_config and has_model:
                        print(f"   🎯 VALID MODEL DIRECTORY!")
                        return str(path)
                    
                except Exception as e:
                    print(f"   ❌ Error reading directory: {e}")
            else:
                print(f"   ⚠️  Path exists but is not a directory")
        else:
            print(f"   ❌ Path does not exist")
        print()
    
    if not found_any:
        print("❌ No valid model directories found!")
        print("\n💡 Recommendations:")
        print("1. Check if your model files are properly mounted")
        print("2. Verify the MODEL_PATH environment variable")
        print("3. Consider using fallback model download")
        print("4. Check Docker volume mounts in docker-compose.yml")
    
    return None

if __name__ == "__main__":
    check_model_paths()
EOF

python3 /tmp/model_diagnostic.py

# Step 3: Check and fix missing endpoint
print_step "Step 3: Checking for missing /gpu/status endpoint"

response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/gpu/status)

if [ "$response" = "404" ]; then
    print_error "GPU status endpoint is missing (404)"
    print_warning "This indicates the endpoint is not implemented in main.py"
    echo "The main.py needs to be updated with the enhanced version"
else
    print_success "GPU status endpoint exists (HTTP $response)"
fi

# Step 4: Test model loading
print_step "Step 4: Testing model info endpoint"

model_info=$(curl -s http://localhost:8002/model/info)
echo "Model info response:"
echo "$model_info" | python3 -m json.tool 2>/dev/null || echo "$model_info"

# Step 5: Check if model is actually loaded
model_loaded=$(echo "$model_info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('is_loaded', False))" 2>/dev/null || echo "false")

if [ "$model_loaded" = "True" ] || [ "$model_loaded" = "true" ]; then
    print_success "Model is loaded"
else
    print_error "Model is not loaded"
    
    # Try to get more details
    print_step "Getting detailed model parameters..."
    echo "$model_info" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    params = data.get('parameters', {})
    status = params.get('status', 'unknown')
    total_params = params.get('total_parameters', 0)
    error = params.get('error', 'No error info')
    print(f'Status: {status}')
    print(f'Parameters: {total_params}')
    if 'error' in params:
        print(f'Error: {error}')
except:
    print('Could not parse model info')
"
fi

# Step 6: Try a simple generation test
print_step "Step 6: Testing text generation"

if [ "$model_loaded" = "True" ] || [ "$model_loaded" = "true" ]; then
    echo "Testing generation with a simple prompt..."
    
    generation_response=$(curl -s -X POST http://localhost:8002/generate \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello", "max_length": 20, "temperature": 0.7}')
    
    echo "Generation response:"
    echo "$generation_response" | python3 -m json.tool 2>/dev/null || echo "$generation_response"
else
    print_warning "Skipping generation test - model not loaded"
fi

# Step 7: Provide specific fixes
print_step "Step 7: Recommended fixes"

echo ""
echo "🔧 IMMEDIATE FIXES NEEDED:"
echo ""

# Fix 1: Missing endpoint
if [ "$response" = "404" ]; then
    echo "1. 📝 UPDATE MAIN.PY:"
    echo "   The /gpu/status endpoint is missing from your main.py"
    echo "   You need to replace your main.py with the enhanced version"
    echo ""
fi

# Fix 2: Model loading
if [ "$model_loaded" != "True" ] && [ "$model_loaded" != "true" ]; then
    echo "2. 📚 FIX MODEL LOADING:"
    echo "   Your model is not loading properly"
    echo ""
    echo "   Quick fixes to try:"
    echo "   a) Check if model files exist:"
    echo "      docker exec -it agentic-llm-service ls -la /app/models/llm/"
    echo ""
    echo "   b) Check model path in container:"
    echo "      docker exec -it agentic-llm-service find /app -name '*.bin' -o -name '*.safetensors' | head -10"
    echo ""
    echo "   c) Check environment variables:"
    echo "      docker exec -it agentic-llm-service env | grep MODEL"
    echo ""
    echo "   d) Try restarting with fallback model download:"
    echo "      docker-compose down llm-service"
    echo "      docker-compose up llm-service -d"
    echo "      docker logs -f agentic-llm-service"
    echo ""
fi

# Fix 3: Docker setup
echo "3. 🐳 DOCKER CONFIGURATION:"
echo "   Ensure your docker-compose.yml has proper volume mounts:"
echo ""
echo "   llm-service:"
echo "     volumes:"
echo "       - ./models:/app/models  # Make sure this path exists"
echo "     environment:"
echo "       - MODEL_PATH=/app/models/llm"
echo "       - MODEL_NAME=gpt2-fa"
echo ""

# Step 8: Create helper scripts
print_step "Step 8: Creating helper scripts"

# Create a quick test script
cat > /tmp/test_llm_quick.sh << 'TESTEOF'
#!/bin/bash

echo "🧪 Quick LLM Service Test"
echo "========================"

echo "1. Health check:"
curl -s http://localhost:8002/health | python3 -m json.tool

echo -e "\n2. Model info:"
curl -s http://localhost:8002/model/info | python3 -m json.tool

echo -e "\n3. GPU status (if available):"
curl -s http://localhost:8002/gpu/status | python3 -m json.tool 2>/dev/null || echo "Endpoint not available"

echo -e "\n4. Simple generation test:"
curl -s -X POST http://localhost:8002/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "Test", "max_length": 10}' | python3 -m json.tool

echo -e "\n5. Metrics sample:"
curl -s http://localhost:8002/metrics | grep llm | head -5
TESTEOF

chmod +x /tmp/test_llm_quick.sh
print_success "Created quick test script: /tmp/test_llm_quick.sh"

# Create debug script
cat > /tmp/debug_model_paths.sh << 'DEBUGEOF'
#!/bin/bash

echo "🔍 Model Path Debug Script"
echo "=========================="

echo "Current environment:"
echo "MODEL_PATH: ${MODEL_PATH:-/app/models/llm}"
echo "MODEL_NAME: ${MODEL_NAME:-gpt2-fa}"
echo ""

echo "Checking inside container:"
if docker ps | grep -q llm-service; then
    container_name=$(docker ps --format "table {{.Names}}" | grep llm-service | head -1)
    echo "Container: $container_name"
    echo ""
    
    echo "Environment variables in container:"
    docker exec "$container_name" env | grep -E "(MODEL|PATH)" || echo "No MODEL variables found"
    echo ""
    
    echo "File system check:"
    docker exec "$container_name" ls -la /app/models/ 2>/dev/null || echo "/app/models/ not found"
    docker exec "$container_name" ls -la /app/models/llm/ 2>/dev/null || echo "/app/models/llm/ not found"
    docker exec "$container_name" ls -la /app/models/llm/gpt2-fa/ 2>/dev/null || echo "/app/models/llm/gpt2-fa/ not found"
    echo ""
    
    echo "Search for model files:"
    docker exec "$container_name" find /app -name "*.bin" -o -name "*.safetensors" -o -name "config.json" 2>/dev/null | head -10
    echo ""
    
    echo "Python path check:"
    docker exec "$container_name" python3 -c "
import sys
print('Python path:')
for p in sys.path[:5]:
    print(f'  {p}')
"
    echo ""
    
    echo "Try importing transformers:"
    docker exec "$container_name" python3 -c "
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('✅ Transformers imported successfully')
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
except Exception as e:
    print(f'❌ Import error: {e}')
"
else
    echo "❌ LLM service container not running"
    echo "Start it with: docker-compose up llm-service -d"
fi
DEBUGEOF

chmod +x /tmp/debug_model_paths.sh
print_success "Created debug script: /tmp/debug_model_paths.sh"

# Step 9: Final recommendations
print_step "Step 9: Next steps"

echo ""
echo "🎯 PRIORITIZED ACTION PLAN:"
echo ""
echo "1. 🔧 IMMEDIATE (Fix code issues):"
echo "   - Replace main.py with the enhanced version (fixes 404 error)"
echo "   - Replace loader.py with enhanced version (better model loading)"
echo ""
echo "2. 📁 MODEL SETUP (Fix model loading):"
echo "   - Run debug script: /tmp/debug_model_paths.sh"
echo "   - Ensure model files are in correct location"
echo "   - Or allow fallback model download"
echo ""
echo "3. 🐳 DOCKER (Fix configuration):"
echo "   - Check volume mounts in docker-compose.yml"
echo "   - Restart services: docker-compose restart llm-service"
echo ""
echo "4. ✅ VERIFY (Test everything):"
echo "   - Run: /tmp/test_llm_quick.sh"
echo "   - Run full test: python3 test_llm_gpu.py"
echo ""

print_step "Useful commands to run now"

echo ""
echo "📋 Copy and run these commands:"
echo ""
echo "# Check what's in your container:"
echo "/tmp/debug_model_paths.sh"
echo ""
echo "# Quick service test:"
echo "/tmp/test_llm_quick.sh"
echo ""
echo "# Check service logs:"
echo "docker logs agentic-llm-service --tail 50"
echo ""
echo "# Restart service (if needed):"
echo "docker-compose restart llm-service"
echo ""
echo "# Follow logs during restart:"
echo "docker logs -f agentic-llm-service"
echo ""

print_success "Quick fix script completed!"
print_warning "The main issues are code-related. You need to update main.py and loader.py"
print_warning "Then address the model loading issues based on the debug output."

echo ""
echo "🔗 Summary of issues found:"
echo "1. Missing /gpu/status endpoint (404) - Code fix needed"
echo "2. Model not loading properly - Path/file issue"
echo "3. Generation failing due to #2"
echo ""
echo "💡 The enhanced code files provided above will fix these issues!"
echo "   Replace your main.py and loader.py with the enhanced versions."