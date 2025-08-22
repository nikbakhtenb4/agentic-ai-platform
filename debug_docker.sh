#!/bin/bash

# Debug Docker Container for LLM Service
# این اسکریپت برای دیباگ مشکلات Docker Container استفاده می‌شود

set -e

echo "🐳 Docker Container Debug Script"
echo "================================"

# Function to check if container exists and is running
check_container_status() {
    local container_name="agentic-llm-service"
    
    echo "📋 Checking container status..."
    
    if docker ps -a --format "table {{.Names}}\t{{.Status}}" | grep -q "$container_name"; then
        echo "✅ Container $container_name found"
        docker ps -a --filter "name=$container_name"
    else
        echo "❌ Container $container_name not found"
        return 1
    fi
}

# Function to check container logs
check_container_logs() {
    local container_name="agentic-llm-service"
    
    echo "📝 Checking container logs..."
    echo "Latest 50 lines:"
    echo "================================"
    
    if docker logs --tail 50 "$container_name" 2>&1; then
        echo "✅ Logs retrieved successfully"
    else
        echo "❌ Failed to retrieve logs"
        return 1
    fi
}

# Function to inspect container
inspect_container() {
    local container_name="agentic-llm-service"
    
    echo "🔍 Inspecting container..."
    
    echo "Environment variables:"
    docker exec "$container_name" env | grep -E "(PYTHON|MODEL|PATH)" || true
    
    echo "Python path:"
    docker exec "$container_name" python -c "import sys; print('\\n'.join(sys.path))" || true
    
    echo "Working directory:"
    docker exec "$container_name" pwd || true
    
    echo "Files in /app:"
    docker exec "$container_name" ls -la /app/ || true
    
    echo "Files in /app/models:"
    docker exec "$container_name" ls -la /app/models/ || true
    
    echo "Files in /app/utils:"
    docker exec "$container_name" ls -la /app/utils/ || true
}

# Function to test imports inside container
test_imports_in_container() {
    local container_name="agentic-llm-service"
    
    echo "🧪 Testing imports inside container..."
    
    # Test Python import
    echo "Testing Python imports:"
    docker exec "$container_name" python -c "
import sys
print('Python version:', sys.version)
print('Python path:', sys.path)

try:
    import models.loader
    print('✅ models.loader import successful')
except ImportError as e:
    print('❌ models.loader import failed:', e)

try:
    import models.text_generation
    print('✅ models.text_generation import successful')
except ImportError as e:
    print('❌ models.text_generation import failed:', e)

try:
    import utils.gpu_manager
    print('✅ utils.gpu_manager import successful')
except ImportError as e:
    print('❌ utils.gpu_manager import failed:', e)
" || true
}

# Function to restart container with better debugging
restart_with_debug() {
    local container_name="agentic-llm-service"
    
    echo "🔄 Restarting container with debug mode..."
    
    # Stop existing container
    docker stop "$container_name" 2>/dev/null || true
    docker rm "$container_name" 2>/dev/null || true
    
    # Start with interactive mode for debugging
    echo "Starting container in debug mode..."
    docker run -it --name "${container_name}-debug" \
        -p 8002:8002 \
        -e ENV=development \
        -e LOG_LEVEL=DEBUG \
        -e PYTHONPATH=/app:/app/models:/app/utils \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/utils:/app/utils" \
        --entrypoint /bin/bash \
        agentic-llm-service
}

# Function to rebuild container
rebuild_container() {
    echo "🔨 Rebuilding LLM service container..."
    
    # Remove existing container and image
    docker stop agentic-llm-service 2>/dev/null || true
    docker rm agentic-llm-service 2>/dev/null || true
    docker rmi agentic-llm-service 2>/dev/null || true
    
    # Rebuild
    cd "$(dirname "$0")"
    docker build -t agentic-llm-service -f services/llm-service/Dockerfile services/llm-service/
    
    echo "✅ Container rebuilt successfully"
}

# Function to run quick tests
run_quick_tests() {
    echo "⚡ Running quick tests..."
    
    # Check if required files exist locally
    echo "Checking local files:"
    
    files=(
        "services/llm-service/main.py"
        "services/llm-service/models/loader.py"
        "services/llm-service/models/text_generation.py"
        "services/llm-service/utils/gpu_manager.py"
        "services/llm-service/requirements.txt"
    )
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            echo "✅ $file exists"
        else
            echo "❌ $file missing"
        fi
    done
    
    # Check Docker image
    echo "Checking Docker image:"
    if docker images | grep -q "agentic-llm-service"; then
        echo "✅ Docker image exists"
    else
        echo "❌ Docker image not found"
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "Choose an option:"
    echo "1. Check container status"
    echo "2. Check container logs"
    echo "3. Inspect container"
    echo "4. Test imports in container"
    echo "5. Restart with debug mode"
    echo "6. Rebuild container"
    echo "7. Run quick tests"
    echo "8. Full diagnostic"
    echo "9. Exit"
    echo ""
}

# Full diagnostic
full_diagnostic() {
    echo "🩺 Running full diagnostic..."
    echo "================================"
    
    run_quick_tests
    echo ""
    check_container_status
    echo ""
    check_container_logs
    echo ""
    inspect_container
    echo ""
    test_imports_in_container
    
    echo "🩺 Full diagnostic completed"
}

# Main script
main() {
    while true; do
        show_menu
        read -p "Enter your choice (1-9): " choice
        
        case $choice in
            1) check_container_status ;;
            2) check_container_logs ;;
            3) inspect_container ;;
            4) test_imports_in_container ;;
            5) restart_with_debug ;;
            6) rebuild_container ;;
            7) run_quick_tests ;;
            8) full_diagnostic ;;
            9) echo "👋 Goodbye!"; exit 0 ;;
            *) echo "❌ Invalid option. Please choose 1-9." ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi