(base) PS D:\Project\Server-OpenAI\agentic-ai-platform> python scripts/setup.py
✅ Windows encoding configured for UTF-8

    ╔══════════════════════════════════════╗
    ║        Agentic AI Platform           ║
    ║   LLM + STT + GPU Sharing Setup     ║
    ║       با بهینه‌سازی Cache              ║
    ╚══════════════════════════════════════╝

🔧 Loading environment variables...
✅ Environment variables loaded successfully
🔍 Checking requirements...
✅ Docker: Docker version 27.2.0, build 3ab4256
✅ Docker Compose: Docker Compose version v2.29.2-desktop.2
✅ Docker daemon is running
🎮 Checking GPU support and CUDA configuration...
✅ NVIDIA GPU detected
💾 GPU Memory: | N/A 49C P0 7W / 74W | 0MiB / 4096MiB | 0% Default |
✅ Docker GPU runtime available
🚀 GPU acceleration will be used for both LLM and STT services
🔄 GPU Coordinator will manage resource sharing
💾 Checking system resources...
🐏 Total RAM: 31.7 GB
🐏 Available RAM: 12.3 GB
✅ Excellent RAM for both LLM and STT processing
🔧 CPU cores: 14 physical, 20 logical
⚠️ Could not check disk space: argument 1 (impossible<bad format char>)
🤖 Checking LLM model...
✅ Found 4 model files: config.json, pytorch_model.bin, tokenizer.json, vocab.json
✅ All essential LLM model files found
📦 LLM Model size: 467.1 MB
🎙️ Checking STT model configuration...
✅ Found 2 Whisper model files
📁 large-v3.pt: 2944.3 MB
📁 medium.pt: 1457.2 MB
🔧 Configured Whisper model size: medium
📊 Estimated download size: ~769MB
📊 Max file size: 100MB
🌐 Supported languages: fa,en
📁 Creating required directories...
✅ Created 18/18 directories
🔧 Validating environment configuration...
🔍 Environment variables status:
✅ MODEL_PATH: /app/models/llm/gpt2-fa
✅ LLM_SERVICE_PORT: 8002
✅ STT_SERVICE_PORT: 8003
✅ GPU_COORDINATOR_PORT: 8080
✅ WHISPER_MODEL_SIZE: medium
✅ CUDA_VISIBLE_DEVICES: 0
✅ MAX_MEMORY_MB: 6144
⚠️ DOCKER_BUILDKIT: not set, using default (1)
⚠️ COMPOSE_DOCKER_CLI_BUILD: not set, using default (1)
⚠️ Some environment variables were missing - defaults applied

🖥️ Platform: Windows 11
🪟 Windows-specific optimizations applied

🚀 Cache Optimization Features:
────────────────────────────────────────────────────────────
✅ Docker BuildKit enabled for faster builds
✅ Multi-stage build caching
✅ Layer caching for base images
✅ Image tagging for better cache reuse
✅ Pre-pulling base images

📊 Cache Commands:
docker system df # Show cache usage
docker builder prune # Clear build cache
docker image ls # List cached images
docker buildx du # BuildKit cache usage
────────────────────────────────────────────────────────────
🏗️ Building services with optimized cache usage...
🔍 Checking existing Docker images for cache optimization...
📦 No existing images found - first time build
🔧 Enabling Docker BuildKit for better cache performance...
✅ Docker BuildKit enabled
📈 This improves build performance and cache efficiency
📥 Pre-pulling base images for better cache performance...
📦 Pulling python:3.11-slim...
📥 3.11-slim: Pulling from library/python...
✅ python:3.11-slim pulled successfully
📦 Pulling pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime...
📥 2.0.1-cuda11.7-cudnn8-runtime: Pulling from pytorch/pytorch...
📥 4ade0a4bc5d5: Pulling fs layer...
📥 4f4fb700ef54: Pulling fs layer...
📥 035a286326d6: Pulling fs layer...
📥 2185b402c9ca: Pulling fs layer...
📥 99803d4b97f3: Pulling fs layer...
