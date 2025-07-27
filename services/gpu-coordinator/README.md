# GPU Coordinator Service

## Overview
The GPU Coordinator Service manages GPU resources and coordinates model deployment across available GPUs in the agentic AI platform.

## Features
- GPU resource monitoring
- Model deployment coordination
- Health checking
- Resource allocation management

## API Endpoints

### Health Check
- **GET** `/health` - Service health status
- **GET** `/` - Root endpoint with service information
- **GET** `/gpu/status` - GPU status and resource information

## Environment Variables
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8080)
- `LOG_LEVEL` - Logging level (default: info)
- `REDIS_URL` - Redis connection URL

## Docker Usage

### Build the image
```bash
cd services/gpu-coordinator
chmod +x build.sh
./build.sh
```

### Run with Docker
```bash
docker run -p 8080:8080 gpu-coordinator:latest
```

### Run with Docker Compose
```bash
# From project root
docker-compose up gpu-coordinator
```

## Development

### Local Development
```bash
cd services/gpu-coordinator
pip install -r requirements.txt
python main.py
```

### Testing
```bash
curl http://localhost:8080/health
curl http://localhost:8080/gpu/status
```

## Troubleshooting

### Build Issues
If you encounter repository issues during Docker build, try:
1. Use the robust Dockerfile: `docker build -f Dockerfile.robust -t gpu-coordinator:latest .`
2. Check your internet connection
3. Try building without cache: `docker build --no-cache -t gpu-coordinator:latest .`

### Common Issues
- **503 Service Unavailable**: Repository mirrors may be down, try using default Dockerfile
- **Package installation fails**: Check network connectivity and retry
- **CUDA/GPU detection**: Ensure proper GPU drivers are installed