#!/bin/bash

echo "🐳 Building GPU Coordinator Docker image..."

# Build the Docker image
docker build -t gpu-coordinator:latest .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 You can run the container with:"
    echo "   docker run -p 8080:8080 gpu-coordinator:latest"
else
    echo "❌ Build failed!"
    echo "🔧 Try using the robust Dockerfile:"
    echo "   docker build -f Dockerfile.robust -t gpu-coordinator:latest ."
fi