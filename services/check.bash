#!/bin/bash
# اسکریپت دیباگ سریع

echo "=== Container Status ==="
docker compose ps

echo -e "\n=== Failed Containers Logs ==="
echo "--- GPU Coordinator ---"
docker compose logs --tail=20 gpu-coordinator

echo -e "\n--- LLM Service ---"  
docker compose logs --tail=20 llm-service

echo -e "\n--- API Gateway ---"
docker compose logs --tail=20 api-gateway

echo -e "\n=== Resource Usage ==="
docker stats --no-stream

echo -e "\n=== Port Check ==="
netstat -an | findstr "8000 8002 8080"

echo -e "\n=== System Info ==="
docker info | findstr -i "memory\|cpu\|error"

echo -e "\n=== Quick Fix Commands ==="
echo "1. Restart failed services:"
echo "   docker compose restart gpu-coordinator llm-service api-gateway"
echo ""
echo "2. Rebuild services:"
echo "   docker compose build gpu-coordinator llm-service --no-cache"
echo ""
echo "3. Check individual service:"
echo "   docker compose up gpu-coordinator"