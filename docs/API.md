# تست STT Service

curl http://localhost:8003/health
curl http://localhost:8003/model/info

# تست LLM Service

curl http://localhost:8002/health
curl http://localhost:8002/model/info

# تست GPU Coordinator

curl http://localhost:8080/health
curl http://localhost:8080/status

# تست API Gateway

curl http://localhost:8000/health

# تست Main Platform

curl http://localhost/health

# آیا LLM می‌تواند GPU بگیرد؟

curl -X POST http://localhost:8080/request -H "Content-Type: application/json" -d "{\"service_name\": \"llm-test\", \"estimated_memory\": 3.0, \"priority\": \"normal\", \"timeout\": 300}"
