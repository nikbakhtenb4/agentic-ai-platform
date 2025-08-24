ساخت ایمیج
docker-compose build --no-cache llm-service
اجرا کانتینر
docker-compose up -d llm-service

کانتینرهای در حال اجزا:
docker ps

ریستارت:
docker-compose restart llm-service

لاگ:
docker logs -f agentic-llm-service

Stop همه کانتینرهای در حال اجرا
docker stop $(docker ps -q)

لاگ گیری:
docker compose logs -f stt-service

docker compose logs api-gateway --tail=20

docker compose logs gpu-coordinator --tail=20

# بررسی لاگ STT Service

docker compose logs stt-service --tail=30

# شاید stuck شده در کجایی

---

مرحله 1: آزاد کردن GPU

# Force release expired task

curl -X POST https://localhost:8080/release/task_0_1756073519

# چک کردن که آزاد شد

curl https://localhost:8080/tasks

تست stt برای gpu sharing کار میکنه

# تست درخواست جدید

curl -X POST https://localhost:8080/request -H "Content-Type: application/json" -d "{\"service_name\": \"test\", \"estimated_memory\": 1.0, \"priority\": \"normal\", \"timeout\": 300}"

# باید بگوید allocated: true

# Restart GPU Coordinator

docker compose restart gpu-coordinator

# صبر 30 ثانیه

sleep 30

# Test دوباره

curl http://localhost:8080/tasks
