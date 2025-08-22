# راهنمای اجرای Agentic AI Platform

# 1. اجرای setup.py

echo "🚀 اجرای setup script..."
python scripts/setup.py

# اگر setup.py موجود نیست، مستقیماً Docker Compose اجرا کنید:

echo "🐳 اجرای مستقیم Docker Compose..."
docker-compose up -d

# 2. بررسی وضعیت سرویس‌ها

echo "📊 بررسی وضعیت کانتینرها..."
docker-compose ps

# 3. مشاهده لاگ‌ها

echo "📋 مشاهده لاگ‌های سرویس‌ها:"
echo "همه سرویس‌ها:"
docker-compose logs -f

echo "سرویس خاص (مثال):"
docker-compose logs -f api-gateway

# 4. تست سرویس‌ها

echo "🧪 تست سرویس‌ها:"
echo "API Gateway Health Check:"
curl http://localhost:8000/health

echo "GPU Coordinator Status:"
curl http://localhost:8080/status

echo "LLM Service Health:"
curl http://localhost:8002/health

echo "STT Service Health:"
curl http://localhost:8003/health

# 5. دسترسی به وب اینترفیس‌ها

echo "🌐 دسترسی به وب اینترفیس‌ها:"
echo "• API Gateway: http://localhost:8000"
echo "• GPU Coordinator: http://localhost:8080"
echo "• Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "• Prometheus: http://localhost:9090"

# 6. دستورات مفید

echo "🔧 دستورات مفید:"
echo "متوقف کردن همه سرویس‌ها:"
echo "docker-compose down"

echo "ری‌استارت سرویس خاص:"
echo "docker-compose restart api-gateway"

echo "rebuild کردن سرویس:"
echo "docker-compose up -d --build api-gateway"

echo "پاک کردن volumes (احتیاط!):"
echo "docker-compose down -v"

---

# چک‌لیست پس از راه‌اندازی Agentic AI Platform

## 1. بررسی وضعیت کانتینرها

```bash
docker-compose ps
```

**انتظار:** همه سرویس‌ها باید در وضعیت `Up` باشند

## 2. تست Health Check سرویس‌ها

### API Gateway

```bash
curl http://localhost:8000/health
```

**انتظار:** `{"status": "healthy"}`

### GPU Coordinator

```bash
curl http://localhost:8080/health
```

**انتظار:** `{"status": "healthy", "gpu_available": true/false}`

### LLM Service

```bash
curl http://localhost:8002/health
```

**انتظار:** `{"status": "healthy", "model_loaded": true}`

### STT Service

```bash
curl http://localhost:8003/health
```

**انتظار:** `{"status": "healthy", "whisper_ready": true}`

## 3. بررسی لاگ‌ها برای خطاها

```bash
# بررسی لاگ‌های کلی
docker-compose logs --tail=50

# بررسی سرویس خاص
docker-compose logs -f gpu-coordinator
docker-compose logs -f llm-service
docker-compose logs -f stt-service
```

## 4. تست عملکرد GPU Coordinator

```bash
# دریافت وضعیت GPU
curl http://localhost:8080/status

# دریافت آمار صف
curl http://localhost:8080/queue

# دریافت آمار کلی
curl http://localhost:8080/stats
```

## 5. تست اتصال به دیتابیس

```bash
# اتصال به PostgreSQL
docker exec -it agentic-postgres psql -U postgres -d agentic_db

# تست Redis
docker exec -it agentic-redis redis-cli ping
```

## 6. دسترسی به Dashboard ها

- **Grafana:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090
- **API Documentation:** http://localhost:8000/docs

## 7. تست عملکرد End-to-End

### تست LLM Service

```bash
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "سلام، چطوری؟", "max_length": 50}'
```

### تست STT Service (با فایل صوتی)

```bash
curl -X POST http://localhost:8003/transcribe \
  -F "file=@path/to/audio.wav" \
  -F "language=fa"
```

## مشکلات محتمل و راه‌حل‌ها

### اگر GPU Coordinator خطا می‌دهد:

- مطمئن شوید NVIDIA drivers نصب است
- در صورت عدم وجود GPU، سرویس باید در حالت CPU کار کند

### اگر LLM Service بالا نمی‌آید:

- بررسی کنید مدل GPT2-FA در مسیر درست است
- حافظه کافی برای مدل موجود باشد

### اگر STT Service مشکل دارد:

- Whisper model به صورت خودکار دانلود می‌شود
- اتصال اینترنت برای دانلود اولیه لازم است

## نکات مهم

1. **اولین اجرا:** ممکن است چند دقیقه طول بکشد تا مدل‌ها دانلود شوند
2. **GPU Memory:** اگر GPU memory کم است، `MAX_MEMORY_MB` را در .env کاهش دهید
3. **Performance:** برای بهترین عملکرد، `WHISPER_MODEL_SIZE=base` استفاده کنید

## دستورات اضطراری

```bash
# ری‌استارت کامل
docker-compose down && docker-compose up -d

# پاک کردن cache و rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d

# بررسی استفاده منابع
docker stats
```

C:\Users\Elham>curl -X POST http://localhost:8002/generate -H "Content-Type: application/json" -d "{\"text\": \"چیک شعر عاشقانه چهار بیتی در سبک حافظ بنویس\", \"max_length\": 400, \"temperature\": 0.7}"
{"detail":"Model not loaded"}
