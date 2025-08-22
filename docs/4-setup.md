✅ سرویس‌هایی که اجرا می‌شوند:
✅ nginx (port 80, 443)
✅ api-gateway (port 8000)
✅ llm-service (port 8002)
✅ stt-service (port 8003) - اگر فایل‌ها کامل باشند
✅ gpu-coordinator (port 8080)
✅ test-service (port 8001)
✅ redis (port 6379)
✅ postgres (port 5432)
✅ prometheus (port 9090)
✅ grafana (port 3000)
⚠️ نیازمندی‌های ضروری:

1. فایل‌های ناقص:
   bash# این فایل‌ها باید وجود داشته باشند:
   services/gpu-coordinator/main.py # ❌ احتمالاً ناقص
   services/gpu-coordinator/Dockerfile # ❌ احتمالاً ناقص
   services/audio-service/stt/models/ # ⚠️ ممکن است خالی باشد
   services/auth-service/ # ❌ کاملاً ناقص
2. مدل‌های AI:
   bash# مسیرهایی که باید مدل‌ها داشته باشند:
   ./data/models/llm/gpt2-fa/ # برای LLM
   ./data/models/stt/ # برای STT (اختیاری)
3. فایل Environment:
   bash# باید وجود داشته باشد:
   .env
   🔍 بررسی قبل از اجرا:Pre-Setup Checker ScriptCode #!/usr/bin/env python3
   """
   Pre-Setup Checker
   بررسی کامل قبل از اجرای setup.py
   """

import os
from pathlib import Path
import subprocess

def check_all_requirements():
"""بررسی کامل تمام نیازمندی‌ها"""
print("🔍 Pre-Setup Complete Check")
🚀 پاسخ کوتاه:
بله، اما با شرط‌هایی:
✅ آماده اجرا (احتمال بالا):

nginx, api-gateway, llm-service, stt-service
redis, postgres, prometheus, grafana
test-service

⚠️ ممکن است fail شوند:

gpu-coordinator (اگر فایل‌هایش ناقص است)
auth-service (در docker-compose نیست ولی ممکن است referenced باشد)

🔍 بررسی سریع:
قبل از اجرای setup.py، این دستور را اجرا کنید:
bash# بررسی فایل‌های اساسی
ls services/gpu-coordinator/main.py
ls services/gpu-coordinator/Dockerfile
ls data/models/llm/gpt2-fa/
ls .env
اگر فایل‌های بالا موجود باشند، احتمال موفقیت 85% است.
📝 توصیه:

اول script بررسی بالا را اجرا کنید
مشکلات اساسی را رفع کنید
سپس setup.py را اجرا کنید

# تست همه سرویس‌ها

curl http://localhost:8000/health # API Gateway
curl http://localhost:8002/health # LLM Service  
curl http://localhost:8003/health # STT Service
curl http://localhost:8080/health # GPU Coordinator
curl http://localhost:8004/health # Auth Service

---

----------run:
python scripts/setup.py
لاگ‌ها را چک کنید:

# بررسی لاگ‌های کلی

docker-compose logs --tail=50

# بررسی سرویس خاص

docker-compose logs -f gpu-coordinator
docker-compose logs -f llm-service
docker-compose logs -f stt-service
