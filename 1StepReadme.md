! حالا اسکلت کامل پروژه آماده شد. 🎉
چیزهایی که ساختیم:
✅ سرویس‌های اصلی:

API Gateway با FastAPI (پورت 8000)
Test Service برای تست (پورت 8001)
Nginx به عنوان Load Balancer (پورت 80)

✅ پایگاه داده:

PostgreSQL با جداول اولیه
Redis برای cache

✅ مانیتورینگ:

Prometheus (پورت 9090)
Grafana (پورت 3000)

✅ ابزارها:

Docker Compose کامل
اسکریپت راه‌اندازی خودکار
README جامع

🚀 حالا چطور اجرا کنی:

تمام فایل‌ها رو در یک پوشه قرار بده
Python اجرا کن:

bashpython scripts/setup.py

یا دستی راه‌اندازی کن:

bashdocker compose up -d

---

تست کن:

http://localhost - صفحه اصلی
http://localhost/health - بررسی سلامت
http://localhost:3000 - Grafana (admin/admin)

---------------------دانلود دستی مدل زبانی
mkdir -p data/models/llm

# دانلود با git-lfs (پیشنهادی)

cd data/models/llm
