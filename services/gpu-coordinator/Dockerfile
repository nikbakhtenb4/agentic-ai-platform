# gpu-coordinator/DockerFile
FROM python:3.11-slim

# 🧰 نصب ابزارهای ضروری بدون دستکاری mirror
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gcc \
        g++ \
        python3-dev \
        pkg-config && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 🏗 ست کردن پوشه کاری
WORKDIR /app

# 📦 کپی requirements فقط برای کش کردن بهتر
COPY services/gpu-coordinator/requirements.txt .

# 📦 نصب torch (اگر wheel داری، خط زیر رو نگه دار، وگرنه حذفش کن یا از pip نصب کن)
# RUN pip install --no-cache-dir /wheels/torch-2.7.1*.whl

# ✅ نصب پکیج‌ها از PyPI اصلی
RUN pip install --no-cache-dir --prefer-binary \
        --default-timeout=300 --retries=10 \
        -r requirements.txt

# 📁 کپی کد برنامه
# COPY services/gpu-coordinator /app
COPY . /app/
# ♻️ کپی ماژول‌های مشترک
COPY shared /app/shared

# 📚 تنظیم PYTHONPATH
ENV PYTHONPATH="/app:/app/shared"

# ⚙️ تنظیم متغیرهای محیطی
ENV HOST=0.0.0.0
ENV PORT=8080
ENV LOG_LEVEL=info

# 🩺 healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 🌐 باز کردن پورت
EXPOSE 8080

# 🚀 اجرای اپلیکیشن
CMD ["python", "main.py"]
