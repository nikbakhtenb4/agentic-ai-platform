# Multi-stage build برای GPU Coordinator
FROM python:3.10-slim as base

# تنظیم environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# نصب basic system dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# کپی requirements و نصب Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ایجاد non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# کپی کد برنامه
COPY . .

# ایجاد directory های لازم
RUN mkdir -p /app/logs /app/shared \
    && chown -R appuser:appuser /app

# تغییر به non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f https://localhost:8080/health || exit 1

# اجرای برنامه
CMD ["python", "main.py"]