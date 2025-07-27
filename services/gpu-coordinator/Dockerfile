FROM python:3.11-slim

# 🛠 Install system dependencies using default repositories first
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gcc \
        g++ \
        python3-dev \
        pkg-config && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 🏗 Set working directory
WORKDIR /app

# 📦 Copy only requirements first for caching
COPY requirements.txt .

# ⚙️ Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary \
        --default-timeout=300 --retries=20 \
        -i https://pypi.org/simple \
        -r requirements.txt

# 📁 Copy the app source code
COPY . /app

# ♻️ Copy shared module
COPY ../../shared /app/shared

# 📚 Set PYTHONPATH to include shared modules
ENV PYTHONPATH="/app:/app/shared"

# ⚙️ Environment variables
ENV HOST=0.0.0.0
ENV PORT=8080
ENV LOG_LEVEL=info

# ✅ Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 🌐 Expose port
EXPOSE 8080

# 🚀 Run the app
CMD ["python", "main.py"]