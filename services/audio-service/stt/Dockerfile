# audio-service/stt/Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY ./services/audio-service/stt/requirements.txt .

# Install Python dependencies with better error handling
RUN pip install --timeout=1000 --retries=10 -r requirements.txt

# Create directories
RUN mkdir -p /app/models \
             /app/cache \
             /app/logs \
             /app/uploads \
             /app/temp

# Copy application code
COPY ./services/audio-service/stt/main.py .
COPY ./services/audio-service/stt/utils/ ./utils/
COPY ./services/audio-service/stt/config/ ./config/

# Copy shared modules
COPY ./shared /app/shared

# Set PYTHONPATH to include shared modules
ENV PYTHONPATH="/app:/app/shared"

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 8003

# Health check (Fixed: HTTP instead of HTTPS)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the application
CMD ["python", "main.py"]