
# services/gpu-coordinator/Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gcc \
        g++ \
        python3-dev \
        pkg-config && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --prefer-binary \
        --default-timeout=300 --retries=10 \
        -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH="/app:/app/shared"
ENV HOST=0.0.0.0
ENV PORT=8080
ENV LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "main.py"]