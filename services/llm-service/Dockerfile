# Multi-stage build برای بهینه‌سازی حجم
# llm-service/DockerFile
FROM python:3.11-slim as base

RUN echo "deb http://mirrors.aliyun.com/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*


# RUN apt-get update && apt-get install -y \

# Create non-root user
RUN useradd -m -u 1000 llmuser

RUN pip install typing_extensions>=4.0.0
FROM base as builder

# Install Python dependencies
# COPY requirements.txt .
COPY ./services/llm-service/requirements.txt .
# نصب torch از mirror سریع
RUN pip install --upgrade pip
# نصب با debug
RUN pip install --user torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# تست های مختلف
RUN pip list
RUN pip show torch
RUN python -c "import torch; print('SUCCESS: torch imported')"
RUN python -c "import sys; print('Python path:', sys.path)"

# نصب torch با verbose output
# RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu -v

RUN pip install --user -r requirements.txt --default-timeout=1000
FROM base as runtime

# Copy Python packages from builder
COPY --from=builder /root/.local /home/llmuser/.local
# نصب مجدد torch در runtime (برای اطمینان)
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Set environment variables
ENV PATH=/home/llmuser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV TOKENIZERS_PARALLELISM=false

# Create directories
RUN mkdir -p /app/models /app/cache /app/logs && \
    chown -R llmuser:llmuser /app

# Switch to non-root user
USER llmuser
WORKDIR /app

# Copy application code
# COPY --chown=llmuser:llmuser . .
COPY --chown=llmuser:llmuser ./services/llm-service/ .
COPY --chown=llmuser:llmuser ./shared /app/shared
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Expose port
EXPOSE 8002

# Run the application
CMD ["python", "main.py"]

