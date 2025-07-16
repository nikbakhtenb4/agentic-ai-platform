# Multi-stage build برای بهینه‌سازی حجم
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
COPY requirements.txt .

# نصب torch از mirror سریع
RUN pip install --upgrade pip
# نصب با debug
RUN pip install --no-cache-dir --user torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# تست های مختلف
RUN pip list
RUN pip show torch
RUN python -c "import torch; print('SUCCESS: torch imported')"
RUN python -c "import sys; print('Python path:', sys.path)"

# نصب torch با verbose output
# RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu -v

RUN pip install --no-cache-dir --user -r requirements.txt --default-timeout=1000
FROM base as runtime

# Copy Python packages from builder
COPY --from=builder /root/.local /home/llmuser/.local
# نصب مجدد torch در runtime (برای اطمینان)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

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
COPY --chown=llmuser:llmuser . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Expose port
EXPOSE 8002

# Run the application
CMD ["python", "main.py"]




# # services/llm-service/Dockerfile
# FROM nvidia/cuda:11.8-devel-ubuntu22.04

# # Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONUNBUFFERED=1
# ENV CUDA_VISIBLE_DEVICES=0

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     python3.10 \
#     python3.10-dev \
#     python3-pip \
#     git \
#     wget \
#     curl \
#     build-essential \
#     cmake \
#     pkg-config \
#     libffi-dev \
#     libssl-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Create symbolic link for python
# RUN ln -s /usr/bin/python3.10 /usr/bin/python

# # Set working directory
# WORKDIR /app

# # Copy requirements first for better caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN pip install --no-cache-dir -r requirements.txt

# # Install additional ML libraries
# RUN pip install --no-cache-dir \
#     accelerate \
#     bitsandbytes \
#     sentence-transformers \
#     optimum \
#     auto-gptq

# # Create necessary directories
# RUN mkdir -p /app/models /app/config /app/logs /app/cache /tmp/offload

# # Copy shared modules
# COPY ../../shared /app/shared

# # Copy service code
# COPY . /app/

# # Create non-root user for security
# RUN useradd -m -u 1000 llmuser && \
#     chown -R llmuser:llmuser /app && \
#     chown -R llmuser:llmuser /root/.cache

# # Switch to non-root user
# USER llmuser

# # Expose port
# EXPOSE 8002

# # Health check
# HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
#     CMD curl -f http://localhost:8002/health || exit 1

# # Run the application
# CMD ["python", "main.py"]