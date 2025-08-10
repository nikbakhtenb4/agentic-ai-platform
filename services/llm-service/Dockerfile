
# services/llm-service/Dockerfile
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 llmuser

FROM base as builder

# Install Python dependencies
COPY requirements.txt .

# Install torch and other dependencies
RUN pip install --upgrade pip
RUN pip install --user torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --user -r requirements.txt --default-timeout=1000

FROM base as runtime

# Copy Python packages from builder
COPY --from=builder /root/.local /home/llmuser/.local

# Set environment variables
ENV PATH=/home/llmuser/.local/bin:$PATH
ENV PYTHONPATH=/app:/app/shared
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