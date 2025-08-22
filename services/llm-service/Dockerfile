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
ENV PYTHONPATH=/app:/app/models:/app/utils:/app/shared
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV TOKENIZERS_PARALLELISM=false

# Create directories with proper permissions
RUN mkdir -p /app/models /app/utils /app/cache /app/logs /app/shared && \
    chown -R llmuser:llmuser /app

# Switch to non-root user
USER llmuser
WORKDIR /app

# First copy the Python modules to ensure they exist
COPY --chown=llmuser:llmuser models/ /app/models/
COPY --chown=llmuser:llmuser utils/ /app/utils/

# Copy main application files
COPY --chown=llmuser:llmuser main.py /app/
COPY --chown=llmuser:llmuser requirements.txt /app/

# Copy any additional shared modules
COPY --chown=llmuser:llmuser . .

# Create __init__.py files to make directories proper Python packages
RUN touch /app/__init__.py && \
    touch /app/models/__init__.py && \
    touch /app/utils/__init__.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f https://localhost:8002/health || exit 1

# Expose port
EXPOSE 8002

# Run the application
CMD ["python", "/app/main.py"]