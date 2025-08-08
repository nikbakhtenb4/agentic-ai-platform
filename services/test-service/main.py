from fastapi import FastAPI
import uvicorn
import os
import redis
from datetime import datetime
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Test Service",
    description="Simple test service for Agentic AI Platform",
    version="1.0.0",
)

# Redis connection
redis_client = None
try:
    redis_client = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
    )
    redis_client.ping()
    logger.info("✅ Test Service Redis connected")
except Exception as e:
    logger.error(f"❌ Test Service Redis connection failed: {e}")


@app.get("/")
async def root():
    return {
        "service": "test-service",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health():
    return {
        "service": "test-service",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/ping")
async def ping():
    return {"message": "pong", "timestamp": datetime.now().isoformat()}


@app.get("/redis-test")
async def redis_test():
    if not redis_client:
        return {"error": "Redis not available"}

    try:
        # Test Redis
        test_key = f"test_service_key_{datetime.now().timestamp()}"
        redis_client.set(test_key, "test_value", ex=60)
        value = redis_client.get(test_key)

        return {
            "redis_status": "working",
            "test_key": test_key,
            "stored_value": value,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/echo/{message}")
async def echo(message: str):
    return {
        "original_message": message,
        "echo": f"Echo: {message}",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True if os.getenv("ENV") == "development" else False,
    )
