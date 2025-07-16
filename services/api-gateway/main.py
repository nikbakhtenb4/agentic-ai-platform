from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import redis
import asyncpg
import os
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global connections
redis_client = None
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global redis_client, db_pool
    
    # Startup
    logger.info("🚀 Starting API Gateway...")
    
    # Initialize Redis
    try:
        redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
        redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
    
    # Initialize PostgreSQL
    try:
        db_pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/agentic_db"),
            min_size=1,
            max_size=10
        )
        logger.info("✅ PostgreSQL connected")
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down API Gateway...")
    if db_pool:
        await db_pool.close()
    if redis_client:
        redis_client.close()

# FastAPI app
app = FastAPI(
    title="Agentic AI Platform - API Gateway",
    description="Central API Gateway for Agentic AI Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در production محدود کن
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # در production محدود کن
)

# Dependency functions
async def get_redis():
    """Get Redis client"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    return redis_client

async def get_db():
    """Get database connection"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    return db_pool

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic AI Platform API Gateway",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check Redis
    try:
        if redis_client:
            redis_client.ping()
            health_status["services"]["redis"] = "healthy"
        else:
            health_status["services"]["redis"] = "unavailable"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check PostgreSQL
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health_status["services"]["postgres"] = "healthy"
        else:
            health_status["services"]["postgres"] = "unavailable"
    except Exception as e:
        health_status["services"]["postgres"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/api/info")
async def api_info():
    """API information"""
    return {
        "platform": "Agentic AI",
        "gateway_version": "1.0.0",
        "available_services": [
            "api-gateway",
            "test-service"
        ],
        "environment": os.getenv("ENV", "development"),
        "features": [
            "health_monitoring",
            "request_logging",
            "basic_routing"
        ]
    }

@app.get("/api/test-connection")
async def test_connections(redis: redis.Redis = Depends(get_redis), db = Depends(get_db)):
    """Test all connections"""
    results = {}
    
    # Test Redis
    try:
        redis.set("test_key", "test_value", ex=60)
        value = redis.get("test_key")
        results["redis"] = {"status": "success", "test": f"stored and retrieved: {value}"}
    except Exception as e:
        results["redis"] = {"status": "error", "message": str(e)}
    
    # Test PostgreSQL
    try:
        async with db.acquire() as conn:
            result = await conn.fetchval("SELECT current_timestamp")
            results["postgres"] = {"status": "success", "timestamp": str(result)}
    except Exception as e:
        results["postgres"] = {"status": "error", "message": str(e)}
    
    return results

@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint"""
    # در آینده Prometheus metrics اضافه می‌کنیم
    return {
        "requests_total": "not_implemented",
        "active_connections": "not_implemented",
        "uptime": "not_implemented"
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if os.getenv("ENV") == "development" else False,
        log_level="info"
    )