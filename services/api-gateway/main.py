# services/api-gateway/main.py
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
import httpx

# Import route modules
from routes import llm, stt

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
            os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
        )
        redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")

    # Initialize PostgreSQL
    try:
        db_pool = await asyncpg.create_pool(
            os.getenv(
                "DATABASE_URL",
                "postgresql://postgres:password@localhost:5432/agentic_db",
            ),
            min_size=1,
            max_size=10,
        )
        logger.info("✅ PostgreSQL connected")
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")

    # Test service connections
    await test_service_connections()

    yield

    # Shutdown
    logger.info("🛑 Shutting down API Gateway...")
    if db_pool:
        await db_pool.close()
    if redis_client:
        redis_client.close()


async def test_service_connections():
    """Test connections to microservices on startup"""
    services = {
        "LLM Service": os.getenv("LLM_SERVICE_URL", "https://llm-service:8001"),
        "STT Service": os.getenv("STT_SERVICE_URL", "https://stt-service:8002"),
        "GPU Coordinator": os.getenv(
            "GPU_COORDINATOR_URL", "https://gpu-coordinator:8080"
        ),
    }

    for service_name, service_url in services.items():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service_url}/health")
                if response.status_code == 200:
                    logger.info(f"✅ {service_name} is healthy")
                else:
                    logger.warning(
                        f"⚠️ {service_name} returned status {response.status_code}"
                    )
        except Exception as e:
            logger.warning(f"⚠️ {service_name} connection failed: {e}")


# FastAPI app
app = FastAPI(
    title="Agentic AI Platform - API Gateway",
    description="Central API Gateway for Agentic AI Platform with LLM and STT services",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
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
    allowed_hosts=["*"],  # در production محدود کن
)

# Include routers
app.include_router(llm.router, prefix="/api/v1")
app.include_router(stt.router, prefix="/api/v1")


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


# Main routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic AI Platform API Gateway",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "services": {"llm": "/api/v1/llm", "stt": "/api/v1/stt", "docs": "/docs"},
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "infrastructure": {},
    }

    # Check Redis
    try:
        if redis_client:
            redis_client.ping()
            health_status["infrastructure"]["redis"] = "healthy"
        else:
            health_status["infrastructure"]["redis"] = "unavailable"
    except Exception as e:
        health_status["infrastructure"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check PostgreSQL
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health_status["infrastructure"]["postgres"] = "healthy"
        else:
            health_status["infrastructure"]["postgres"] = "unavailable"
    except Exception as e:
        health_status["infrastructure"]["postgres"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check microservices
    microservices = {
        "llm": os.getenv("LLM_SERVICE_URL", "https://llm-service:8001"),
        "stt": os.getenv("STT_SERVICE_URL", "https://stt-service:8002"),
        "gpu_coordinator": os.getenv(
            "GPU_COORDINATOR_URL", "https://gpu-coordinator:8080"
        ),
    }

    for service_name, service_url in microservices.items():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{service_url}/health")
                if response.status_code == 200:
                    health_status["services"][service_name] = "healthy"
                else:
                    health_status["services"][service_name] = (
                        f"status: {response.status_code}"
                    )
                    health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"][service_name] = f"unreachable: {str(e)}"
            health_status["status"] = "degraded"

    return health_status


@app.get("/api/info")
async def api_info():
    """API information and available endpoints"""
    return {
        "platform": "Agentic AI",
        "gateway_version": "1.0.0",
        "available_services": [
            "api-gateway",
            "llm-service",
            "stt-service",
            "gpu-coordinator",
        ],
        "environment": os.getenv("ENV", "development"),
        "features": [
            "health_monitoring",
            "request_logging",
            "service_routing",
            "gpu_management",
            "llm_generation",
            "speech_to_text",
            "cors_support",
            "rate_limiting",
        ],
        "endpoints": {
            "llm": {
                "generate": "POST /api/v1/llm/generate",
                "chat": "POST /api/v1/llm/chat",
                "model_info": "GET /api/v1/llm/model/info",
                "batch_generate": "POST /api/v1/llm/batch/generate",
                "health": "GET /api/v1/llm/health",
            },
            "stt": {
                "transcribe": "POST /api/v1/stt/transcribe",
                "transcribe_base64": "POST /api/v1/stt/transcribe-base64",
                "batch_transcribe": "POST /api/v1/stt/transcribe-batch",
                "languages": "GET /api/v1/stt/languages",
                "health": "GET /api/v1/stt/health",
            },
        },
    }


@app.get("/api/test-connection")
async def test_connections(redis: redis.Redis = Depends(get_redis), db=Depends(get_db)):
    """Test all connections including microservices"""
    results = {}

    # Test Redis
    try:
        redis.set("test_key", "test_value", ex=60)
        value = redis.get("test_key")
        results["redis"] = {
            "status": "success",
            "test": f"stored and retrieved: {value}",
        }
    except Exception as e:
        results["redis"] = {"status": "error", "message": str(e)}

    # Test PostgreSQL
    try:
        async with db.acquire() as conn:
            result = await conn.fetchval("SELECT current_timestamp")
            results["postgres"] = {"status": "success", "timestamp": str(result)}
    except Exception as e:
        results["postgres"] = {"status": "error", "message": str(e)}

    # Test microservices
    microservices = {
        "llm_service": os.getenv("LLM_SERVICE_URL", "https://llm-service:8001"),
        "stt_service": os.getenv("STT_SERVICE_URL", "http://stt-service:8002"),
        "gpu_coordinator": os.getenv(
            "GPU_COORDINATOR_URL", "https://gpu-coordinator:8080"
        ),
    }

    for service_name, service_url in microservices.items():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service_url}/health")
                results[service_name] = {
                    "status": "success" if response.status_code == 200 else "degraded",
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                    "details": response.json()
                    if response.status_code == 200
                    else response.text,
                }
        except Exception as e:
            results[service_name] = {"status": "error", "message": str(e)}

    return results


@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint for monitoring"""
    # در آینده Prometheus metrics اضافه می‌کنیم
    metrics_data = {
        "gateway": {
            "requests_total": "not_implemented",
            "active_connections": "not_implemented",
            "uptime": "not_implemented",
        },
        "services": {
            "llm": {
                "requests_total": "not_implemented",
                "avg_processing_time": "not_implemented",
            },
            "stt": {
                "requests_total": "not_implemented",
                "avg_processing_time": "not_implemented",
            },
        },
        "gpu": {
            "utilization": "not_implemented",
            "memory_usage": "not_implemented",
            "active_allocations": "not_implemented",
        },
    }

    # Try to get real GPU metrics
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{os.getenv('GPU_COORDINATOR_URL', 'https://gpu-coordinator:8080')}/stats"
            )
            if response.status_code == 200:
                gpu_stats = response.json()
                metrics_data["gpu"] = gpu_stats
    except Exception as e:
        logger.warning(f"Could not fetch GPU metrics: {e}")

    return metrics_data


@app.get("/api/services")
async def list_services():
    """List all available services and their status"""
    services = {
        "api_gateway": {
            "status": "running",
            "url": "localhost:8000",
            "description": "Central API Gateway",
        }
    }

    microservices = {
        "llm_service": {
            "url": os.getenv("LLM_SERVICE_URL", "https://llm-service:8001"),
            "description": "Large Language Model Service",
        },
        "stt_service": {
            "url": os.getenv("STT_SERVICE_URL", "http://stt-service:8002"),
            "description": "Speech-to-Text Service",
        },
        "gpu_coordinator": {
            "url": os.getenv("GPU_COORDINATOR_URL", "https://gpu-coordinator:8080"),
            "description": "GPU Resource Coordinator",
        },
    }

    for service_name, service_info in microservices.items():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{service_info['url']}/health")
                services[service_name] = {
                    **service_info,
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                }
        except Exception as e:
            services[service_name] = {
                **service_info,
                "status": "unreachable",
                "error": str(e),
            }

    return services


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "detail": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url),
        },
    )


# Custom middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()

    # Log request
    logger.info(f"Request: {request.method} {request.url}")

    response = await call_next(request)

    # Log response
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.2f}s")

    return response


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if os.getenv("ENV") == "development" else False,
        log_level="info",
        access_log=True,
    )
