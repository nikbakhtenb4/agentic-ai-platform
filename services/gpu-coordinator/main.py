# services/gpu-coordinator/main.py
#!/usr/bin/env python3
"""
GPU Coordinator Service - Main Entry Point
سرویس هماهنگ‌کننده GPU
"""

import asyncio
import logging
import os
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

# Add shared modules to path
sys.path.append("/app/shared")

# Import GPU coordinator from the file you provided
from gpu_coordinator import (
    GPUCoordinator,
    ServiceType,
    GPUTaskPriority,
    get_gpu_coordinator,
    initialize_gpu_coordinator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="GPU Coordinator Service",
    description="Manages shared GPU resources for AI services",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global coordinator instance
coordinator: Optional[GPUCoordinator] = None


# Pydantic models
class GPURequest(BaseModel):
    service_type: str
    service_name: str
    priority: str = "NORMAL"
    estimated_duration: float = 30.0
    memory_requirement: int = 1024


class GPUReleaseRequest(BaseModel):
    request_id: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize GPU coordinator on startup"""
    global coordinator
    try:
        max_memory = int(os.getenv("MAX_MEMORY_MB", "6144"))
        coordinator = await initialize_gpu_coordinator(max_memory_mb=max_memory)
        logger.info(f"🎮 GPU Coordinator started with {max_memory}MB limit")
    except Exception as e:
        logger.error(f"Failed to initialize GPU coordinator: {e}")
        # Don't fail startup - continue without GPU coordination
        coordinator = GPUCoordinator(
            max_memory_mb=int(os.getenv("MAX_MEMORY_MB", "6144"))
        )


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global coordinator
    if coordinator:
        await coordinator.stop()
        logger.info("🛑 GPU Coordinator stopped")


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "gpu-coordinator",
        "gpu_available": coordinator.gpu_available if coordinator else False,
        "coordinator_running": coordinator.scheduler_running if coordinator else False,
    }


# GPU status
@app.get("/status")
async def get_status():
    """Get GPU coordinator status"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="GPU coordinator not initialized")

    return coordinator.get_gpu_status()


# Queue status
@app.get("/queue")
async def get_queue_status():
    """Get queue status"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="GPU coordinator not initialized")

    return coordinator.get_queue_status()


# Statistics
@app.get("/stats")
async def get_statistics():
    """Get comprehensive statistics"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="GPU coordinator not initialized")

    return coordinator.get_statistics()


# Request GPU
@app.post("/request")
async def request_gpu(request: GPURequest):
    """Request GPU access"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="GPU coordinator not initialized")

    try:
        # Convert string enums
        service_type = ServiceType(request.service_type.lower())
        priority = GPUTaskPriority[request.priority.upper()]

        request_id = await coordinator.request_gpu(
            service_type=service_type,
            service_name=request.service_name,
            priority=priority,
            estimated_duration=request.estimated_duration,
            memory_requirement=request.memory_requirement,
        )

        return {
            "request_id": request_id,
            "status": "queued",
            "estimated_wait_time": len(coordinator.pending_requests)
            * 2,  # rough estimate
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request parameters: {e}")
    except Exception as e:
        logger.error(f"GPU request failed: {e}")
        raise HTTPException(status_code=500, detail=f"GPU request failed: {e}")


# Release GPU
@app.post("/release")
async def release_gpu(request: GPUReleaseRequest):
    """Release GPU allocation"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="GPU coordinator not initialized")

    try:
        success = await coordinator.release_gpu(request.request_id)

        if success:
            return {"status": "released", "request_id": request.request_id}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Request {request.request_id} not found or already released",
            )

    except Exception as e:
        logger.error(f"GPU release failed: {e}")
        raise HTTPException(status_code=500, detail=f"GPU release failed: {e}")


# Check allocation status
@app.get("/allocation/{request_id}")
async def check_allocation(request_id: str):
    """Check if GPU is allocated for a request"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="GPU coordinator not initialized")

    with coordinator.lock:
        if request_id in coordinator.active_allocations:
            allocation = coordinator.active_allocations[request_id]
            return {
                "status": "allocated",
                "request_id": request_id,
                "service_name": allocation.service_name,
                "allocated_memory": allocation.allocated_memory,
                "start_time": allocation.start_time,
                "estimated_end_time": allocation.estimated_end_time,
            }

        # Check if still pending
        for req in coordinator.pending_requests:
            if req.request_id == request_id:
                queue_position = coordinator.pending_requests.index(req) + 1
                return {
                    "status": "pending",
                    "request_id": request_id,
                    "queue_position": queue_position,
                    "estimated_wait_time": queue_position * 5,  # rough estimate
                }

        return {"status": "not_found", "request_id": request_id}


# Cleanup expired allocations
@app.post("/cleanup")
async def cleanup_expired():
    """Manual cleanup of expired allocations"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="GPU coordinator not initialized")

    try:
        await coordinator.cleanup_expired_allocations()
        return {"status": "cleanup_completed"}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {e}")


# Background task for periodic cleanup
async def periodic_cleanup():
    """Periodic cleanup of expired allocations"""
    while True:
        try:
            if coordinator:
                await coordinator.cleanup_expired_allocations()
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}")

        await asyncio.sleep(60)  # Run every minute


# Start periodic cleanup
@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(periodic_cleanup())


if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    logger.info(f"🚀 Starting GPU Coordinator Service on {host}:{port}")

    uvicorn.run("main:app", host=host, port=port, log_level=log_level, reload=False)
