#!/usr/bin/env python3
# services/gpu-coordinator/main.py
"""
Improved GPU Coordinator
مدیریت پیشرفته GPU برای سرویس‌های مختلف
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import subprocess
import json
import statistics
import threading
import psutil
import time

# تنظیم logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPU Coordinator", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
start_time = time.time()
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "8192"))
SCHEDULER_INTERVAL = float(os.getenv("SCHEDULER_INTERVAL", "5.0"))


# Models
class GPURequest(BaseModel):
    service_name: str
    estimated_memory: float = 2.0
    priority: str = "normal"
    timeout: int = 300


class GPUInfo(BaseModel):
    id: int
    name: str
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    utilization_percent: float
    temperature: Optional[int] = None
    available: bool = True
    running_tasks: int = 0


# Global storage
gpu_allocations: Dict[str, dict] = {}
task_queue: List[dict] = []
gpu_devices: Dict[int, GPUInfo] = {}
last_gpu_scan = None
allocated_gpus: Dict[int, dict] = {}
request_queue: List[dict] = []


# GPU Manager simulation
class GPUManager:
    def __init__(self):
        self.device_count = 0
        self.gpu_available = False
        self.total_requests = 0
        self.successful_allocations = 0
        self.failed_allocations = 0
        self.avg_allocation_time = 0.0
        self._initialize()

    def _initialize(self):
        """Initialize GPU manager"""
        try:
            # Try to detect real GPUs first
            detect_gpus()
            self.device_count = len(gpu_devices)
            self.gpu_available = self.device_count > 0
            logger.info(f"✅ GPUManager initialized with {self.device_count} devices")
        except Exception as e:
            logger.error(f"❌ GPUManager initialization failed: {e}")
            self.device_count = 1  # Fallback
            self.gpu_available = True

    def get_gpu_info(self, gpu_id: int) -> Dict[str, Any]:
        """Get information about specific GPU"""
        if gpu_id in gpu_devices:
            gpu = gpu_devices[gpu_id]
            return {
                "name": gpu.name,
                "total_memory_mb": int(gpu.memory_total_gb * 1024),
                "used_memory_mb": int(gpu.memory_used_gb * 1024),
                "free_memory_mb": int(gpu.memory_free_gb * 1024),
                "utilization_percent": gpu.utilization_percent,
                "temperature": gpu.temperature,
                "power_usage": 150,  # Simulated power usage
            }
        return {
            "name": f"GPU-{gpu_id}",
            "total_memory_mb": 8192,
            "used_memory_mb": 2048,
            "free_memory_mb": 6144,
            "utilization_percent": 25.0,
            "temperature": 65,
            "power_usage": 150,
        }


# Initialize GPU manager
gpu_manager = GPUManager()


def detect_gpus():
    """تشخیص GPU های موجود در سیستم"""
    global gpu_devices, last_gpu_scan

    try:
        # تلاش برای استفاده از nvidia-ml-py
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_gb = mem_info.total / (1024**3)
                used_gb = mem_info.used / (1024**3)
                free_gb = mem_info.free / (1024**3)

                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                except:
                    utilization = 0.0

                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except:
                    temp = None

                gpu_devices[i] = GPUInfo(
                    id=i,
                    name=name,
                    memory_total_gb=total_gb,
                    memory_used_gb=used_gb,
                    memory_free_gb=free_gb,
                    utilization_percent=utilization,
                    temperature=temp,
                    available=free_gb > 1.0,  # حداقل 1GB آزاد
                    running_tasks=len(
                        [
                            t
                            for t in gpu_allocations.values()
                            if t.get("gpu_id") == i and t.get("status") == "running"
                        ]
                    ),
                )

            last_gpu_scan = datetime.now()
            logger.info(f"✅ Detected {device_count} GPU(s) using pynvml")
            return

        except ImportError:
            logger.warning("⚠️ pynvml not available, trying nvidia-smi...")
        except Exception as e:
            logger.warning(f"⚠️ pynvml failed: {e}, trying nvidia-smi...")

        # فال‌بک به nvidia-smi
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 6:
                            gpu_id = int(parts[0])
                            name = parts[1]
                            total_mb = float(parts[2])
                            used_mb = float(parts[3])
                            free_mb = float(parts[4])
                            util = (
                                float(parts[5])
                                if parts[5] != "[Not Supported]"
                                else 0.0
                            )
                            temp = (
                                int(parts[6])
                                if len(parts) > 6 and parts[6] != "[Not Supported]"
                                else None
                            )

                            gpu_devices[gpu_id] = GPUInfo(
                                id=gpu_id,
                                name=name,
                                memory_total_gb=total_mb / 1024,
                                memory_used_gb=used_mb / 1024,
                                memory_free_gb=free_mb / 1024,
                                utilization_percent=util,
                                temperature=temp,
                                available=free_mb > 1024,  # حداقل 1GB آزاد
                                running_tasks=len(
                                    [
                                        t
                                        for t in gpu_allocations.values()
                                        if t.get("gpu_id") == gpu_id
                                        and t.get("status") == "running"
                                    ]
                                ),
                            )

                last_gpu_scan = datetime.now()
                logger.info(f"✅ Detected {len(gpu_devices)} GPU(s) using nvidia-smi")
                return

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as e:
            logger.warning(f"⚠️ nvidia-smi failed: {e}")

        # اگر هیچ GPU تشخیص داده نشد، یک GPU شبیه‌سازی شده ایجاد کن
        logger.warning("⚠️ No real GPUs detected, creating simulated GPU")
        gpu_devices[0] = GPUInfo(
            id=0,
            name="Simulated GPU",
            memory_total_gb=8.0,
            memory_used_gb=2.0,
            memory_free_gb=6.0,
            utilization_percent=25.0,
            temperature=65,
            available=True,
            running_tasks=0,
        )
        last_gpu_scan = datetime.now()

    except Exception as e:
        logger.error(f"❌ GPU detection failed: {e}")
        # GPU شبیه‌سازی شده در صورت خرابی
        gpu_devices[0] = GPUInfo(
            id=0,
            name="Fallback GPU",
            memory_total_gb=4.0,
            memory_used_gb=1.0,
            memory_free_gb=3.0,
            utilization_percent=10.0,
            available=True,
            running_tasks=0,
        )


def find_best_gpu(memory_needed: float) -> Optional[int]:
    """یافتن بهترین GPU برای تخصیص"""
    available_gpus = [
        (gpu_id, gpu)
        for gpu_id, gpu in gpu_devices.items()
        if gpu.available and gpu.memory_free_gb >= memory_needed
    ]

    if not available_gpus:
        return None

    # مرتب‌سازی براساس حافظه آزاد (بیشترین اول)
    available_gpus.sort(key=lambda x: x[1].memory_free_gb, reverse=True)
    return available_gpus[0][0]


@app.on_event("startup")
async def startup_event():
    """راه‌اندازی اولیه"""
    logger.info("🚀 Starting GPU Coordinator...")
    detect_gpus()

    # نصب pynvml اگر موجود نباشد
    try:
        import pynvml
    except ImportError:
        logger.info("📦 Installing pynvml...")
        try:
            subprocess.run(["pip", "install", "nvidia-ml-py"], check=True)
            logger.info("✅ pynvml installed successfully")
        except:
            logger.warning("⚠️ Could not install pynvml, using fallback methods")


@app.get("/health")
async def health():
    """بررسی سلامت سرویس"""
    # به‌روزرسانی اطلاعات GPU اگر قدیمی باشد
    if not last_gpu_scan or datetime.now() - last_gpu_scan > timedelta(minutes=1):
        detect_gpus()

    total_gpus = len(gpu_devices)
    available_gpus = len([g for g in gpu_devices.values() if g.available])

    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "gpu_available": available_gpus > 0,
        "total_gpus": total_gpus,
        "available_gpus": available_gpus,
        "coordinator_version": "2.0.0",
    }


@app.get("/status")
async def get_status():
    """دریافت وضعیت کامل سیستم"""
    # به‌روزرسانی اطلاعات GPU
    detect_gpus()

    return {
        "gpus": {str(gpu_id): gpu.dict() for gpu_id, gpu in gpu_devices.items()},
        "queue": {
            "pending_tasks": len(task_queue),
            "total_tasks": len(gpu_allocations),
            "running_tasks": len(
                [t for t in gpu_allocations.values() if t.get("status") == "running"]
            ),
        },
        "system": {
            "total_memory_gb": sum(gpu.memory_total_gb for gpu in gpu_devices.values()),
            "used_memory_gb": sum(gpu.memory_used_gb for gpu in gpu_devices.values()),
            "free_memory_gb": sum(gpu.memory_free_gb for gpu in gpu_devices.values()),
            "memory_utilization_percent": (
                sum(gpu.memory_used_gb for gpu in gpu_devices.values())
                / sum(gpu.memory_total_gb for gpu in gpu_devices.values())
                * 100
                if gpu_devices
                else 0
            ),
            "average_utilization": (
                sum(gpu.utilization_percent for gpu in gpu_devices.values())
                / len(gpu_devices)
                if gpu_devices
                else 0
            ),
        },
        "last_updated": last_gpu_scan,
    }


@app.get("/stats")
async def get_detailed_stats():
    """آمار تفصیلی GPU Coordinator"""
    try:
        current_time = time.time()

        # آمار GPU
        gpu_stats = []
        if gpu_manager:
            for gpu_id in range(gpu_manager.device_count):
                gpu_info = gpu_manager.get_gpu_info(gpu_id)
                gpu_stats.append(
                    {
                        "gpu_id": gpu_id,
                        "name": gpu_info.get("name", "Unknown"),
                        "total_memory_mb": gpu_info.get("total_memory_mb", 0),
                        "used_memory_mb": gpu_info.get("used_memory_mb", 0),
                        "free_memory_mb": gpu_info.get("free_memory_mb", 0),
                        "utilization_percent": gpu_info.get("utilization_percent", 0),
                        "temperature": gpu_info.get("temperature", 0),
                        "power_usage": gpu_info.get("power_usage", 0),
                        "allocated_to": allocated_gpus.get(gpu_id, {}).get(
                            "service", None
                        ),
                        "allocation_time": allocated_gpus.get(gpu_id, {}).get(
                            "allocated_at", None
                        ),
                    }
                )

        # آمار صف (Queue)
        queue_stats = {
            "total_pending": len(request_queue),
            "llm_pending": sum(
                1 for req in request_queue if req.get("service_name") == "llm-service"
            ),
            "stt_pending": sum(
                1 for req in request_queue if req.get("service_name") == "stt-service"
            ),
            "average_wait_time": statistics.mean(
                [req.get("wait_time", 0) for req in request_queue]
            )
            if request_queue
            else 0,
            "queue_details": [
                {
                    "position": i + 1,
                    "service": req.get("service_name", "unknown"),
                    "memory_required": req.get(
                        "memory_gb", req.get("estimated_memory", 0)
                    ),
                    "priority": req.get("priority", "normal"),
                    "queued_at": req.get("queued_at", current_time),
                    "wait_time": current_time - req.get("queued_at", current_time),
                }
                for i, req in enumerate(request_queue)
            ],
        }

        # آمار تخصیصات
        allocation_stats = {
            "total_allocations": len(allocated_gpus),
            "active_services": list(
                set(
                    gpu.get("service")
                    for gpu in allocated_gpus.values()
                    if gpu.get("service")
                )
            ),
            "allocation_details": {
                gpu_id: {
                    "service": gpu_info.get("service"),
                    "memory_allocated": gpu_info.get("memory_gb"),
                    "allocated_at": gpu_info.get("allocated_at"),
                    "duration": current_time
                    - gpu_info.get("allocated_at", current_time),
                    "task_id": gpu_info.get("task_id"),
                }
                for gpu_id, gpu_info in allocated_gpus.items()
            },
        }

        # آمار سیستم
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        system_stats = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_used_gb": round(memory.used / 1024**3, 2),
            "memory_total_gb": round(memory.total / 1024**3, 2),
            "uptime_seconds": current_time - start_time,
            "active_threads": len([t for t in threading.enumerate() if t.is_alive()]),
        }

        # آمار عملکرد
        performance_stats = {
            "total_requests_processed": getattr(gpu_manager, "total_requests", 0),
            "successful_allocations": getattr(gpu_manager, "successful_allocations", 0),
            "failed_allocations": getattr(gpu_manager, "failed_allocations", 0),
            "average_allocation_time": getattr(gpu_manager, "avg_allocation_time", 0),
            "gpu_utilization_avg": statistics.mean(
                [gpu.get("utilization_percent", 0) for gpu in gpu_stats]
            )
            if gpu_stats
            else 0,
        }

        return {
            "timestamp": current_time,
            "coordinator_status": "active",
            "gpu_stats": gpu_stats,
            "queue_stats": queue_stats,
            "allocation_stats": allocation_stats,
            "system_stats": system_stats,
            "performance_stats": performance_stats,
            "configuration": {
                "max_memory_mb": MAX_MEMORY_MB,
                "scheduler_interval": SCHEDULER_INTERVAL,
                "device_count": gpu_manager.device_count if gpu_manager else 0,
                "gpu_available": gpu_manager.gpu_available if gpu_manager else False,
            },
        }

    except Exception as e:
        logger.error(f"Error generating stats: {e}")
        return {
            "timestamp": time.time(),
            "coordinator_status": "error",
            "error": str(e),
            "basic_stats": {
                "gpu_count": gpu_manager.device_count if gpu_manager else 0,
                "queue_length": len(request_queue),
                "active_allocations": len(allocated_gpus),
            },
        }


@app.get("/stats/summary")
async def get_stats_summary():
    """خلاصه آمار GPU Coordinator"""
    try:
        return {
            "gpu_count": gpu_manager.device_count if gpu_manager else 0,
            "gpus_allocated": len(allocated_gpus),
            "gpus_available": (gpu_manager.device_count - len(allocated_gpus))
            if gpu_manager
            else 0,
            "queue_length": len(request_queue),
            "services_active": len(
                set(
                    gpu.get("service")
                    for gpu in allocated_gpus.values()
                    if gpu.get("service")
                )
            ),
            "coordinator_status": "healthy",
            "uptime_minutes": round((time.time() - start_time) / 60, 1),
        }
    except Exception as e:
        return {"coordinator_status": "error", "error": str(e)}


@app.get("/stats/gpu/{gpu_id}")
async def get_gpu_specific_stats(gpu_id: int):
    """آمار خاص یک GPU"""
    try:
        if not gpu_manager or gpu_id >= gpu_manager.device_count:
            raise HTTPException(status_code=404, detail=f"GPU {gpu_id} not found")

        gpu_info = gpu_manager.get_gpu_info(gpu_id)
        allocation_info = allocated_gpus.get(gpu_id, {})

        return {
            "gpu_id": gpu_id,
            "hardware_info": gpu_info,
            "allocation_info": allocation_info,
            "is_allocated": gpu_id in allocated_gpus,
            "history": getattr(gpu_manager, f"gpu_{gpu_id}_history", []),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/request")
async def request_gpu(request: GPURequest):
    """درخواست تخصیص GPU"""
    task_id = f"task_{len(gpu_allocations)}_{int(datetime.now().timestamp())}"

    # Update counters
    gpu_manager.total_requests += 1

    # یافتن بهترین GPU
    gpu_id = find_best_gpu(request.estimated_memory)

    if gpu_id is not None:
        # تخصیص موفق
        gpu_allocations[task_id] = {
            "service_name": request.service_name,
            "gpu_id": gpu_id,
            "status": "running",
            "memory_requested": request.estimated_memory,
            "priority": request.priority,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=request.timeout),
        }

        # Update allocated_gpus
        allocated_gpus[gpu_id] = {
            "service": request.service_name,
            "memory_gb": request.estimated_memory,
            "allocated_at": time.time(),
            "task_id": task_id,
        }

        # به‌روزرسانی وضعیت GPU
        if gpu_id in gpu_devices:
            gpu_devices[gpu_id].running_tasks += 1
            gpu_devices[gpu_id].memory_used_gb += request.estimated_memory
            gpu_devices[gpu_id].memory_free_gb -= request.estimated_memory

            if gpu_devices[gpu_id].memory_free_gb < 0.5:  # حداقل 500MB آزاد
                gpu_devices[gpu_id].available = False

        gpu_manager.successful_allocations += 1

        logger.info(
            f"✅ GPU {gpu_id} allocated to {request.service_name} (task: {task_id})"
        )

        return {
            "task_id": task_id,
            "gpu_id": gpu_id,
            "allocated": True,
            "message": f"GPU {gpu_id} allocated successfully",
            "expires_at": gpu_allocations[task_id]["expires_at"],
        }
    else:
        # اضافه کردن به صف انتظار
        queue_item = {
            "task_id": task_id,
            "service_name": request.service_name,
            "memory_requested": request.estimated_memory,
            "estimated_memory": request.estimated_memory,
            "memory_gb": request.estimated_memory,
            "priority": request.priority,
            "queued_at": time.time(),
            "wait_time": 0,
        }
        task_queue.append(queue_item)
        request_queue.append(queue_item)

        gpu_manager.failed_allocations += 1

        logger.warning(f"⚠️ No GPU available for {request.service_name}, added to queue")

        return {
            "task_id": task_id,
            "gpu_id": None,
            "allocated": False,
            "message": "No GPU available, added to queue",
            "queue_position": len(task_queue),
            "estimated_wait_time": len(task_queue) * 60,  # تخمین 1 دقیقه به ازای هر تسک
        }


@app.post("/release/{task_id}")
async def release_gpu(task_id: str):
    """آزادسازی GPU"""
    if task_id not in gpu_allocations:
        raise HTTPException(status_code=404, detail="Task not found")

    allocation = gpu_allocations[task_id]
    gpu_id = allocation.get("gpu_id")
    memory_requested = allocation.get("memory_requested", 0)

    # به‌روزرسانی وضعیت
    allocation["status"] = "completed"
    allocation["completed_at"] = datetime.now()

    # آزادسازی حافظه GPU
    if gpu_id is not None and gpu_id in gpu_devices:
        gpu_devices[gpu_id].running_tasks = max(
            0, gpu_devices[gpu_id].running_tasks - 1
        )
        gpu_devices[gpu_id].memory_used_gb -= memory_requested
        gpu_devices[gpu_id].memory_free_gb += memory_requested
        gpu_devices[gpu_id].available = True

        # Remove from allocated_gpus
        if gpu_id in allocated_gpus:
            del allocated_gpus[gpu_id]

        logger.info(f"✅ GPU {gpu_id} released (task: {task_id})")

    # پردازش صف انتظار
    await process_queue()

    return {
        "status": "success",
        "message": f"GPU released for {task_id}",
        "gpu_id": gpu_id,
    }


async def process_queue():
    """پردازش صف انتظار"""
    if not task_queue:
        return

    # مرتب‌سازی براساس اولویت و زمان
    task_queue.sort(
        key=lambda x: (
            0 if x["priority"] == "high" else 1 if x["priority"] == "normal" else 2,
            x["queued_at"],
        )
    )

    processed = []
    for i, queue_item in enumerate(task_queue):
        gpu_id = find_best_gpu(queue_item["memory_requested"])
        if gpu_id is not None:
            # تخصیص از صف
            task_id = queue_item["task_id"]
            gpu_allocations[task_id] = {
                "service_name": queue_item["service_name"],
                "gpu_id": gpu_id,
                "status": "running",
                "memory_requested": queue_item["memory_requested"],
                "priority": queue_item["priority"],
                "created_at": datetime.now(),
                "queued_at": queue_item["queued_at"],
            }

            # Update allocated_gpus
            allocated_gpus[gpu_id] = {
                "service": queue_item["service_name"],
                "memory_gb": queue_item["memory_requested"],
                "allocated_at": time.time(),
                "task_id": task_id,
            }

            # به‌روزرسانی GPU
            gpu_devices[gpu_id].running_tasks += 1
            gpu_devices[gpu_id].memory_used_gb += queue_item["memory_requested"]
            gpu_devices[gpu_id].memory_free_gb -= queue_item["memory_requested"]

            if gpu_devices[gpu_id].memory_free_gb < 0.5:
                gpu_devices[gpu_id].available = False

            processed.append(i)
            logger.info(f"✅ Queued task {task_id} allocated to GPU {gpu_id}")

    # حذف تسک‌های پردازش شده از صف
    for i in reversed(processed):
        task_queue.pop(i)
        if i < len(request_queue):
            request_queue.pop(i)


@app.get("/tasks")
async def get_tasks():
    """دریافت لیست تسک‌ها"""
    return {
        "tasks": gpu_allocations,
        "queue_length": len(task_queue),
        "running_tasks": len(
            [t for t in gpu_allocations.values() if t.get("status") == "running"]
        ),
    }


@app.get("/queue")
async def get_queue():
    """دریافت وضعیت صف انتظار"""
    return {
        "queue": task_queue,
        "queue_length": len(task_queue),
        "estimated_wait_time": len(task_queue) * 60,
    }


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """لغو تسک"""
    # حذف از صف انتظار
    task_queue[:] = [t for t in task_queue if t["task_id"] != task_id]
    request_queue[:] = [t for t in request_queue if t["task_id"] != task_id]

    # لغو تخصیص فعال
    if task_id in gpu_allocations:
        await release_gpu(task_id)

    return {"status": "success", "message": f"Task {task_id} cancelled"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    uvicorn.run(app, host=host, port=port, log_level=log_level)
