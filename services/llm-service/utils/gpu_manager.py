# ===============================
# services/llm-service/utils/gpu_manager.py
# ===============================
# مدیریت GPU برای سرویس LLM
# استفاده از GPU Client برای coordination با سرویس GPU coordinator

import os
import logging
import torch
import psutil
from typing import Dict, Optional, Any
import asyncio

logger = logging.getLogger(__name__)

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logger.warning("⚠️ GPUtil not available - limited GPU monitoring")

try:
    from .gpu_client import SimpleGPUClient, GPUClientError, managed_gpu

    GPU_CLIENT_AVAILABLE = True
except ImportError:
    GPU_CLIENT_AVAILABLE = False
    logger.warning("⚠️ GPU Client not available - direct GPU access only")


class GPUManager:
    """مدیریت GPU برای سرویس LLM"""

    def __init__(self, service_name: str = "llm-service"):
        self.service_name = service_name
        self.gpu_available = self._check_gpu_availability()
        self.gpu_client: Optional[SimpleGPUClient] = None
        self.current_gpu_id: Optional[int] = None

        logger.info(f"🎮 GPU Manager initialized - GPU Available: {self.gpu_available}")

        if self.gpu_available:
            self._log_gpu_info()

    def _check_gpu_availability(self) -> bool:
        """بررسی در دسترس بودن GPU"""
        try:
            # بررسی PyTorch CUDA
            if not torch.cuda.is_available():
                logger.info("🚫 CUDA not available in PyTorch")
                return False

            # بررسی تعداد GPU ها
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                logger.info("🚫 No CUDA devices found")
                return False

            logger.info(f"✅ Found {gpu_count} CUDA device(s)")
            return True

        except Exception as e:
            logger.error(f"❌ GPU availability check failed: {e}")
            return False

    def _log_gpu_info(self):
        """لاگ اطلاعات GPU"""
        if not self.gpu_available:
            return

        try:
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"🎮 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

        except Exception as e:
            logger.warning(f"⚠️ Could not log GPU info: {e}")

    def get_gpu_info(self) -> Dict[str, Any]:
        """دریافت اطلاعات GPU"""
        if not self.gpu_available:
            return {
                "available": False,
                "count": 0,
                "current_gpu": None,
                "memory_info": {},
                "utilization": 0,
            }

        try:
            gpu_count = torch.cuda.device_count()
            current_gpu = (
                torch.cuda.current_device() if torch.cuda.is_available() else None
            )

            # اطلاعات حافظه GPU فعلی
            memory_info = {}
            utilization = 0

            if current_gpu is not None:
                memory_allocated = torch.cuda.memory_allocated(current_gpu) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(current_gpu) / 1024**3
                memory_total = (
                    torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
                )

                memory_info = {
                    "allocated_gb": round(memory_allocated, 2),
                    "reserved_gb": round(memory_reserved, 2),
                    "total_gb": round(memory_total, 2),
                    "utilization_percent": round(
                        (memory_allocated / memory_total) * 100, 1
                    ),
                }
                utilization = memory_info["utilization_percent"]

            # اطلاعات اضافی با GPUtil اگر موجود باشد
            gpu_details = []
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_details.append(
                            {
                                "id": gpu.id,
                                "name": gpu.name,
                                "load": round(gpu.load * 100, 1),
                                "memory_used": round(gpu.memoryUsed / 1024, 2),
                                "memory_total": round(gpu.memoryTotal / 1024, 2),
                                "temperature": gpu.temperature
                                if hasattr(gpu, "temperature")
                                else None,
                            }
                        )
                except Exception as e:
                    logger.warning(f"⚠️ GPUtil info failed: {e}")

            return {
                "available": True,
                "count": gpu_count,
                "current_gpu": current_gpu,
                "coordinator_gpu": self.current_gpu_id,  # GPU از coordinator
                "memory_info": memory_info,
                "utilization": utilization,
                "gpu_details": gpu_details,
            }

        except Exception as e:
            logger.error(f"❌ GPU info error: {e}")
            return {"available": False, "error": str(e)}

    async def request_gpu_via_coordinator(
        self, memory_gb: float = 3.0, priority: str = "normal", max_wait_time: int = 120
    ) -> bool:
        """درخواست GPU از طریق coordinator"""

        if not GPU_CLIENT_AVAILABLE:
            logger.warning("⚠️ GPU Client not available - using direct GPU access")
            return self.gpu_available

        try:
            coordinator_url = os.getenv(
                "GPU_COORDINATOR_URL", "https://gpu-coordinator:8080"
            )

            self.gpu_client = SimpleGPUClient(
                coordinator_url=coordinator_url, service_name=self.service_name
            )

            # بررسی در دسترس بودن coordinator
            if not await self.gpu_client.is_coordinator_available():
                logger.warning("⚠️ GPU Coordinator not available")
                return self.gpu_available

            # درخواست GPU
            allocation = await self.gpu_client.wait_for_gpu(
                memory_gb=memory_gb, priority=priority, max_wait_time=max_wait_time
            )

            if allocation.allocated:
                self.current_gpu_id = allocation.gpu_id
                os.environ["CUDA_VISIBLE_DEVICES"] = str(allocation.gpu_id)
                logger.info(f"✅ GPU {allocation.gpu_id} allocated via coordinator")
                return True
            else:
                logger.warning("⚠️ GPU allocation failed")
                return False

        except Exception as e:
            logger.error(f"❌ GPU coordinator request failed: {e}")
            return self.gpu_available

    async def release_gpu_from_coordinator(self) -> bool:
        """آزادسازی GPU از coordinator"""
        if self.gpu_client and self.gpu_client.is_gpu_allocated():
            try:
                await self.gpu_client.release_gpu()
                self.current_gpu_id = None
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                logger.info("✅ GPU released from coordinator")
                return True
            except Exception as e:
                logger.error(f"❌ GPU release failed: {e}")
                return False
        return True

    def set_gpu_device(self, device_id: int = 0):
        """تنظیم GPU device مورد استفاده"""
        if not self.gpu_available:
            logger.warning("⚠️ No GPU available to set")
            return False

        try:
            if device_id >= torch.cuda.device_count():
                logger.error(f"❌ GPU {device_id} not available")
                return False

            torch.cuda.set_device(device_id)
            logger.info(f"🎮 GPU device set to {device_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to set GPU device: {e}")
            return False

    def clear_gpu_memory(self):
        """پاکسازی حافظه GPU"""
        if not self.gpu_available:
            return

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("🧹 GPU memory cleared")

        except Exception as e:
            logger.warning(f"⚠️ GPU memory clear failed: {e}")

    def get_memory_usage(self) -> Dict[str, float]:
        """دریافت استفاده از حافظه سیستم و GPU"""
        # حافظه سیستم
        memory = psutil.virtual_memory()
        result = {
            "system_memory_gb": round(memory.total / 1024**3, 2),
            "system_used_gb": round(memory.used / 1024**3, 2),
            "system_available_gb": round(memory.available / 1024**3, 2),
            "system_usage_percent": memory.percent,
        }

        # حافظه GPU
        if self.gpu_available:
            try:
                current_gpu = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(current_gpu) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(current_gpu) / 1024**3
                memory_total = (
                    torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
                )

                result.update(
                    {
                        "gpu_memory_total_gb": round(memory_total, 2),
                        "gpu_memory_allocated_gb": round(memory_allocated, 2),
                        "gpu_memory_reserved_gb": round(memory_reserved, 2),
                        "gpu_memory_free_gb": round(memory_total - memory_reserved, 2),
                        "gpu_usage_percent": round(
                            (memory_allocated / memory_total) * 100, 1
                        ),
                    }
                )

            except Exception as e:
                logger.warning(f"⚠️ GPU memory info failed: {e}")

        return result

    def get_optimal_batch_size(self, base_batch_size: int = 1) -> int:
        """تخمین batch size بهینه بر اساس حافظه موجود"""
        if not self.gpu_available:
            return base_batch_size

        try:
            current_gpu = torch.cuda.current_device()
            memory_total = torch.cuda.get_device_properties(current_gpu).total_memory
            memory_allocated = torch.cuda.memory_allocated(current_gpu)
            memory_free = memory_total - memory_allocated

            # تخمین تقریبی: هر batch حدود 1GB حافظه نیاز دارد
            estimated_batch_size = max(1, int(memory_free / (1024**3)))

            # محدود کردن به یک حد منطقی
            optimal_batch_size = min(base_batch_size * 2, estimated_batch_size)

            logger.info(f"🎯 Optimal batch size: {optimal_batch_size}")
            return optimal_batch_size

        except Exception as e:
            logger.warning(f"⚠️ Batch size estimation failed: {e}")
            return base_batch_size

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        await self.release_gpu_from_coordinator()
        self.clear_gpu_memory()


# =======================
# Utility Functions
# =======================


def get_gpu_info_summary() -> Dict[str, Any]:
    """دریافت خلاصه اطلاعات GPU"""
    manager = GPUManager()
    return manager.get_gpu_info()


def check_gpu_compatibility() -> bool:
    """بررسی سازگاری GPU"""
    if not torch.cuda.is_available():
        return False

    try:
        # تست ساده GPU
        device = torch.cuda.current_device()
        tensor = torch.tensor([1.0], device=device)
        result = tensor * 2
        return result.item() == 2.0

    except Exception as e:
        logger.error(f"❌ GPU compatibility test failed: {e}")
        return False


# =======================
# Example Usage
# =======================


async def example_gpu_management():
    """نمونه استفاده از GPU Manager"""

    async with GPUManager("example-service") as gpu_manager:
        # دریافت اطلاعات GPU
        info = gpu_manager.get_gpu_info()
        print(f"GPU Info: {info}")

        # درخواست GPU از coordinator
        if await gpu_manager.request_gpu_via_coordinator(memory_gb=2.0):
            print("✅ GPU allocated via coordinator")

            # انجام کار با GPU
            await asyncio.sleep(2)

        # آزادسازی خودکار در __aexit__


if __name__ == "__main__":
    # تست GPU Manager
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_gpu_management())
