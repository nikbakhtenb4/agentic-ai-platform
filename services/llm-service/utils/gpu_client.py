# services\llm-service\utils\gpu_client.py
"""
GPU Client for LLM Service - FIXED VERSION
Integration with GPU Coordinator with better error handling
"""

import httpx
import asyncio
import os
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class GPUClientError(Exception):
    """Custom exception for GPU client errors"""

    pass


class SimpleGPUClient:
    def __init__(self, coordinator_url=None, service_name="llm-service"):
        # Fix URL format - ensure HTTP protocol
        raw_url = coordinator_url or os.getenv(
            "GPU_COORDINATOR_URL", "http://gpu-coordinator:8080"
        )

        # Clean up URL - remove https if present, use http for local
        if raw_url.startswith("https://"):
            raw_url = raw_url.replace("https://", "http://")
        elif not raw_url.startswith("http://"):
            raw_url = f"http://{raw_url}"

        self.coordinator_url = raw_url
        self.service_name = service_name
        self.current_task_id = None
        self.current_gpu_id = None
        self.retry_attempts = 3
        self.retry_delay = 2
        self.timeout = 30

        logger.info(f"🎮 GPU Client initialized for {service_name}")
        logger.info(f"🔗 Coordinator URL: {self.coordinator_url}")

    async def is_coordinator_available(self) -> bool:
        """Check if GPU coordinator is available"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.coordinator_url}/health")
                if response.status_code == 200:
                    logger.info("✅ GPU Coordinator is available")
                    return True
                else:
                    logger.warning(
                        f"⚠️ GPU Coordinator health check failed: {response.status_code}"
                    )
                    return False
        except httpx.TimeoutException:
            logger.warning("⏰ GPU Coordinator health check timeout")
            return False
        except Exception as e:
            logger.warning(f"⚠️ GPU Coordinator not available: {e}")
            return False

    async def wait_for_gpu(self, memory_gb=3.0, priority="normal", max_wait_time=120):
        """Wait for GPU allocation with proper response handling"""
        logger.info(f"🔄 Requesting {memory_gb}GB GPU memory...")

        # First check if coordinator is available
        if not await self.is_coordinator_available():
            logger.warning(
                "⚠️ GPU Coordinator not available, returning non-allocated response"
            )
            return type(
                "Allocation",
                (),
                {
                    "allocated": False,
                    "gpu_id": None,
                    "message": "Coordinator not available",
                },
            )()

        for attempt in range(self.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.coordinator_url}/request",
                        json={
                            "service_name": self.service_name,
                            "estimated_memory": memory_gb,
                            "priority": priority,
                            "timeout": max_wait_time,
                        },
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("allocated"):
                            self.current_task_id = data.get("task_id")
                            self.current_gpu_id = data.get("gpu_id")
                            if self.current_gpu_id is not None:
                                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                                    self.current_gpu_id
                                )
                            logger.info(
                                f"✅ GPU {self.current_gpu_id} allocated for LLM (task: {self.current_task_id})"
                            )
                            return type(
                                "Allocation",
                                (),
                                {
                                    "allocated": True,
                                    "gpu_id": self.current_gpu_id,
                                    "task_id": self.current_task_id,
                                },
                            )()
                        else:
                            logger.warning(
                                f"⚠️ GPU allocation failed: {data.get('message', 'Unknown error')}"
                            )
                            if data.get("queue_position"):
                                logger.info(
                                    f"📋 Added to queue at position {data['queue_position']}"
                                )
                    else:
                        logger.error(
                            f"❌ GPU request failed with status {response.status_code}: {response.text}"
                        )

            except httpx.TimeoutException:
                logger.error(
                    f"⏰ GPU request timeout (attempt {attempt + 1}/{self.retry_attempts})"
                )
            except Exception as e:
                logger.error(
                    f"❌ GPU request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )

            if attempt < self.retry_attempts - 1:
                wait_time = self.retry_delay * (attempt + 1)
                logger.info(f"⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

        logger.error("❌ All GPU request attempts failed")
        return type(
            "Allocation",
            (),
            {"allocated": False, "gpu_id": None, "message": "All attempts failed"},
        )()

    async def request_gpu(self, memory_gb=3.0, priority="normal", timeout_seconds=120):
        """Legacy method name for backward compatibility"""
        allocation = await self.wait_for_gpu(memory_gb, priority, timeout_seconds)
        return allocation.allocated

    async def release_gpu(self):
        """آزادسازی GPU"""
        if not self.current_task_id:
            logger.warning("⚠️ No GPU allocated to release")
            return True

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{self.coordinator_url}/release/{self.current_task_id}"
                )

                if response.status_code == 200:
                    logger.info(
                        f"✅ GPU {self.current_gpu_id} released from LLM (task: {self.current_task_id})"
                    )
                else:
                    logger.warning(
                        f"⚠️ GPU release returned status {response.status_code}: {response.text}"
                    )

                # پاکسازی state محلی
                self.current_task_id = None
                self.current_gpu_id = None
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                return True

        except Exception as e:
            logger.error(f"❌ GPU release failed: {e}")
            # Clean up local state even if release failed
            self.current_task_id = None
            self.current_gpu_id = None
            return False

    def is_gpu_allocated(self) -> bool:
        """بررسی وضعیت تخصیص GPU"""
        return self.current_task_id is not None

    def get_current_gpu_id(self) -> Optional[int]:
        """Get currently allocated GPU ID"""
        return self.current_gpu_id

    async def get_coordinator_status(self) -> Optional[Dict]:
        """دریافت وضعیت GPU Coordinator"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.coordinator_url}/status")
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        f"⚠️ Coordinator status check failed: {response.status_code}"
                    )
                    return None
        except Exception as e:
            logger.error(f"❌ Failed to get coordinator status: {e}")
        return None

    async def get_queue_status(self) -> Optional[Dict]:
        """دریافت وضعیت صف انتظار"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.coordinator_url}/queue")
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        f"⚠️ Queue status check failed: {response.status_code}"
                    )
                    return None
        except Exception as e:
            logger.error(f"❌ Failed to get queue status: {e}")
        return None


class ManagedGPU:
    """Context manager برای GPU allocation در LLM Service"""

    def __init__(self, service_name="llm-service", memory_gb=3.0, priority="normal"):
        self.client = SimpleGPUClient(service_name=service_name)
        self.memory_gb = memory_gb
        self.priority = priority
        self.allocated = False
        self.gpu_id = None

    async def __aenter__(self):
        logger.info(f"🔄 Attempting to allocate {self.memory_gb}GB GPU for LLM...")
        allocation = await self.client.wait_for_gpu(
            memory_gb=self.memory_gb, priority=self.priority
        )
        self.allocated = allocation.allocated
        self.gpu_id = allocation.gpu_id

        if self.allocated:
            logger.info(f"✅ LLM GPU allocation successful: GPU {self.gpu_id}")
        else:
            logger.warning("⚠️ LLM GPU allocation failed, using CPU fallback")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.allocated:
            await self.client.release_gpu()
            logger.info("✅ LLM GPU automatically released")


# Helper function برای LLM service
async def get_gpu_for_llm(
    memory_gb=3.0, max_wait_time=300
) -> tuple[bool, Optional[int], str]:
    """
    درخواست GPU برای LLM service با انتظار در صف

    Args:
        memory_gb: مقدار حافظه مورد نیاز
        max_wait_time: حداکثر زمان انتظار در ثانیه

    Returns:
        tuple: (success: bool, gpu_id: int|None, message: str)
    """
    client = SimpleGPUClient(service_name="llm-service")

    # تلاش اول
    allocation = await client.wait_for_gpu(memory_gb=memory_gb)
    if allocation.allocated:
        return (
            True,
            allocation.gpu_id,
            f"GPU {allocation.gpu_id} allocated immediately",
        )

    # چک کردن صف
    queue_status = await client.get_queue_status()
    if queue_status and queue_status.get("queue_length", 0) > 0:
        estimated_wait = queue_status.get("estimated_wait_time", 60)
        logger.info(f"📋 LLM service in queue, estimated wait: {estimated_wait}s")

        # انتظار تا حداکثر زمان مشخص
        wait_cycles = min(max_wait_time // 10, estimated_wait // 10)
        for cycle in range(int(wait_cycles)):
            await asyncio.sleep(10)
            allocation = await client.wait_for_gpu(memory_gb=memory_gb)
            if allocation.allocated:
                return (
                    True,
                    allocation.gpu_id,
                    f"GPU {allocation.gpu_id} allocated after {cycle * 10}s wait",
                )

    return False, None, "GPU allocation failed after timeout"
