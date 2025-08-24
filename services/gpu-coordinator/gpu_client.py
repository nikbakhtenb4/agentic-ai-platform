# services/gpu-coordinator/gpu_client.py
"""
Simple GPU Client for services - Fixed Version
"""

import httpx
import asyncio
import os
import logging
import time

logger = logging.getLogger(__name__)


class SimpleGPUClient:
    def __init__(self, coordinator_url=None, service_name="unknown"):
        # 🔧 Fixed: استفاده از http بجای https
        self.coordinator_url = coordinator_url or os.getenv(
            "GPU_COORDINATOR_URL", "http://gpu-coordinator:8080"
        )
        self.service_name = service_name
        self.current_task_id = None
        self.current_gpu_id = None
        self.retry_attempts = 3
        self.retry_delay = 1

    async def request_gpu(self, memory_gb=2.0, priority="normal", timeout_seconds=30):
        """درخواست GPU با retry mechanism"""
        for attempt in range(self.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        f"{self.coordinator_url}/request",
                        json={
                            "service_name": self.service_name,
                            "estimated_memory": memory_gb,
                            "priority": priority,
                            "timeout": timeout_seconds,
                        },
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("allocated"):
                            self.current_task_id = data["task_id"]
                            self.current_gpu_id = data["gpu_id"]
                            os.environ["CUDA_VISIBLE_DEVICES"] = str(data["gpu_id"])
                            logger.info(
                                f"✅ GPU {data['gpu_id']} allocated (task: {self.current_task_id})"
                            )
                            return True
                        else:
                            logger.warning(
                                f"⚠️ GPU allocation failed: {data.get('message', 'Unknown error')}"
                            )
                    else:
                        logger.error(
                            f"❌ GPU request failed with status {response.status_code}"
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
                await asyncio.sleep(self.retry_delay)

        logger.error("❌ All GPU request attempts failed")
        return False

    async def release_gpu(self):
        """آزادسازی GPU"""
        if not self.current_task_id:
            logger.warning("⚠️ No GPU allocated to release")
            return True

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{self.coordinator_url}/release/{self.current_task_id}"
                )

                if response.status_code == 200:
                    logger.info(
                        f"✅ GPU {self.current_gpu_id} released (task: {self.current_task_id})"
                    )
                else:
                    logger.warning(
                        f"⚠️ GPU release returned status {response.status_code}"
                    )

                # پاکسازی state محلی
                self.current_task_id = None
                self.current_gpu_id = None
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                return True

        except Exception as e:
            logger.error(f"❌ GPU release failed: {e}")
            return False

    def is_gpu_allocated(self):
        """بررسی وضعیت تخصیص GPU"""
        return self.current_task_id is not None

    async def get_gpu_status(self):
        """دریافت وضعیت GPU"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.coordinator_url}/status")
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"❌ Failed to get GPU status: {e}")
        return None


class ManagedGPU:
    """Context manager برای GPU allocation"""

    def __init__(self, service_name, memory_gb=2.0, priority="normal"):
        self.client = SimpleGPUClient(service_name=service_name)
        self.memory_gb = memory_gb
        self.priority = priority
        self.allocated = False
        self.gpu_id = None

    async def __aenter__(self):
        self.allocated = await self.client.request_gpu(
            memory_gb=self.memory_gb, priority=self.priority
        )
        self.gpu_id = self.client.current_gpu_id
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.allocated:
            await self.client.release_gpu()


# Helper function برای backward compatibility
async def managed_gpu(service_name, memory_gb=2.0):
    """Context manager ساده برای GPU"""
    async with ManagedGPU(service_name, memory_gb) as gpu_mgr:
        yield gpu_mgr
