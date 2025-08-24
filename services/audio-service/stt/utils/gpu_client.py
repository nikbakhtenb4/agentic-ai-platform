# services/audio-service/stt/utils/gpu_client.py
"""
GPU Client for STT Service (Speech-to-Text)
Integration with GPU Coordinator for enhanced resource management
مدیریت GPU برای سرویس تبدیل گفتار به متن
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
    """GPU Client specifically optimized for STT workloads"""

    def __init__(self, coordinator_url=None, service_name="stt-service"):
        # Fix URL format - ensure HTTP protocol
        raw_url = coordinator_url or os.getenv(
            "GPU_COORDINATOR_URL", "http://gpu-coordinator:8080"
        )

        # Clean up URL - remove https if present, use http for internal services
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

        # STT-specific configuration
        self.default_memory_gb = float(os.getenv("STT_GPU_MEMORY_GB", "2.0"))
        self.whisper_models = {
            "tiny": 1.0,  # GB memory requirement
            "base": 1.5,
            "small": 2.0,
            "medium": 3.0,
            "large": 4.0,
            "large-v2": 4.5,
            "large-v3": 5.0,
        }

        logger.info(f"🎤 STT GPU Client initialized for {service_name}")
        logger.info(f"🔗 Coordinator URL: {self.coordinator_url}")
        logger.info(f"💾 Default memory requirement: {self.default_memory_gb}GB")

    def get_memory_requirement_for_model(self, model_name: str) -> float:
        """Get memory requirement for specific Whisper model"""
        return self.whisper_models.get(model_name, self.default_memory_gb)

    async def is_coordinator_available(self) -> bool:
        """Check if GPU coordinator is available"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.coordinator_url}/health")
                if response.status_code == 200:
                    logger.info("✅ GPU Coordinator is available for STT")
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
            logger.warning(f"⚠️ GPU Coordinator not available for STT: {e}")
            return False

    async def wait_for_gpu(
        self,
        memory_gb: Optional[float] = None,
        model_name: Optional[str] = None,
        priority: str = "normal",
        max_wait_time: int = 120,
    ):
        """
        Wait for GPU allocation optimized for STT workloads

        Args:
            memory_gb: Required memory in GB (auto-calculated if model_name provided)
            model_name: Whisper model name (tiny, base, small, medium, large, etc.)
            priority: Priority level (low, normal, high)
            max_wait_time: Maximum wait time in seconds
        """
        # Determine memory requirement
        if model_name:
            memory_gb = self.get_memory_requirement_for_model(model_name)
            logger.info(f"🎯 Memory requirement for {model_name}: {memory_gb}GB")
        elif memory_gb is None:
            memory_gb = self.default_memory_gb

        logger.info(f"📋 Requesting {memory_gb}GB GPU for STT processing...")

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
                    "model_compatible": False,
                },
            )()

        for attempt in range(self.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    request_payload = {
                        "service_name": self.service_name,
                        "estimated_memory": memory_gb,
                        "priority": priority,
                        "timeout": max_wait_time,
                    }

                    # Add STT-specific metadata
                    if model_name:
                        request_payload["metadata"] = {
                            "whisper_model": model_name,
                            "task_type": "speech_to_text",
                        }

                    response = await client.post(
                        f"{self.coordinator_url}/request", json=request_payload
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
                                f"✅ GPU {self.current_gpu_id} allocated for STT (task: {self.current_task_id})"
                            )

                            # Verify GPU compatibility for the model
                            model_compatible = await self._verify_model_compatibility(
                                model_name
                            )

                            return type(
                                "Allocation",
                                (),
                                {
                                    "allocated": True,
                                    "gpu_id": self.current_gpu_id,
                                    "task_id": self.current_task_id,
                                    "memory_allocated": memory_gb,
                                    "model_name": model_name,
                                    "model_compatible": model_compatible,
                                    "expires_at": data.get("expires_at"),
                                },
                            )()
                        else:
                            logger.warning(
                                f"⚠️ STT GPU allocation failed: {data.get('message', 'Unknown error')}"
                            )
                            if data.get("queue_position"):
                                logger.info(
                                    f"🔄 STT service added to queue at position {data['queue_position']}"
                                )
                                logger.info(
                                    f"⏱️ Estimated wait time: {data.get('estimated_wait_time', 60)}s"
                                )
                    else:
                        logger.error(
                            f"❌ STT GPU request failed with status {response.status_code}: {response.text}"
                        )

            except httpx.TimeoutException:
                logger.error(
                    f"⏰ STT GPU request timeout (attempt {attempt + 1}/{self.retry_attempts})"
                )
            except Exception as e:
                logger.error(
                    f"❌ STT GPU request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )

            if attempt < self.retry_attempts - 1:
                wait_time = self.retry_delay * (attempt + 1)
                logger.info(f"⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

        logger.error("❌ All STT GPU request attempts failed")
        return type(
            "Allocation",
            (),
            {
                "allocated": False,
                "gpu_id": None,
                "message": "All attempts failed",
                "model_compatible": False,
            },
        )()

    async def _verify_model_compatibility(self, model_name: Optional[str]) -> bool:
        """Verify if allocated GPU is compatible with requested model"""
        if not model_name or not self.current_gpu_id:
            return True

        try:
            import torch

            if torch.cuda.is_available():
                gpu_properties = torch.cuda.get_device_properties(self.current_gpu_id)
                gpu_memory_gb = gpu_properties.total_memory / (1024**3)
                required_memory = self.get_memory_requirement_for_model(model_name)

                is_compatible = gpu_memory_gb >= required_memory
                logger.info(f"🔍 GPU {self.current_gpu_id} compatibility check:")
                logger.info(
                    f"   Available: {gpu_memory_gb:.1f}GB, Required: {required_memory:.1f}GB"
                )
                logger.info(f"   Compatible: {'✅' if is_compatible else '❌'}")

                return is_compatible
        except Exception as e:
            logger.warning(f"⚠️ Could not verify model compatibility: {e}")

        return True  # Assume compatible if can't verify

    async def request_gpu(self, memory_gb=2.0, priority="normal", timeout_seconds=120):
        """Legacy method name for backward compatibility"""
        allocation = await self.wait_for_gpu(memory_gb, None, priority, timeout_seconds)
        return allocation.allocated

    async def release_gpu(self):
        """Release allocated GPU from STT service"""
        if not self.current_task_id:
            logger.warning("⚠️ No GPU allocated to release from STT service")
            return True

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{self.coordinator_url}/release/{self.current_task_id}"
                )

                if response.status_code == 200:
                    logger.info(
                        f"✅ GPU {self.current_gpu_id} released from STT (task: {self.current_task_id})"
                    )
                else:
                    logger.warning(
                        f"⚠️ STT GPU release returned status {response.status_code}: {response.text}"
                    )

                # Clean up local state
                released_gpu_id = self.current_gpu_id
                self.current_task_id = None
                self.current_gpu_id = None

                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]

                logger.info(f"🧹 STT GPU {released_gpu_id} local state cleaned up")
                return True

        except Exception as e:
            logger.error(f"❌ STT GPU release failed: {e}")
            # Clean up local state even if release failed
            self.current_task_id = None
            self.current_gpu_id = None
            return False

    def is_gpu_allocated(self) -> bool:
        """Check if GPU is currently allocated for STT"""
        return self.current_task_id is not None

    def get_current_gpu_id(self) -> Optional[int]:
        """Get currently allocated GPU ID"""
        return self.current_gpu_id

    def get_allocated_memory(self) -> Optional[float]:
        """Get allocated memory amount"""
        # This could be enhanced to track actual allocated memory
        return self.default_memory_gb if self.is_gpu_allocated() else None

    async def extend_gpu_allocation(self, additional_time: int = 300):
        """Extend current GPU allocation time"""
        if not self.current_task_id:
            logger.warning("⚠️ No active GPU allocation to extend")
            return False

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{self.coordinator_url}/extend/{self.current_task_id}",
                    json={"additional_seconds": additional_time},
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"⏰ STT GPU allocation extended by {additional_time}s")
                    logger.info(f"🔄 New expiry: {data.get('new_expires_at')}")
                    return True
                else:
                    logger.warning(
                        f"⚠️ Failed to extend STT GPU allocation: {response.status_code}"
                    )
                    return False

        except Exception as e:
            logger.error(f"❌ GPU allocation extension failed: {e}")
            return False

    async def get_coordinator_status(self) -> Optional[Dict]:
        """Get GPU Coordinator status"""
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
        """Get queue status for STT service"""
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

    async def get_stt_optimized_gpu(self, model_name: str = "base") -> Optional[Dict]:
        """Get GPU optimized for specific STT model"""
        try:
            coordinator_status = await self.get_coordinator_status()
            if not coordinator_status:
                return None

            # Find GPUs with enough memory for the model
            required_memory = self.get_memory_requirement_for_model(model_name)
            suitable_gpus = []

            for gpu_id, gpu_info in coordinator_status.get("gpus", {}).items():
                if (
                    gpu_info.get("available", False)
                    and gpu_info.get("memory_free_gb", 0) >= required_memory
                ):
                    suitable_gpus.append(
                        {
                            "gpu_id": int(gpu_id),
                            "free_memory": gpu_info.get("memory_free_gb", 0),
                            "utilization": gpu_info.get("utilization_percent", 100),
                            "running_tasks": gpu_info.get("running_tasks", 0),
                            "score": self._calculate_suitability_score(
                                gpu_info, required_memory
                            ),
                        }
                    )

            if suitable_gpus:
                # Sort by suitability score (higher is better)
                suitable_gpus.sort(key=lambda x: x["score"], reverse=True)
                return suitable_gpus[0]

        except Exception as e:
            logger.error(f"❌ Failed to find optimized GPU: {e}")

        return None

    def _calculate_suitability_score(
        self, gpu_info: dict, required_memory: float
    ) -> float:
        """Calculate suitability score for STT workload"""
        free_memory = gpu_info.get("memory_free_gb", 0)
        utilization = gpu_info.get("utilization_percent", 100)
        running_tasks = gpu_info.get("running_tasks", 10)

        # Score based on available memory (higher is better)
        memory_score = min(free_memory / required_memory, 3.0)  # Cap at 3x requirement

        # Score based on low utilization (lower is better)
        util_score = (100 - utilization) / 100

        # Score based on fewer running tasks (lower is better)
        task_score = max(0, 5 - running_tasks) / 5

        # Weighted combination
        total_score = (memory_score * 0.5) + (util_score * 0.3) + (task_score * 0.2)

        return total_score


class ManagedGPU:
    """Context manager for STT GPU allocation with enhanced features"""

    def __init__(
        self,
        service_name="stt-service",
        memory_gb: Optional[float] = None,
        model_name: Optional[str] = None,
        priority: str = "normal",
        auto_extend: bool = True,
    ):
        self.client = SimpleGPUClient(service_name=service_name)
        self.memory_gb = memory_gb
        self.model_name = model_name
        self.priority = priority
        self.auto_extend = auto_extend
        self.allocated = False
        self.gpu_id = None
        self.allocation_info = None

    async def __aenter__(self):
        logger.info(f"🔄 Attempting to allocate GPU for STT...")
        if self.model_name:
            logger.info(f"🎯 Target model: {self.model_name}")
        if self.memory_gb:
            logger.info(f"💾 Memory requirement: {self.memory_gb}GB")

        allocation = await self.client.wait_for_gpu(
            memory_gb=self.memory_gb, model_name=self.model_name, priority=self.priority
        )

        self.allocated = allocation.allocated
        self.gpu_id = allocation.gpu_id
        self.allocation_info = allocation

        if self.allocated:
            logger.info(f"✅ STT GPU allocation successful: GPU {self.gpu_id}")
            if (
                hasattr(allocation, "model_compatible")
                and not allocation.model_compatible
            ):
                logger.warning("⚠️ GPU may not be optimal for requested model")
        else:
            logger.warning("⚠️ STT GPU allocation failed, will use CPU fallback")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.allocated:
            await self.client.release_gpu()
            logger.info("✅ STT GPU automatically released")

    def is_model_compatible(self) -> bool:
        """Check if allocated GPU is compatible with requested model"""
        if not self.allocation_info:
            return True
        return getattr(self.allocation_info, "model_compatible", True)

    async def extend_allocation(self, additional_time: int = 300) -> bool:
        """Extend current allocation"""
        if self.allocated:
            return await self.client.extend_gpu_allocation(additional_time)
        return False


# Helper functions for STT service
async def get_gpu_for_stt(
    model_name: str = "base",
    memory_gb: Optional[float] = None,
    max_wait_time: int = 300,
    priority: str = "normal",
) -> tuple[bool, Optional[int], str, bool]:
    """
    Request GPU for STT service with model-specific optimization

    Args:
        model_name: Whisper model name (tiny, base, small, medium, large, etc.)
        memory_gb: Override memory requirement (calculated from model if None)
        max_wait_time: Maximum wait time in seconds
        priority: Request priority (low, normal, high)

    Returns:
        tuple: (success: bool, gpu_id: int|None, message: str, model_compatible: bool)
    """
    client = SimpleGPUClient(service_name="stt-service")

    # Try immediate allocation
    allocation = await client.wait_for_gpu(
        memory_gb=memory_gb,
        model_name=model_name,
        priority=priority,
        max_wait_time=max_wait_time,
    )

    if allocation.allocated:
        model_compatible = getattr(allocation, "model_compatible", True)
        return (
            True,
            allocation.gpu_id,
            f"GPU {allocation.gpu_id} allocated for {model_name} model",
            model_compatible,
        )

    # Check queue if not immediately available
    queue_status = await client.get_queue_status()
    if queue_status and queue_status.get("queue_length", 0) > 0:
        estimated_wait = queue_status.get("estimated_wait_time", 60)
        logger.info(f"🔄 STT service in queue, estimated wait: {estimated_wait}s")

        # Wait with periodic checks
        if estimated_wait <= max_wait_time:
            wait_cycles = min(max_wait_time // 10, estimated_wait // 10)
            for cycle in range(int(wait_cycles)):
                await asyncio.sleep(10)
                allocation = await client.wait_for_gpu(
                    memory_gb=memory_gb, model_name=model_name, priority=priority
                )
                if allocation.allocated:
                    model_compatible = getattr(allocation, "model_compatible", True)
                    return (
                        True,
                        allocation.gpu_id,
                        f"GPU {allocation.gpu_id} allocated after {cycle * 10}s wait",
                        model_compatible,
                    )

    return (
        False,
        None,
        f"GPU allocation failed for {model_name} model after timeout",
        False,
    )


async def get_optimal_gpu_for_model(model_name: str) -> Optional[Dict]:
    """Find the most suitable GPU for a specific Whisper model"""
    client = SimpleGPUClient(service_name="stt-service")
    return await client.get_stt_optimized_gpu(model_name)


# Pre-defined configurations for common STT scenarios
STT_GPU_CONFIGS = {
    "realtime_transcription": {
        "model_name": "base",
        "priority": "high",
        "memory_gb": 2.0,
        "description": "Fast response for real-time applications",
    },
    "batch_processing": {
        "model_name": "small",
        "priority": "normal",
        "memory_gb": 2.5,
        "description": "Balanced accuracy/speed for batch jobs",
    },
    "high_accuracy": {
        "model_name": "medium",
        "priority": "normal",
        "memory_gb": 3.5,
        "description": "Higher accuracy for critical transcriptions",
    },
    "maximum_accuracy": {
        "model_name": "large-v3",
        "priority": "low",
        "memory_gb": 5.0,
        "description": "Best possible accuracy, slower processing",
    },
    "low_resource": {
        "model_name": "tiny",
        "priority": "normal",
        "memory_gb": 1.0,
        "description": "Minimal resources for simple transcription",
    },
}


async def get_gpu_with_config(
    config_name: str,
) -> tuple[bool, Optional[int], str, bool]:
    """Get GPU using predefined configuration"""
    if config_name not in STT_GPU_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. Available: {list(STT_GPU_CONFIGS.keys())}"
        )

    config = STT_GPU_CONFIGS[config_name]
    logger.info(f"🎯 Using STT config '{config_name}': {config['description']}")

    return await get_gpu_for_stt(
        model_name=config["model_name"],
        memory_gb=config["memory_gb"],
        priority=config["priority"],
    )
