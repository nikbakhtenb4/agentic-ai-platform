# shared/gpu_coordinator.py
"""
GPU Coordinator for Shared GPU Management
مدیر هماهنگ‌کننده GPU برای مدیریت مشترک منابع GPU بین سرویس‌ها
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import psutil
import uuid

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Types of services that can use GPU"""

    LLM = "llm"
    STT = "stt"
    TTS = "tts"
    OTHER = "other"


class GPUTaskPriority(Enum):
    """Priority levels for GPU tasks"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class GPURequest:
    """GPU resource request"""

    request_id: str
    service_type: ServiceType
    service_name: str
    priority: GPUTaskPriority
    estimated_duration: float  # in seconds
    memory_requirement: int  # in MB
    timestamp: float
    callback: Optional[callable] = None


@dataclass
class GPUAllocation:
    """GPU allocation information"""

    request_id: str
    service_type: ServiceType
    service_name: str
    allocated_memory: int
    start_time: float
    estimated_end_time: float
    actual_usage: Dict[str, Any]


class GPUCoordinator:
    """
    Coordinates GPU usage between multiple AI services
    هماهنگ‌کننده استفاده از GPU بین سرویس‌های مختلف هوش مصنوعی
    """

    def __init__(self, max_memory_mb: int = None):
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        self.current_device = 0

        # Memory management
        if self.gpu_available and max_memory_mb is None:
            # Auto-detect GPU memory
            props = torch.cuda.get_device_properties(0)
            total_memory_mb = props.total_memory // (1024 * 1024)
            # Reserve 20% for system
            self.max_memory_mb = int(total_memory_mb * 0.8)
        else:
            self.max_memory_mb = max_memory_mb or 4096  # Default 4GB

        # Request management
        self.pending_requests: List[GPURequest] = []
        self.active_allocations: Dict[str, GPUAllocation] = {}
        self.request_history: List[GPURequest] = []

        # Threading
        self.lock = threading.RLock()
        self.scheduler_running = False
        self.scheduler_task = None

        # Statistics
        self.stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "average_wait_time": 0.0,
            "average_execution_time": 0.0,
            "memory_peak_usage": 0,
            "service_usage": {},
        }

        logger.info(f"🎮 GPU Coordinator initialized")
        logger.info(f"   GPU Available: {self.gpu_available}")
        logger.info(f"   Device Count: {self.device_count}")
        logger.info(f"   Max Memory: {self.max_memory_mb}MB")

        if self.gpu_available:
            self._initialize_gpu_monitoring()

    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring"""
        try:
            # Try to initialize NVML for detailed GPU monitoring
            try:
                import pynvml

                pynvml.nvmlInit()
                self.nvml_available = True
                logger.info("✅ NVML initialized for GPU monitoring")
            except ImportError:
                self.nvml_available = False
                logger.info("⚠️ NVML not available, using basic GPU monitoring")

            # Clear any existing GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"GPU monitoring initialization failed: {e}")

    async def request_gpu(
        self,
        service_type: ServiceType,
        service_name: str,
        priority: GPUTaskPriority = GPUTaskPriority.NORMAL,
        estimated_duration: float = 30.0,
        memory_requirement: int = 1024,
        callback: Optional[callable] = None,
    ) -> str:
        """
        Request GPU access for a service

        Args:
            service_type: Type of service requesting GPU
            service_name: Name of the requesting service
            priority: Priority level
            estimated_duration: Estimated execution time in seconds
            memory_requirement: Required memory in MB
            callback: Optional callback when GPU is available

        Returns:
            Request ID
        """
        request_id = str(uuid.uuid4())

        with self.lock:
            request = GPURequest(
                request_id=request_id,
                service_type=service_type,
                service_name=service_name,
                priority=priority,
                estimated_duration=estimated_duration,
                memory_requirement=memory_requirement,
                timestamp=time.time(),
                callback=callback,
            )

            self.pending_requests.append(request)
            self.stats["total_requests"] += 1

            # Sort by priority and timestamp
            self.pending_requests.sort(key=lambda r: (-r.priority.value, r.timestamp))

            logger.info(
                f"📋 GPU request queued: {service_name} ({service_type.value}, "
                f"priority={priority.name}, memory={memory_requirement}MB)"
            )

        # Start scheduler if not running
        if not self.scheduler_running:
            await self._start_scheduler()

        return request_id

    async def release_gpu(self, request_id: str) -> bool:
        """
        Release GPU allocation

        Args:
            request_id: ID of the request to release

        Returns:
            True if successfully released
        """
        with self.lock:
            if request_id in self.active_allocations:
                allocation = self.active_allocations[request_id]
                del self.active_allocations[request_id]

                # Update statistics
                execution_time = time.time() - allocation.start_time
                self.stats["completed_requests"] += 1
                self._update_average_execution_time(execution_time)

                logger.info(
                    f"🔓 GPU released: {allocation.service_name} "
                    f"(duration: {execution_time:.2f}s)"
                )

                # Clear GPU memory
                if self.gpu_available:
                    torch.cuda.empty_cache()

                return True
            else:
                logger.warning(f"⚠️ GPU release failed: Request {request_id} not found")
                return False

    async def _start_scheduler(self):
        """Start the GPU scheduler task"""
        if not self.scheduler_running:
            self.scheduler_running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info("🚀 GPU scheduler started")

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        try:
            while self.scheduler_running:
                await self._process_requests()
                await asyncio.sleep(1.0)  # Check every second
        except Exception as e:
            logger.error(f"GPU scheduler error: {e}")
        finally:
            self.scheduler_running = False
            logger.info("🛑 GPU scheduler stopped")

    async def _process_requests(self):
        """Process pending GPU requests"""
        with self.lock:
            if not self.pending_requests:
                return

            # Check if GPU is available
            available_memory = self._get_available_memory()

            # Process highest priority requests first
            processed_requests = []

            for request in self.pending_requests:
                if self._can_allocate_gpu(request, available_memory):
                    # Allocate GPU
                    allocation = GPUAllocation(
                        request_id=request.request_id,
                        service_type=request.service_type,
                        service_name=request.service_name,
                        allocated_memory=request.memory_requirement,
                        start_time=time.time(),
                        estimated_end_time=time.time() + request.estimated_duration,
                        actual_usage={},
                    )

                    self.active_allocations[request.request_id] = allocation
                    processed_requests.append(request)

                    # Update statistics
                    wait_time = time.time() - request.timestamp
                    self._update_average_wait_time(wait_time)
                    self._update_service_usage(
                        request.service_type, request.service_name
                    )

                    logger.info(
                        f"🎮 GPU allocated: {request.service_name} "
                        f"(wait: {wait_time:.2f}s)"
                    )

                    # Call callback if provided
                    if request.callback:
                        try:
                            if asyncio.iscoroutinefunction(request.callback):
                                await request.callback(request.request_id, True)
                            else:
                                request.callback(request.request_id, True)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                    available_memory -= request.memory_requirement

                    # Only allocate one GPU per cycle to avoid conflicts
                    break

            # Remove processed requests
            for request in processed_requests:
                self.pending_requests.remove(request)
                self.request_history.append(request)

                # Keep history limited
                if len(self.request_history) > 1000:
                    self.request_history = self.request_history[-500:]

    def _can_allocate_gpu(self, request: GPURequest, available_memory: int) -> bool:
        """Check if GPU can be allocated for request"""

        if not self.gpu_available:
            return False

        # Check memory requirement
        if request.memory_requirement > available_memory:
            return False

        # Check if there are higher priority active allocations
        for allocation in self.active_allocations.values():
            # If there's a critical task running, only allow other critical tasks
            if (
                allocation.service_type == ServiceType.LLM
                and request.service_type != ServiceType.LLM
                and request.priority != GPUTaskPriority.CRITICAL
            ):
                return False

        return True

    def _get_available_memory(self) -> int:
        """Get available GPU memory in MB"""
        if not self.gpu_available:
            return 0

        try:
            # Calculate used memory from active allocations
            allocated_memory = sum(
                alloc.allocated_memory for alloc in self.active_allocations.values()
            )

            return max(0, self.max_memory_mb - allocated_memory)

        except Exception as e:
            logger.error(f"Error getting available memory: {e}")
            return 0

    def _update_average_wait_time(self, wait_time: float):
        """Update average wait time statistic"""
        current_avg = self.stats["average_wait_time"]
        completed = self.stats["completed_requests"]

        if completed <= 1:
            self.stats["average_wait_time"] = wait_time
        else:
            self.stats["average_wait_time"] = (
                current_avg * (completed - 1) + wait_time
            ) / completed

    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time statistic"""
        current_avg = self.stats["average_execution_time"]
        completed = self.stats["completed_requests"]

        if completed <= 1:
            self.stats["average_execution_time"] = execution_time
        else:
            self.stats["average_execution_time"] = (
                current_avg * (completed - 1) + execution_time
            ) / completed

    def _update_service_usage(self, service_type: ServiceType, service_name: str):
        """Update service usage statistics"""
        key = f"{service_type.value}_{service_name}"

        if key not in self.stats["service_usage"]:
            self.stats["service_usage"][key] = 0

        self.stats["service_usage"][key] += 1

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status"""
        status = {
            "gpu_available": self.gpu_available,
            "device_count": self.device_count,
            "max_memory_mb": self.max_memory_mb,
            "available_memory_mb": self._get_available_memory(),
            "pending_requests": len(self.pending_requests),
            "active_allocations": len(self.active_allocations),
            "scheduler_running": self.scheduler_running,
        }

        # Add GPU utilization if available
        if self.gpu_available:
            try:
                memory_info = self._get_gpu_memory_info()
                if memory_info:
                    status.update(memory_info)
            except Exception as e:
                logger.warning(f"Could not get GPU utilization: {e}")

        return status

    def _get_gpu_memory_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed GPU memory information"""
        try:
            if self.nvml_available:
                import pynvml

                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                return {
                    "memory_used_mb": memory_info.used // (1024 * 1024),
                    "memory_total_mb": memory_info.total // (1024 * 1024),
                    "memory_free_mb": memory_info.free // (1024 * 1024),
                    "gpu_utilization": util.gpu,
                    "memory_utilization": util.memory,
                }
            else:
                # Fallback to PyTorch memory info
                if torch.cuda.is_available():
                    return {
                        "memory_allocated_mb": torch.cuda.memory_allocated()
                        // (1024 * 1024),
                        "memory_reserved_mb": torch.cuda.memory_reserved()
                        // (1024 * 1024),
                        "memory_cached_mb": torch.cuda.memory_cached() // (1024 * 1024),
                    }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")

        return None

    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status information"""
        with self.lock:
            pending_by_service = {}
            pending_by_priority = {}

            for request in self.pending_requests:
                # By service
                service_key = f"{request.service_type.value}_{request.service_name}"
                pending_by_service[service_key] = (
                    pending_by_service.get(service_key, 0) + 1
                )

                # By priority
                priority_key = request.priority.name
                pending_by_priority[priority_key] = (
                    pending_by_priority.get(priority_key, 0) + 1
                )

            active_by_service = {}
            for allocation in self.active_allocations.values():
                service_key = (
                    f"{allocation.service_type.value}_{allocation.service_name}"
                )
                active_by_service[service_key] = (
                    active_by_service.get(service_key, 0) + 1
                )

            return {
                "pending_requests": {
                    "total": len(self.pending_requests),
                    "by_service": pending_by_service,
                    "by_priority": pending_by_priority,
                },
                "active_allocations": {
                    "total": len(self.active_allocations),
                    "by_service": active_by_service,
                },
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = self.stats.copy()

        # Add queue information
        stats.update(self.get_queue_status())

        # Add GPU status
        stats.update(self.get_gpu_status())

        return stats

    async def cleanup_expired_allocations(self):
        """Clean up expired allocations (safety mechanism)"""
        current_time = time.time()
        expired_allocations = []

        with self.lock:
            for request_id, allocation in self.active_allocations.items():
                # If allocation is way past estimated end time (with 50% buffer)
                buffer_time = (
                    allocation.estimated_end_time
                    + (allocation.estimated_end_time - allocation.start_time) * 0.5
                )

                if current_time > buffer_time:
                    expired_allocations.append(request_id)
                    logger.warning(
                        f"⚠️ Cleaning up expired GPU allocation: {allocation.service_name}"
                    )

            for request_id in expired_allocations:
                await self.release_gpu(request_id)

    async def stop(self):
        """Stop the GPU coordinator"""
        self.scheduler_running = False

        if self.scheduler_task:
            await self.scheduler_task

        # Release all active allocations
        with self.lock:
            for request_id in list(self.active_allocations.keys()):
                await self.release_gpu(request_id)

        logger.info("🛑 GPU Coordinator stopped")


# Global GPU coordinator instance
gpu_coordinator: Optional[GPUCoordinator] = None


def get_gpu_coordinator() -> GPUCoordinator:
    """Get global GPU coordinator instance"""
    global gpu_coordinator

    if gpu_coordinator is None:
        gpu_coordinator = GPUCoordinator()

    return gpu_coordinator


async def initialize_gpu_coordinator(max_memory_mb: int = None):
    """Initialize global GPU coordinator"""
    global gpu_coordinator

    if gpu_coordinator is None:
        gpu_coordinator = GPUCoordinator(max_memory_mb=max_memory_mb)
        await gpu_coordinator._start_scheduler()

    return gpu_coordinator


class GPUContextManager:
    """Context manager for GPU allocation"""

    def __init__(
        self,
        service_type: ServiceType,
        service_name: str,
        priority: GPUTaskPriority = GPUTaskPriority.NORMAL,
        estimated_duration: float = 30.0,
        memory_requirement: int = 1024,
    ):
        self.service_type = service_type
        self.service_name = service_name
        self.priority = priority
        self.estimated_duration = estimated_duration
        self.memory_requirement = memory_requirement
        self.request_id = None
        self.coordinator = get_gpu_coordinator()

    async def __aenter__(self):
        """Request GPU on context enter"""
        self.request_id = await self.coordinator.request_gpu(
            service_type=self.service_type,
            service_name=self.service_name,
            priority=self.priority,
            estimated_duration=self.estimated_duration,
            memory_requirement=self.memory_requirement,
        )

        # Wait for allocation
        max_wait_time = 300  # 5 minutes max wait
        wait_time = 0

        while wait_time < max_wait_time:
            if self.request_id in self.coordinator.active_allocations:
                logger.info(f"🎮 GPU allocated for {self.service_name}")
                return self.request_id

            await asyncio.sleep(1)
            wait_time += 1

        raise TimeoutError(f"GPU allocation timeout for {self.service_name}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release GPU on context exit"""
        if self.request_id:
            await self.coordinator.release_gpu(self.request_id)
            logger.info(f"🔓 GPU released for {self.service_name}")
