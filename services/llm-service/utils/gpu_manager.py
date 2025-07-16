# ===============================
# services/llm-service/utils/gpu_manager.py
# ===============================
import logging
import torch
from typing import Dict, Optional, Any
import psutil

logger = logging.getLogger(__name__)

class GPUManager:
    """GPU management utility"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.current_device = torch.cuda.current_device() if self.gpu_available else None
        
        if self.gpu_available:
            logger.info(f"✅ GPU available: {self.gpu_count} device(s)")
            for i in range(self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  Device {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
        else:
            logger.info("ℹ️ GPU not available, using CPU")
    
    def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information"""
        if not self.gpu_available:
            return None
        
        try:
            # Try to get GPU utilization using nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                return {
                    "utilization": util.gpu,
                    "memory_utilization": util.memory,
                    "memory_used": memory_info.used,
                    "memory_total": memory_info.total,
                    "memory_free": memory_info.free
                }
            except ImportError:
                # Fallback to torch-only information
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated()
                    memory_reserved = torch.cuda.memory_reserved()
                    
                    return {
                        "utilization": 0,  # Can't get utilization without nvidia-ml-py
                        "memory_allocated": memory_allocated,
                        "memory_reserved": memory_reserved,
                        "memory_allocated_gb": memory_allocated / 1024**3,
                        "memory_reserved_gb": memory_reserved / 1024**3
                    }
                
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get system memory usage"""
        memory = psutil.virtual_memory()
        return {
            "used_gb": round(memory.used / 1024**3, 2),
            "available_gb": round(memory.available / 1024**3, 2),
            "percent": memory.percent,
            "total_gb": round(memory.total / 1024**3, 2)
        }
    
    def clear_gpu_cache(self):
        """Clear GPU cache"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            logger.info("🧹 GPU cache cleared")
    
    def set_device(self, device_id: int = 0):
        """Set current GPU device"""
        if self.gpu_available and device_id < self.gpu_count:
            torch.cuda.set_device(device_id)
            self.current_device = device_id
            logger.info(f"🔧 GPU device set to: {device_id}")
        else:
            logger.warning(f"⚠️ Cannot set GPU device {device_id}")
    
    def get_optimal_device(self) -> str:
        """Get optimal device for model loading"""
        if self.gpu_available:
            return f"cuda:{self.current_device}"
        return "cpu"
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        info = {
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "current_device": self.current_device,
            "optimal_device": self.get_optimal_device(),
            "memory_usage": self.get_memory_usage()
        }
        
        if self.gpu_available:
            gpu_info = self.get_gpu_info()
            if gpu_info:
                info["gpu_stats"] = gpu_info
        
        return info