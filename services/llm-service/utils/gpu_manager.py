import torch
import logging

logger = logging.getLogger(__name__)


class GPUManager:
    def __init__(self):
        try:
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.device_count = torch.cuda.device_count()
            else:
                self.device_count = 0
        except:
            self.gpu_available = False
            self.device_count = 0

    def get_gpu_info(self):
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_gpu = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(current_gpu) / 1024**3
                memory_total = (
                    torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
                )

                return {
                    "available": True,
                    "count": gpu_count,
                    "current_gpu": current_gpu,
                    "memory_info": {
                        "allocated_gb": round(memory_allocated, 2),
                        "total_gb": round(memory_total, 2),
                        "utilization_percent": round(
                            (memory_allocated / memory_total) * 100, 1
                        ),
                    },
                    "utilization": round((memory_allocated / memory_total) * 100, 1),
                }
        except:
            pass

        return {
            "available": False,
            "count": 0,
            "current_gpu": None,
            "memory_info": {},
            "utilization": 0,
        }
