# ===============================
# services/llm-service/models/loader.py
# ===============================
# و وظیفه‌اش بارگذاری و مدیریت مدل LLM به‌صورت محلی است.
# بارگذاری مدل و tokenizer از روی دیسک (لوکال)
# قراردادن مدل روی GPU یا CPU
# پاکسازی حافظه در صورت نیاز
# برگرداندن اطلاعات مدل (تعداد پارامترها، نوع و...)
# امکان ری‌لود مدل

import os
import torch
import logging
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.model_name: str = "gpt2-fa"
        # تصحیح مسیر مدل - مسیر پایه بدون تکرار نام مدل
        self.model_path: str = os.getenv("MODEL_PATH", "/app/models")
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded: bool = False
        
    async def initialize(self):
        """Initialize and load the model"""
        try:
            # تصحیح مسیر مدل
            model_dir = Path(self.model_path) / self.model_name
            
            logger.info(f"🔄 Loading model {self.model_name} from {model_dir}")
            
            if not model_dir.exists():
                # اگر مسیر با نام مدل وجود نداشت، بررسی کن که آیا خود model_path همان مسیر مدل است
                if Path(self.model_path).exists() and self._is_model_directory(Path(self.model_path)):
                    model_dir = Path(self.model_path)
                    logger.info(f"Using direct model path: {model_dir}")
                else:
                    raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False
            )
            
            # Add pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info(f"Loading model on {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=False,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"✅ Model {self.model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _is_model_directory(self, path: Path) -> bool:
        """Check if the given path is a model directory"""
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        return all((path / file).exists() for file in required_files)
    
    async def reload(self):
        """Reload the model"""
        await self.cleanup()
        await self.initialize()
    
    async def cleanup(self):
        """Clean up model and free memory"""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info("🧹 Model cleanup completed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information - returns only parameter info for ModelInfo.parameters field"""
        if not self.is_loaded:
            return {
                "total_parameters": 0,
                "trainable_parameters": 0,
                "device": self.device,
                "model_type": "Unknown",
                "vocab_size": 0,
                "status": "not_loaded"
            }
        
        try:
            num_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "total_parameters": num_params,
                "trainable_parameters": trainable_params,
                "device": str(self.device),
                "model_type": type(self.model).__name__,
                "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
                "status": "loaded"
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "total_parameters": 0,
                "trainable_parameters": 0,
                "device": str(self.device),
                "model_type": "Unknown",
                "vocab_size": 0,
                "status": "error",
                "error": str(e)
            }
    
    def set_model_path(self, path: str):
        """Set model path - useful for testing or different model locations"""
        self.model_path = path
        logger.info(f"Model path set to: {path}")
    
    def get_model(self):
        """Get the loaded model"""
        return self.model
    
    def get_tokenizer(self):
        """Get the loaded tokenizer"""
        return self.tokenizer