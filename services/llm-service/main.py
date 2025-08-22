# ===============================
# services/llm-service/main.py - FIXED VERSION
# ===============================
import asyncio
import time
import psutil
import uuid
import os
import sys
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CollectorRegistry,
)
from fastapi.responses import Response

# Setup Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "models"))
sys.path.insert(0, os.path.join(current_dir, "utils"))

import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project modules with error handling
try:
    from models.loader import ModelLoader

    logger.info("✅ ModelLoader imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import ModelLoader: {e}")
    ModelLoader = None

try:
    from models.text_generation import TextGenerator

    logger.info("✅ TextGenerator imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import TextGenerator: {e}")
    TextGenerator = None

try:
    from utils.gpu_manager import GPUManager

    logger.info("✅ GPUManager imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import GPUManager: {e}")
    GPUManager = None

try:
    from utils.gpu_client import SimpleGPUClient, GPUClientError

    GPU_COORDINATION_AVAILABLE = True
    logger.info("✅ GPU Client imported successfully")
except ImportError as e:
    GPU_COORDINATION_AVAILABLE = False
    logger.warning(f"⚠️ GPU Client not available: {e}")
    SimpleGPUClient = None

    class GPUClientError(Exception):
        pass


# Create a custom registry to avoid conflicts
registry = CollectorRegistry()

# Prometheus metrics with custom registry
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["method", "endpoint"],
    registry=registry,
)
REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds", "LLM request duration", registry=registry
)
ACTIVE_CONNECTIONS = Gauge(
    "llm_active_connections", "Active connections", registry=registry
)
MODEL_MEMORY_USAGE = Gauge(
    "llm_model_memory_usage_bytes", "Model memory usage", registry=registry
)
GPU_UTILIZATION = Gauge(
    "llm_gpu_utilization_percent", "GPU utilization", registry=registry
)
GENERATION_TOKENS = Counter(
    "llm_tokens_generated_total", "Total tokens generated", registry=registry
)
CHAT_SESSIONS = Gauge(
    "llm_chat_sessions_active", "Active chat sessions", registry=registry
)

# Global instances
model_loader: Optional[Any] = None
text_generator: Optional[Any] = None
gpu_manager: Optional[Any] = None
chat_sessions: Dict[str, Dict] = {}
app_start_time: float = 0
gpu_coordination_client: Optional[Any] = None


# Dummy classes for fallback
class DummyModelLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "gpt2-fa"
        self.model_path = os.getenv("MODEL_PATH", "/app/models/llm")
        self.device = "cpu"
        self.is_loaded = False

    async def initialize(self):
        logger.warning("⚠️ Using dummy ModelLoader - attempting to load model anyway")
        # Try to load model even with dummy loader
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from pathlib import Path

            model_dir = Path(self.model_path)
            logger.info(f"🔄 Attempting to load model from {model_dir}")

            if model_dir.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_dir), local_files_only=True, trust_remote_code=False
                )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_dir),
                    local_files_only=True,
                    torch_dtype=torch.float16
                    if torch.cuda.is_available()
                    else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                )

                if not torch.cuda.is_available():
                    self.model = self.model.to("cpu")

                self.model.eval()
                self.is_loaded = True
                logger.info(f"✅ Model loaded successfully on {self.device}")
            else:
                logger.error(f"❌ Model directory not found: {model_dir}")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")

    async def cleanup(self):
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False

    def get_model_info(self):
        if not self.is_loaded:
            return {
                "total_parameters": 0,
                "trainable_parameters": 0,
                "device": self.device,
                "model_type": "Unknown",
                "vocab_size": 0,
                "status": "not_loaded",
            }

        try:
            num_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            return {
                "total_parameters": num_params,
                "trainable_parameters": trainable_params,
                "device": str(self.device),
                "model_type": type(self.model).__name__,
                "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
                "status": "loaded",
            }
        except Exception as e:
            return {
                "total_parameters": 0,
                "trainable_parameters": 0,
                "device": str(self.device),
                "model_type": "Unknown",
                "vocab_size": 0,
                "status": "error",
                "error": str(e),
            }

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer


class DummyTextGenerator:
    def __init__(self, model_loader):
        self.model_loader = model_loader

    async def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ):
        if not self.model_loader.is_loaded:
            return {
                "generated_text": ["مدل بارگذاری نشده است"],
                "model_name": "dummy",
                "token_count": 0,
                "gpu_used": False,
            }

        try:
            import torch

            # Tokenize input
            inputs = self.model_loader.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Move to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model_loader.model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.model_loader.tokenizer.eos_token_id,
                    eos_token_id=self.model_loader.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # Decode outputs
            generated_texts = []
            total_tokens = 0

            for output in outputs:
                generated_tokens = output[inputs["input_ids"].shape[1] :]
                generated_text = self.model_loader.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                generated_texts.append(generated_text.strip())
                total_tokens += len(generated_tokens)

            return {
                "generated_text": generated_texts,
                "model_name": self.model_loader.model_name,
                "token_count": total_tokens,
                "gpu_used": device == "cuda",
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "generated_text": [f"خطا در تولید متن: {str(e)}"],
                "model_name": "error",
                "token_count": 0,
                "gpu_used": False,
            }


class DummyGPUManager:
    def __init__(self):
        try:
            import torch

            self.gpu_available = torch.cuda.is_available()
        except:
            self.gpu_available = False

    def get_gpu_info(self):
        try:
            import torch

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


class DummyGPUClient:
    def __init__(self, *args, **kwargs):
        pass

    async def is_coordinator_available(self):
        return False

    async def wait_for_gpu(self, *args, **kwargs):
        return type("Allocation", (), {"allocated": False})()

    async def release_gpu(self):
        return True

    def is_gpu_allocated(self):
        return False

    def get_current_gpu_id(self):
        return None


async def setup_gpu_coordination():
    """تنظیم GPU coordination"""
    global gpu_coordination_client

    if not GPU_COORDINATION_AVAILABLE or not SimpleGPUClient:
        logger.info("⚠️ GPU coordination not available - using dummy client")
        gpu_coordination_client = DummyGPUClient()
        return False

    try:
        coordinator_url = os.getenv(
            "GPU_COORDINATOR_URL", "https://gpu-coordinator:8080"
        )
        gpu_memory_gb = float(os.getenv("LLM_GPU_MEMORY_GB", "3.0"))

        gpu_coordination_client = SimpleGPUClient(
            coordinator_url=coordinator_url, service_name="llm-service"
        )

        if await gpu_coordination_client.is_coordinator_available():
            logger.info("✅ GPU Coordinator connected for LLM")
            allocation = await gpu_coordination_client.wait_for_gpu(
                memory_gb=gpu_memory_gb,
                priority="normal",
                max_wait_time=120,
            )

            if hasattr(allocation, "allocated") and allocation.allocated:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(allocation.gpu_id)
                logger.info(
                    f"✅ GPU {allocation.gpu_id} allocated for LLM via coordinator"
                )
                return True
            else:
                logger.warning("⚠️ GPU allocation failed - using direct GPU access")
                return False
        else:
            logger.warning("⚠️ GPU Coordinator not reachable")
            return False

    except Exception as e:
        logger.error(f"❌ GPU coordination setup failed: {e}")
        return False


async def cleanup_gpu_coordination():
    """پاکسازی GPU coordination"""
    global gpu_coordination_client

    if gpu_coordination_client and hasattr(gpu_coordination_client, "is_gpu_allocated"):
        try:
            if gpu_coordination_client.is_gpu_allocated():
                await gpu_coordination_client.release_gpu()
                logger.info("✅ GPU released from coordination")
        except Exception as e:
            logger.warning(f"⚠️ GPU release warning: {e}")


def get_gpu_availability() -> bool:
    """Safely check if GPU is available"""
    try:
        if gpu_manager is None:
            return False
        return getattr(gpu_manager, "gpu_available", False)
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {e}")
        return False


def get_gpu_info() -> Dict:
    """Safely get GPU information"""
    try:
        if gpu_manager is None:
            return {}
        if hasattr(gpu_manager, "get_gpu_info"):
            return gpu_manager.get_gpu_info() or {}
        return {}
    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifecycle management"""
    global model_loader, text_generator, gpu_manager, app_start_time

    logger.info("🚀 Starting LLM Service...")
    app_start_time = time.time()

    try:
        # Initialize GPU Manager
        gpu_coordination_success = await setup_gpu_coordination()
        if gpu_coordination_success:
            logger.info("🔄 GPU coordination successful")
        else:
            logger.info("🖥️ Using direct GPU access")

        # Initialize GPU Manager with fallback
        if GPUManager:
            try:
                gpu_manager = GPUManager()
                logger.info("✅ GPU Manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ GPU Manager initialization failed: {e}")
                gpu_manager = DummyGPUManager()
        else:
            gpu_manager = DummyGPUManager()

        # Initialize Model Loader with fallback
        if ModelLoader:
            model_loader = ModelLoader()
        else:
            model_loader = DummyModelLoader()

        try:
            await model_loader.initialize()
            logger.info("✅ Model Loader initialized")
        except Exception as e:
            logger.error(f"❌ Model Loader initialization failed: {e}")

        # Initialize Text Generator with fallback
        if TextGenerator:
            text_generator = TextGenerator(model_loader)
        else:
            text_generator = DummyTextGenerator(model_loader)
        logger.info("✅ Text Generator initialized")

        # Start monitoring task
        asyncio.create_task(monitor_system())

        logger.info("✅ LLM Service started successfully")

    except Exception as e:
        logger.error(f"❌ Failed to start LLM Service: {e}")
        # Ensure we have working instances
        if model_loader is None:
            model_loader = DummyModelLoader()
        if text_generator is None:
            text_generator = DummyTextGenerator(model_loader)
        if gpu_manager is None:
            gpu_manager = DummyGPUManager()

    yield

    # Shutdown
    logger.info("🛑 Shutting down LLM Service...")
    await cleanup_gpu_coordination()

    if model_loader:
        await model_loader.cleanup()


# تعریف اپلیکیشن FastAPI
app = FastAPI(
    title="LLM Service",
    description="Local Persian Language Model Service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class GenerationRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=2000, description="متن ورودی برای تولید"
    )  # Changed from 'prompt' to 'text'
    max_length: int = Field(default=100, ge=1, le=512, description="حداکثر طول خروجی")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="تنظیم خلاقیت")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="تنظیم تنوع")
    do_sample: bool = Field(default=True, description="فعال‌سازی نمونه‌گیری")
    num_return_sequences: int = Field(default=1, ge=1, le=3)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="پیام کاربر")
    conversation_id: Optional[str] = Field(default=None, description="شناسه مکالمه")
    system_prompt: Optional[str] = Field(default=None, description="دستورالعمل سیستم")


class GenerationResponse(BaseModel):
    generated_text: str
    model_name: str
    generation_time: float
    token_count: int
    gpu_used: bool
    model_info: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    model_name: str
    generation_time: float
    token_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]
    uptime: float


class ModelInfo(BaseModel):
    model_name: str
    model_path: str
    is_loaded: bool
    parameters: Dict[str, Any]
    gpu_available: bool
    memory_usage: Dict[str, float]


# متریک‌ها
async def update_metrics():
    """Update system metrics"""
    try:
        ACTIVE_CONNECTIONS.set(len(asyncio.all_tasks()))
        CHAT_SESSIONS.set(len(chat_sessions))

        # Memory usage
        memory = psutil.virtual_memory()
        MODEL_MEMORY_USAGE.set(memory.used)

        # GPU metrics if available
        if get_gpu_availability():
            gpu_info = get_gpu_info()
            if gpu_info:
                GPU_UTILIZATION.set(gpu_info.get("utilization", 0))
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")


def get_or_create_chat_session(conversation_id: Optional[str] = None) -> str:
    """Get existing or create new chat session"""
    if conversation_id and conversation_id in chat_sessions:
        return conversation_id

    new_id = str(uuid.uuid4())
    chat_sessions[new_id] = {
        "created_at": datetime.now(),
        "messages": [],
        "system_prompt": None,
    }
    return new_id


# مسیرها (Endpoints)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()

        memory = psutil.virtual_memory()
        return HealthResponse(
            status="healthy" if model_loader and model_loader.is_loaded else "loading",
            model_loaded=model_loader.is_loaded if model_loader else False,
            gpu_available=get_gpu_availability(),
            memory_usage={
                "used_gb": round(memory.used / 1024**3, 2),
                "available_gb": round(memory.available / 1024**3, 2),
                "percent": memory.percent,
            },
            uptime=time.time() - app_start_time,
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    try:
        REQUEST_COUNT.labels(method="GET", endpoint="/model/info").inc()

        if not model_loader:
            raise HTTPException(status_code=503, detail="Model loader not initialized")

        model_info = model_loader.get_model_info()
        memory = psutil.virtual_memory()

        return ModelInfo(
            model_name=model_loader.model_name,
            model_path=model_loader.model_path,
            is_loaded=model_loader.is_loaded,
            parameters=model_info,
            gpu_available=get_gpu_availability(),
            memory_usage={
                "used_gb": round(memory.used / 1024**3, 2),
                "available_gb": round(memory.available / 1024**3, 2),
                "percent": memory.percent,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate text using loaded model"""
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate").inc()

        if not text_generator or not model_loader.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start_time = time.time()

        with REQUEST_DURATION.time():
            result = await text_generator.generate(
                prompt=request.text,  # Use 'text' field
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                num_return_sequences=request.num_return_sequences,
            )

        generation_time = time.time() - start_time

        # Update metrics
        GENERATION_TOKENS.inc(result["token_count"])
        background_tasks.add_task(update_metrics)

        # Return single text (first result) to match API Gateway expectation
        generated_text = (
            result["generated_text"][0]
            if isinstance(result["generated_text"], list)
            else result["generated_text"]
        )

        return GenerationResponse(
            generated_text=generated_text,
            model_name=result["model_name"],
            generation_time=generation_time,
            token_count=result["token_count"],
            gpu_used=result.get("gpu_used", False),
            model_info={
                "model_path": model_loader.model_path,
                "gpu_available": get_gpu_availability(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        await update_metrics()
        return Response(generate_latest(registry), media_type="text/plain")
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate metrics: {str(e)}"
        )


# Background monitoring
async def monitor_system():
    """Background task for system monitoring"""
    while True:
        try:
            await update_metrics()

            # Clean old chat sessions (older than 1 hour)
            current_time = datetime.now()
            old_sessions = [
                session_id
                for session_id, session in chat_sessions.items()
                if (current_time - session["created_at"]).seconds > 3600
            ]

            for session_id in old_sessions:
                del chat_sessions[session_id]
                logger.info(f"Cleaned old chat session: {session_id}")

            await asyncio.sleep(10)  # Update every 10 seconds

        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        workers=1,
    )
