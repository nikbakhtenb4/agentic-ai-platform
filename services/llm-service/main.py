# ===============================
# services/llm-service/main.py - ENHANCED FIXED VERSION
# ===============================
import asyncio
from pathlib import Path
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

logger.info(f"***************************: {current_dir}")
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


# Enhanced Dummy classes for better fallback
class DummyModelLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "gpt2-fa"
        self.model_path = os.getenv("MODEL_PATH", "/app/models/llm")
        self.device = "cpu"
        self.is_loaded = False
        logger.info(f"🔄 DummyModelLoader initialized with path: {self.model_path}")

    async def initialize(self):
        """Initialize and load the model with enhanced path checking"""
        logger.warning("⚠️ Using dummy ModelLoader - attempting to load model anyway")
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from pathlib import Path

            # Try multiple model paths with better logging
            possible_paths = [
                Path(self.model_path),
                Path(self.model_path) / self.model_name,
                Path("/app/models") / self.model_name,
                Path("/app/models/llm"),
                Path("./models") / self.model_name,
                Path("/app/models/llm/gpt2-fa"),  # Specific path from your error
            ]

            logger.info(
                f"🔍 Searching for model in paths: {[str(p) for p in possible_paths]}"
            )

            model_dir = None
            for path in possible_paths:
                logger.info(f"📁 Checking path: {path}")
                if path.exists():
                    logger.info(f"✅ Path exists: {path}")
                    if self._is_model_directory(path):
                        model_dir = path
                        logger.info(f"🎯 Found model directory: {model_dir}")
                        break
                    else:
                        logger.warning(f"⚠️ Path exists but missing model files: {path}")
                        # List contents for debugging
                        try:
                            contents = list(path.iterdir())
                            logger.info(
                                f"📋 Directory contents: {[f.name for f in contents]}"
                            )
                        except:
                            pass
                else:
                    logger.info(f"❌ Path does not exist: {path}")

            if not model_dir:
                logger.error(
                    f"❌ Model directory not found in any of: {[str(p) for p in possible_paths]}"
                )
                # Create a minimal fake model for testing
                await self._create_dummy_model()
                return

            logger.info(f"🔄 Loading model from {model_dir}")

            # Load tokenizer with better error handling
            try:
                logger.info("📝 Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_dir), local_files_only=True, trust_remote_code=False
                )
                logger.info("✅ Tokenizer loaded successfully")

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info("🔧 Set pad_token to eos_token")

            except Exception as e:
                logger.error(f"❌ Failed to load tokenizer: {e}")
                raise

            # Load model with better error handling
            try:
                logger.info("🤖 Loading model...")
                device_map = "auto" if torch.cuda.is_available() else None
                torch_dtype = (
                    torch.float16 if torch.cuda.is_available() else torch.float32
                )

                logger.info(f"🎯 Using device_map: {device_map}, dtype: {torch_dtype}")

                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_dir),
                    local_files_only=True,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                )
                logger.info("✅ Model loaded successfully")

                if not torch.cuda.is_available():
                    self.model = self.model.to("cpu")
                    logger.info("🖥️ Model moved to CPU")

                self.model.eval()
                self.is_loaded = True
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"✅ Model initialized successfully on {self.device}")

            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                raise

        except Exception as e:
            logger.error(f"❌ Failed to load real model: {e}")
            await self._create_dummy_model()

    def _is_model_directory(self, path: Path) -> bool:
        """Check if path contains model files with better validation"""
        required_files = ["config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors", "model.bin"]
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "tokenizer.model",
        ]

        logger.info(f"🔍 Validating model directory: {path}")

        has_config = (path / "config.json").exists()
        logger.info(f"📝 Has config.json: {has_config}")

        has_model = any((path / f).exists() for f in model_files)
        found_model_files = [(path / f).exists() for f in model_files]
        logger.info(
            f"🤖 Model files check: {dict(zip(model_files, found_model_files))}"
        )

        has_tokenizer = any((path / f).exists() for f in tokenizer_files)
        found_tokenizer_files = [(path / f).exists() for f in tokenizer_files]
        logger.info(
            f"📝 Tokenizer files check: {dict(zip(tokenizer_files, found_tokenizer_files))}"
        )

        is_valid = has_config and (has_model or has_tokenizer)
        logger.info(f"✅ Directory valid: {is_valid}")

        return is_valid

    async def _create_dummy_model(self):
        """Create a minimal dummy model for testing with better fallback"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info("🔄 Creating dummy model for testing...")

            # Try multiple fallback models
            fallback_models = ["gpt2", "distilgpt2"]

            for model_name in fallback_models:
                try:
                    logger.info(f"📥 Attempting to load {model_name} as fallback...")

                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                    )

                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                    self.model.eval()
                    self.is_loaded = True
                    self.device = "cpu"
                    self.model_name = f"{model_name}-fallback"
                    logger.info(f"✅ Dummy model ({model_name}) loaded as fallback")
                    return

                except Exception as e:
                    logger.error(f"❌ Failed to load {model_name}: {e}")
                    continue

            # If all fallbacks fail
            logger.error("❌ All fallback models failed")
            self.tokenizer = None
            self.model = None
            self.is_loaded = False

        except Exception as e:
            logger.error(f"❌ Dummy model creation failed: {e}")
            self.is_loaded = False

    async def cleanup(self):
        import torch  # Add this line

        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        self.is_loaded = False
        logger.info("🧹 Model cleanup completed")

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
            num_params = (
                sum(p.numel() for p in self.model.parameters()) if self.model else 0
            )
            trainable_params = (
                sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                if self.model
                else 0
            )

            return {
                "total_parameters": num_params,
                "trainable_parameters": trainable_params,
                "device": str(self.device),
                "model_type": type(self.model).__name__ if self.model else "Unknown",
                "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
                "status": "loaded" if self.is_loaded else "not_loaded",
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
            logger.warning("⚠️ Model not loaded, returning error response")
            return {
                "generated_text": [
                    "Model is not loaded. Please check model initialization."
                ],
                "model_name": "not-loaded",
                "token_count": 0,
                "gpu_used": False,
            }

        try:
            import torch

            logger.info(f"🔄 Generating text for prompt: {prompt[:50]}...")

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

            logger.info(f"📊 Input tokens: {inputs['input_ids'].shape[1]}")

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

            logger.info(f"✅ Generated {total_tokens} tokens")

            return {
                "generated_text": generated_texts,
                "model_name": self.model_loader.model_name,
                "token_count": total_tokens,
                "gpu_used": device == "cuda",
            }

        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return {
                "generated_text": [f"Generation error: {str(e)}"],
                "model_name": "error",
                "token_count": 0,
                "gpu_used": False,
            }


class DummyGPUManager:
    def __init__(self):
        try:
            import torch

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
        self.coordinator_url = kwargs.get(
            "coordinator_url", "http://gpu-coordinator:8080"
        )
        self.service_name = kwargs.get("service_name", "llm-service")
        self.current_task_id = None
        self.current_gpu_id = None

    async def is_coordinator_available(self):
        return False

    async def wait_for_gpu(self, *args, **kwargs):
        return type("Allocation", (), {"allocated": False})()

    async def release_gpu(self):
        return True

    def is_gpu_allocated(self):
        return self.current_task_id is not None

    def get_current_gpu_id(self):
        return self.current_gpu_id


async def setup_gpu_coordination():
    """Setup GPU coordination with better error handling"""
    global gpu_coordination_client

    if not GPU_COORDINATION_AVAILABLE or not SimpleGPUClient:
        logger.info("⚠️ GPU coordination not available - using dummy client")
        gpu_coordination_client = DummyGPUClient()
        return False

    try:
        coordinator_url = os.getenv(
            "GPU_COORDINATOR_URL", "http://gpu-coordinator:8080"
        )
        gpu_memory_gb = float(os.getenv("LLM_GPU_MEMORY_GB", "3.0"))

        logger.info(f"🔗 Connecting to GPU coordinator: {coordinator_url}")
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
    """Clean up GPU coordination"""
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
            import torch

            return torch.cuda.is_available()
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
    """App lifecycle management with enhanced initialization"""
    global model_loader, text_generator, gpu_manager, app_start_time

    logger.info("🚀 Starting LLM Service...")
    app_start_time = time.time()

    try:
        # Initialize GPU Manager first
        if GPUManager:
            try:
                gpu_manager = GPUManager()
                logger.info("✅ GPU Manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ GPU Manager initialization failed: {e}")
                gpu_manager = DummyGPUManager()
        else:
            gpu_manager = DummyGPUManager()

        # Initialize GPU coordination
        gpu_coordination_success = await setup_gpu_coordination()
        if gpu_coordination_success:
            logger.info("🔥 GPU coordination successful")
        else:
            logger.info("🖥️ Using direct GPU access")

        # Initialize Model Loader with better error handling
        logger.info("📚 Initializing Model Loader...")
        if ModelLoader:
            try:
                model_loader = ModelLoader()
                logger.info("🔄 Initializing real ModelLoader...")
                await model_loader.initialize()
                logger.info("✅ Real Model Loader initialized")
            except Exception as e:
                logger.error(f"❌ Real Model Loader failed: {e}")
                logger.info("🔄 Falling back to DummyModelLoader...")
                model_loader = DummyModelLoader()
                await model_loader.initialize()
        else:
            logger.info("🔄 Using DummyModelLoader...")
            model_loader = DummyModelLoader()
            await model_loader.initialize()

        # Initialize Text Generator
        logger.info("📝 Initializing Text Generator...")
        if TextGenerator:
            try:
                text_generator = TextGenerator(model_loader)
                logger.info("✅ Real Text Generator initialized")
            except Exception as e:
                logger.error(f"❌ Real Text Generator failed: {e}")
                text_generator = DummyTextGenerator(model_loader)
        else:
            text_generator = DummyTextGenerator(model_loader)

        logger.info("✅ Text Generator initialized")

        # Start monitoring task
        asyncio.create_task(monitor_system())

        logger.info("✅ LLM Service started successfully")
        logger.info(f"📊 Model loaded: {model_loader.is_loaded}")
        logger.info(f"🎮 GPU available: {get_gpu_availability()}")

    except Exception as e:
        logger.error(f"❌ Failed to start LLM Service: {e}")
        # Ensure we have working instances
        if model_loader is None:
            model_loader = DummyModelLoader()
            await model_loader.initialize()
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


# FastAPI app definition
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
        ..., min_length=1, max_length=2000, description="Input text for generation"
    )
    max_length: int = Field(
        default=100, ge=1, le=512, description="Maximum output length"
    )
    temperature: float = Field(
        default=0.7, ge=0.1, le=2.0, description="Creativity control"
    )
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Diversity control")
    do_sample: bool = Field(default=True, description="Enable sampling")
    num_return_sequences: int = Field(default=1, ge=1, le=3)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    system_prompt: Optional[str] = Field(
        default=None, description="System instructions"
    )


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


class GPUStatusResponse(BaseModel):
    coordination_available: bool
    gpu_allocated: bool
    gpu_id: Optional[int]
    coordinator_url: str
    service_name: str


# Metrics update function
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


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()

        memory = psutil.virtual_memory()
        status = "healthy" if (model_loader and model_loader.is_loaded) else "loading"

        return HealthResponse(
            status=status,
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


@app.get("/gpu/status", response_model=GPUStatusResponse)
async def get_gpu_status():
    """Get GPU coordination status - FIXED ENDPOINT"""
    try:
        REQUEST_COUNT.labels(method="GET", endpoint="/gpu/status").inc()

        gpu_allocated = False
        gpu_id = None

        if gpu_coordination_client:
            gpu_allocated = gpu_coordination_client.is_gpu_allocated()
            gpu_id = (
                gpu_coordination_client.get_current_gpu_id()
                if hasattr(gpu_coordination_client, "get_current_gpu_id")
                else None
            )

        return GPUStatusResponse(
            coordination_available=GPU_COORDINATION_AVAILABLE,
            gpu_allocated=gpu_allocated,
            gpu_id=gpu_id,
            coordinator_url=gpu_coordination_client.coordinator_url
            if gpu_coordination_client
            else "N/A",
            service_name="llm-service",
        )

    except Exception as e:
        logger.error(f"GPU status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    """Generate text using loaded model - ENHANCED VERSION"""
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate").inc()

        if not text_generator:
            raise HTTPException(
                status_code=503, detail="Text generator not initialized"
            )

        if not model_loader or not model_loader.is_loaded:
            # Return helpful error message
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Model not loaded",
                    "message": "The language model is not loaded. This could be due to missing model files or initialization failure.",
                    "suggestions": [
                        "Check if model files exist in the specified MODEL_PATH",
                        "Verify model directory contains required files (config.json, model files, tokenizer files)",
                        "Check service logs for detailed error messages",
                        "Ensure sufficient memory/disk space",
                    ],
                },
            )

        start_time = time.time()

        logger.info(f"🔄 Processing generation request: {request.text[:50]}...")

        with REQUEST_DURATION.time():
            result = await text_generator.generate(
                prompt=request.text,
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

        logger.info(
            f"✅ Generation completed in {generation_time:.2f}s, {result['token_count']} tokens"
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
                "model_loaded": model_loader.is_loaded,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Chat endpoint with conversation context"""
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/chat").inc()

        if not text_generator:
            raise HTTPException(
                status_code=503, detail="Text generator not initialized"
            )

        if not model_loader or not model_loader.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Get or create conversation session
        conversation_id = get_or_create_chat_session(request.conversation_id)
        session = chat_sessions[conversation_id]

        # Build context-aware prompt
        if request.system_prompt:
            session["system_prompt"] = request.system_prompt

        # Create conversational prompt
        context_parts = []
        if session["system_prompt"]:
            context_parts.append(f"System: {session['system_prompt']}")

        # Add recent conversation history (last 3 exchanges)
        recent_messages = session["messages"][-6:]  # Last 3 exchanges (6 messages)
        for msg in recent_messages:
            context_parts.append(f"User: {msg['user']}")
            context_parts.append(f"Assistant: {msg['assistant']}")

        # Add current message
        context_parts.append(f"User: {request.message}")
        context_parts.append("Assistant:")

        full_prompt = "\n".join(context_parts)

        start_time = time.time()

        # Generate response
        result = await text_generator.generate(
            prompt=full_prompt,
            max_length=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        generation_time = time.time() - start_time

        # Extract response text
        response_text = (
            result["generated_text"][0]
            if isinstance(result["generated_text"], list)
            else result["generated_text"]
        )

        # Clean up response (remove any system prompts that might have leaked through)
        response_text = response_text.strip()
        if response_text.startswith(("User:", "Assistant:", "System:")):
            lines = response_text.split("\n")
            response_text = lines[0].replace("Assistant:", "").strip()

        # Update conversation history
        session["messages"].append(
            {
                "user": request.message,
                "assistant": response_text,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only last 10 exchanges
        if len(session["messages"]) > 10:
            session["messages"] = session["messages"][-10:]

        # Update metrics
        GENERATION_TOKENS.inc(result["token_count"])
        background_tasks.add_task(update_metrics)

        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            model_name=result["model_name"],
            generation_time=generation_time,
            token_count=result["token_count"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    try:
        if conversation_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Conversation not found")

        session = chat_sessions[conversation_id]
        return {
            "conversation_id": conversation_id,
            "created_at": session["created_at"].isoformat(),
            "system_prompt": session.get("system_prompt"),
            "messages": session["messages"],
            "message_count": len(session["messages"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        if conversation_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Conversation not found")

        del chat_sessions[conversation_id]
        return {
            "message": "Conversation deleted successfully",
            "conversation_id": conversation_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations")
async def list_conversations():
    """List all active conversations"""
    try:
        conversations = []
        for conv_id, session in chat_sessions.items():
            conversations.append(
                {
                    "conversation_id": conv_id,
                    "created_at": session["created_at"].isoformat(),
                    "message_count": len(session["messages"]),
                    "last_activity": session["messages"][-1]["timestamp"]
                    if session["messages"]
                    else None,
                }
            )

        return {"conversations": conversations, "total_count": len(conversations)}
    except Exception as e:
        logger.error(f"List conversations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reload")
async def reload_model():
    """Reload the model"""
    try:
        global model_loader, text_generator

        if not model_loader:
            raise HTTPException(status_code=503, detail="Model loader not available")

        logger.info("🔄 Reloading model...")
        await model_loader.cleanup()
        await model_loader.initialize()

        # Reinitialize text generator
        if TextGenerator:
            try:
                text_generator = TextGenerator(model_loader)
            except:
                text_generator = DummyTextGenerator(model_loader)
        else:
            text_generator = DummyTextGenerator(model_loader)

        return {
            "message": "Model reloaded successfully",
            "model_loaded": model_loader.is_loaded,
            "model_name": model_loader.model_name,
        }
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/paths")
async def debug_model_paths():
    """Debug endpoint to check model paths and files"""
    try:
        from pathlib import Path

        model_path = os.getenv("MODEL_PATH", "/app/models/llm")
        model_name = "gpt2-fa"

        paths_info = {}

        # Check various possible paths
        possible_paths = [
            Path(model_path),
            Path(model_path) / model_name,
            Path("/app/models") / model_name,
            Path("/app/models/llm"),
            Path("./models") / model_name,
            Path("/app/models/llm/gpt2-fa"),
        ]

        for path in possible_paths:
            path_info = {
                "exists": path.exists(),
                "is_directory": path.is_dir() if path.exists() else False,
                "contents": [],
            }

            if path.exists() and path.is_dir():
                try:
                    path_info["contents"] = [f.name for f in path.iterdir()]
                except:
                    path_info["contents"] = ["Error reading directory"]

            paths_info[str(path)] = path_info

        return {
            "model_path_env": model_path,
            "model_name": model_name,
            "current_working_directory": os.getcwd(),
            "paths_checked": paths_info,
            "environment_variables": {
                "MODEL_PATH": os.getenv("MODEL_PATH"),
                "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
                "LLM_GPU_MEMORY_GB": os.getenv("LLM_GPU_MEMORY_GB"),
            },
        }
    except Exception as e:
        logger.error(f"Debug paths error: {e}")
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
                logger.info(f"🧹 Cleaned old chat session: {session_id}")

            await asyncio.sleep(30)  # Update every 30 seconds

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
