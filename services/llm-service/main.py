# ===============================
# services/llm-service/main.py
# ===============================
import asyncio
import time
import psutil
import uuid
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry
from fastapi.responses import Response

from models.loader import ModelLoader
from models.text_generation import TextGenerator
from utils.gpu_manager import GPUManager
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a custom registry to avoid conflicts
registry = CollectorRegistry()

# Prometheus metrics with custom registry
REQUEST_COUNT = Counter('llm_requests_total', 'Total LLM requests', ['method', 'endpoint'], registry=registry)
REQUEST_DURATION = Histogram('llm_request_duration_seconds', 'LLM request duration', registry=registry)
ACTIVE_CONNECTIONS = Gauge('llm_active_connections', 'Active connections', registry=registry)
MODEL_MEMORY_USAGE = Gauge('llm_model_memory_usage_bytes', 'Model memory usage', registry=registry)
GPU_UTILIZATION = Gauge('llm_gpu_utilization_percent', 'GPU utilization', registry=registry)
GENERATION_TOKENS = Counter('llm_tokens_generated_total', 'Total tokens generated', registry=registry)
CHAT_SESSIONS = Gauge('llm_chat_sessions_active', 'Active chat sessions', registry=registry)

# Global instances
model_loader: Optional[ModelLoader] = None
text_generator: Optional[TextGenerator] = None
gpu_manager: Optional[GPUManager] = None
chat_sessions: Dict[str, Dict] = {}  # Store chat conversations
app_start_time: float = 0

# Helper function to safely get GPU availability
def get_gpu_availability() -> bool:
    """Safely check if GPU is available"""
    try:
        if gpu_manager is None:
            return False
        return getattr(gpu_manager, 'gpu_available', False)
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {e}")
        return False

# Helper function to safely get GPU info
def get_gpu_info() -> Dict:
    """Safely get GPU information"""
    try:
        if gpu_manager is None:
            return {}
        if hasattr(gpu_manager, 'get_gpu_info'):
            return gpu_manager.get_gpu_info() or {}
        return {}
    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")
        return {}

# مدیریت چرخه حیات برنامه (lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifecycle management"""
    global model_loader, text_generator, gpu_manager, app_start_time
    
    logger.info("🚀 Starting LLM Service...")
    app_start_time = time.time()
    
    try:
        # Initialize GPU Manager
        try:
            gpu_manager = GPUManager()
            logger.info("✅ GPU Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ GPU Manager initialization failed: {e}")
            gpu_manager = None
        
        # Initialize Model Loader
        model_loader = ModelLoader()
        await model_loader.initialize()
        
        # Initialize Text Generator
        text_generator = TextGenerator(model_loader)
        
        # Start monitoring task
        asyncio.create_task(monitor_system())
        
        logger.info("✅ LLM Service started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start LLM Service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down LLM Service...")
    if model_loader:
        await model_loader.cleanup()

# تعریف اپلیکیشن FastAPI
app = FastAPI(
    title="LLM Service",
    description="Local Persian Language Model Service",
    version="1.0.0",
    lifespan=lifespan
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
    prompt: str = Field(..., min_length=1, max_length=2000, description="متن ورودی برای تولید")
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
    model_info: Dict[str, Any] = {}  # Changed from 'any' to 'Any'


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    model_name: str
    generation_time: float
    token_count: int

class BatchGenerationRequest(BaseModel):
    requests: List[GenerationRequest] = Field(..., max_items=10)

class BatchGenerationResponse(BaseModel):
    results: List[GenerationResponse]
    total_time: float
    successful_count: int
    failed_count: int

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
    parameters: Dict[str, Any]  # Changed from 'any' to 'Any'
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
                GPU_UTILIZATION.set(gpu_info.get('utilization', 0))
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")

# مدیریت جلسات گفت‌وگو
def get_or_create_chat_session(conversation_id: Optional[str] = None) -> str:
    """Get existing or create new chat session"""
    if conversation_id and conversation_id in chat_sessions:
        return conversation_id
    
    # Create new session
    new_id = str(uuid.uuid4())
    chat_sessions[new_id] = {
        'created_at': datetime.now(),
        'messages': [],
        'system_prompt': None
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
                "percent": memory.percent
            },
            uptime=time.time() - app_start_time
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate text using loaded model"""
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate").inc()
        
        if not text_generator or not model_loader.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        
        with REQUEST_DURATION.time():
            result = await text_generator.generate(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                num_return_sequences=request.num_return_sequences
            )
        
        generation_time = time.time() - start_time
        
        # Update metrics
        GENERATION_TOKENS.inc(result['token_count'])
        background_tasks.add_task(update_metrics)
        
        # Return single text (first result) to match API Gateway expectation
        generated_text = result['generated_text'][0] if isinstance(result['generated_text'], list) else result['generated_text']
        
        return GenerationResponse(
            generated_text=generated_text,
            model_name=result['model_name'],
            generation_time=generation_time,
            token_count=result['token_count'],
            gpu_used=result.get('gpu_used', False),
            model_info={
                "model_path": model_loader.model_path,
                "gpu_available": get_gpu_availability()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """Chat with the model"""
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/chat").inc()
        
        if not text_generator or not model_loader.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        
        # Get or create chat session
        conversation_id = get_or_create_chat_session(request.conversation_id)
        session = chat_sessions[conversation_id]
        
        # Set system prompt if provided
        if request.system_prompt:
            session['system_prompt'] = request.system_prompt
        
        # Build conversation context
        context = ""
        if session['system_prompt']:
            context += f"System: {session['system_prompt']}\n\n"
        
        # Add previous messages (keep last 5 for context)
        for msg in session['messages'][-5:]:
            context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
        
        # Add current message
        context += f"User: {request.message}\nAssistant:"
        
        with REQUEST_DURATION.time():
            result = await text_generator.generate(
                prompt=context,
                max_length=200,  # Reasonable length for chat
                temperature=0.8,  # Slightly more creative for chat
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )
        
        generation_time = time.time() - start_time
        
        # Extract response
        response_text = result['generated_text'][0] if isinstance(result['generated_text'], list) else result['generated_text']
        
        # Clean up response (remove context)
        if "Assistant:" in response_text:
            response_text = response_text.split("Assistant:")[-1].strip()
        
        # Store in session
        session['messages'].append({
            'user': request.message,
            'assistant': response_text,
            'timestamp': datetime.now()
        })
        
        # Update metrics
        GENERATION_TOKENS.inc(result['token_count'])
        background_tasks.add_task(update_metrics)
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            model_name=result['model_name'],
            generation_time=generation_time,
            token_count=result['token_count']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch/generate", response_model=BatchGenerationResponse)
async def batch_generate(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Batch generation endpoint"""
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/batch/generate").inc()
        
        if not text_generator or not model_loader.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        results = []
        successful_count = 0
        failed_count = 0
        
        for gen_request in request.requests:
            try:
                result = await text_generator.generate(
                    prompt=gen_request.prompt,
                    max_length=gen_request.max_length,
                    temperature=gen_request.temperature,
                    top_p=gen_request.top_p,
                    do_sample=gen_request.do_sample,
                    num_return_sequences=gen_request.num_return_sequences
                )
                
                generated_text = result['generated_text'][0] if isinstance(result['generated_text'], list) else result['generated_text']
                
                results.append(GenerationResponse(
                    generated_text=generated_text,
                    model_name=result['model_name'],
                    generation_time=0,  # Individual timing not tracked in batch
                    token_count=result['token_count'],
                    gpu_used=result.get('gpu_used', False),
                    model_info={}
                ))
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Batch generation error for prompt: {e}")
                failed_count += 1
        
        total_time = time.time() - start_time
        background_tasks.add_task(update_metrics)
        
        return BatchGenerationResponse(
            results=results,
            total_time=total_time,
            successful_count=successful_count,
            failed_count=failed_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model information"""
    try:
        REQUEST_COUNT.labels(method="GET", endpoint="/model/info").inc()
        
        if not model_loader:
            raise HTTPException(status_code=503, detail="Model loader not initialized")
        
        memory = psutil.virtual_memory()
        
        return ModelInfo(
            model_name=model_loader.model_name,
            model_path=model_loader.model_path,
            is_loaded=model_loader.is_loaded,
            parameters=model_loader.get_model_info(),
            gpu_available=get_gpu_availability(),
            memory_usage={
                "used_gb": round(memory.used / 1024**3, 2),
                "available_gb": round(memory.available / 1024**3, 2),
                "percent": memory.percent
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/model/reload")
async def reload_model():
    """Reload the model"""
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/model/reload").inc()
        
        if not model_loader:
            raise HTTPException(status_code=503, detail="Model loader not initialized")
        
        await model_loader.reload()
        return {"status": "success", "message": "Model reloaded successfully"}
        
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")

@app.delete("/chat/{conversation_id}")
async def delete_chat_session(conversation_id: str):
    """Delete a chat session"""
    try:
        REQUEST_COUNT.labels(method="DELETE", endpoint="/chat").inc()
        
        if conversation_id in chat_sessions:
            del chat_sessions[conversation_id]
            return {"status": "success", "message": f"Chat session {conversation_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Chat session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete chat session error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chat session: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        await update_metrics()
        return Response(generate_latest(registry), media_type="text/plain")
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate metrics: {str(e)}")

@app.get("/health/metrics")
async def health_metrics():
    """Health metrics for Prometheus monitoring"""
    try:
        await update_metrics()
        
        memory = psutil.virtual_memory()
        
        return {
            "model_loaded": 1 if model_loader and model_loader.is_loaded else 0,
            "gpu_available": 1 if get_gpu_availability() else 0,
            "memory_usage_percent": memory.percent,
            "active_chat_sessions": len(chat_sessions),
            "uptime_seconds": time.time() - app_start_time
        }
        
    except Exception as e:
        logger.error(f"Health metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health metrics: {str(e)}")

# Background monitoring
async def monitor_system():
    """Background task for system monitoring"""
    while True:
        try:
            await update_metrics()
            
            # Clean old chat sessions (older than 1 hour)
            current_time = datetime.now()
            old_sessions = [
                session_id for session_id, session in chat_sessions.items()
                if (current_time - session['created_at']).seconds > 3600
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
        workers=1  # Single worker برای memory efficiency
    )