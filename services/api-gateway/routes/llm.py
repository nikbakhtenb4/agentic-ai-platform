# services/api-gateway/routes/llm.py

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import httpx
import asyncio
import logging
import time
from datetime import datetime

# تنظیمات لاگینگ
logger = logging.getLogger(__name__)

# تنظیمات LLM Service
LLM_SERVICE_URL = "http://llm-service:8002"
REQUEST_TIMEOUT = 120  # 2 دقیقه timeout

# Router برای LLM endpoints
router = APIRouter(prefix="/llm", tags=["LLM"])

# Pydantic Models
class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="متن ورودی برای تولید")
    max_length: Optional[int] = Field(default=100, ge=1, le=512, description="حداکثر طول خروجی")
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=2.0, description="تنظیم خلاقیت")
    top_p: Optional[float] = Field(default=0.9, ge=0.1, le=1.0, description="تنظیم تنوع")
    do_sample: Optional[bool] = Field(default=True, description="فعال‌سازی نمونه‌گیری")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="پیام کاربر")
    conversation_id: Optional[str] = Field(default=None, description="شناسه مکالمه")
    system_prompt: Optional[str] = Field(default=None, description="دستورالعمل سیستم")

class TextGenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    processing_time: float
    model_info: Dict[str, Any]
    timestamp: datetime

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    processing_time: float
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    memory_usage: Optional[Dict[str, Any]]
    uptime: float

# HTTP Client برای ارتباط با LLM Service
async def get_llm_client():
    return httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
    )

# Dependency برای بررسی دسترسی LLM Service
async def check_llm_service():
    try:
        async with await get_llm_client() as client:
            response = await client.get(f"{LLM_SERVICE_URL}/health", timeout=10)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=503, 
                    detail="LLM Service در دسترس نیست"
                )
            return response.json()
    except httpx.RequestError:
        raise HTTPException(
            status_code=503, 
            detail="خطا در اتصال به LLM Service"
        )

# Health Check Endpoint
@router.get("/health", response_model=HealthResponse)
async def llm_health():
    """بررسی سلامت سرویس LLM"""
    try:
        async with await get_llm_client() as client:
            start_time = time.time()
            response = await client.get(f"{LLM_SERVICE_URL}/health")
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return HealthResponse(
                    status="healthy",
                    model_loaded=data.get("model_loaded", False),
                    memory_usage=data.get("memory_usage"),
                    uptime=data.get("uptime", 0)
                )
            else:
                raise HTTPException(status_code=503, detail="LLM Service غیرفعال")
                
    except httpx.RequestError as e:
        logger.error(f"خطا در health check: {e}")
        raise HTTPException(status_code=503, detail="خطا در اتصال به LLM Service")

# Text Generation Endpoint
@router.post("/generate", response_model=TextGenerationResponse)
async def generate_text(
    request: TextGenerationRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(check_llm_service)
):
    """تولید متن با استفاده از مدل زبانی"""
    
    start_time = time.time()
    
    try:
        async with await get_llm_client() as client:
            # ارسال درخواست به LLM Service
            llm_response = await client.post(
                f"{LLM_SERVICE_URL}/generate",
                json=request.dict(),
                headers={"Content-Type": "application/json"}
            )
            
            if llm_response.status_code != 200:
                error_detail = llm_response.json().get("detail", "خطای نامشخص")
                raise HTTPException(
                    status_code=llm_response.status_code,
                    detail=f"خطا در تولید متن: {error_detail}"
                )
            
            result = llm_response.json()
            processing_time = time.time() - start_time
            
            # لاگ کردن درخواست
            background_tasks.add_task(
                log_llm_request,
                "generate",
                request.prompt[:50] + "..." if len(request.prompt) > 50 else request.prompt,
                processing_time
            )
            
            return TextGenerationResponse(
                generated_text=result["generated_text"],
                prompt=request.prompt,
                processing_time=processing_time,
                model_info=result.get("model_info", {}),
                timestamp=datetime.now()
            )
            
    except httpx.RequestError as e:
        logger.error(f"خطا در ارتباط با LLM Service: {e}")
        raise HTTPException(
            status_code=503,
            detail="خطا در ارتباط با سرویس تولید متن"
        )

# Chat Endpoint
@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(check_llm_service)
):
    """گفتگو با مدل زبانی"""
    
    start_time = time.time()
    
    try:
        async with await get_llm_client() as client:
            # ارسال درخواست به LLM Service
            llm_response = await client.post(
                f"{LLM_SERVICE_URL}/chat",
                json=request.dict(),
                headers={"Content-Type": "application/json"}
            )
            
            if llm_response.status_code != 200:
                error_detail = llm_response.json().get("detail", "خطای نامشخص")
                raise HTTPException(
                    status_code=llm_response.status_code,
                    detail=f"خطا در گفتگو: {error_detail}"
                )
            
            result = llm_response.json()
            processing_time = time.time() - start_time
            
            # لاگ کردن درخواست
            background_tasks.add_task(
                log_llm_request,
                "chat",
                request.message[:50] + "..." if len(request.message) > 50 else request.message,
                processing_time
            )
            
            return ChatResponse(
                response=result["response"],
                conversation_id=result["conversation_id"],
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
    except httpx.RequestError as e:
        logger.error(f"خطا در ارتباط با LLM Service: {e}")
        raise HTTPException(
            status_code=503,
            detail="خطا در ارتباط با سرویس گفتگو"
        )

# Model Info Endpoint
@router.get("/model/info")
async def get_model_info(_: dict = Depends(check_llm_service)):
    """دریافت اطلاعات مدل"""
    
    try:
        async with await get_llm_client() as client:
            response = await client.get(f"{LLM_SERVICE_URL}/model/info")
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="خطا در دریافت اطلاعات مدل"
                )
            
            return response.json()
            
    except httpx.RequestError as e:
        logger.error(f"خطا در دریافت اطلاعات مدل: {e}")
        raise HTTPException(
            status_code=503,
            detail="خطا در ارتباط با سرویس مدل"
        )

# Background Task برای لاگینگ
async def log_llm_request(request_type: str, prompt: str, processing_time: float):
    """لاگ کردن درخواست‌های LLM"""
    logger.info(
        f"LLM Request - Type: {request_type}, "
        f"Prompt: {prompt}, "
        f"Processing Time: {processing_time:.2f}s"
    )

# Batch Processing Endpoint (برای آینده)
@router.post("/batch/generate")
async def batch_generate(
    requests: List[TextGenerationRequest],
    _: dict = Depends(check_llm_service)
):
    """پردازش دسته‌ای درخواست‌ها"""
    
    if len(requests) > 10:
        raise HTTPException(
            status_code=400,
            detail="حداکثر 10 درخواست در هر دسته مجاز است"
        )
    
    try:
        async with await get_llm_client() as client:
            # ارسال درخواست دسته‌ای
            llm_response = await client.post(
                f"{LLM_SERVICE_URL}/batch/generate",
                json=[req.dict() for req in requests],
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT * 2  # timeout بیشتر برای batch
            )
            
            if llm_response.status_code != 200:
                raise HTTPException(
                    status_code=llm_response.status_code,
                    detail="خطا در پردازش دسته‌ای"
                )
            
            return llm_response.json()
            
    except httpx.RequestError as e:
        logger.error(f"خطا در پردازش دسته‌ای: {e}")
        raise HTTPException(
            status_code=503,
            detail="خطا در ارتباط با سرویس پردازش دسته‌ای"
        )