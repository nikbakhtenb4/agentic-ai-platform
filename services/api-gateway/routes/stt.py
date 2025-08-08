# me
# services/api-gateway/routes/stt.py (Enhanced for Multiple Input Sources)
from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Form,
    Depends,
    BackgroundTasks,
)
from fastapi.responses import JSONResponse
from typing import Optional, List
import httpx
import os
import logging
import time
from pydantic import BaseModel, HttpUrl, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt", tags=["Speech-to-Text"])

STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "https://stt-service:8003")


# Request models
class AudioData(BaseModel):
    """For base64 encoded audio from mobile apps"""

    audio_data: str = Field(..., description="Base64 encoded audio data")
    filename: str = Field(..., description="Original filename")
    language: Optional[str] = Field(None, description="Source language (fa, en)")
    format: str = Field("wav", description="Audio format")


class AudioURL(BaseModel):
    """For audio URL from web or external sources"""

    url: HttpUrl = Field(..., description="HTTP/HTTPS URL to audio file")
    language: Optional[str] = Field(None, description="Source language (fa, en)")
    max_file_size: Optional[int] = Field(50, description="Max download size in MB")


class LocalAudioPath(BaseModel):
    """For local audio file path"""

    file_path: str = Field(..., description="Local file system path")
    language: Optional[str] = Field(None, description="Source language (fa, en)")


class TranslationRequest(BaseModel):
    """For translation requests"""

    text: str = Field(..., description="Text to translate")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    service: str = Field("google", description="Translation service")


class WebRTCStreamConfig(BaseModel):
    """For real-time audio streaming"""

    sample_rate: int = Field(16000, description="Audio sample rate")
    chunk_duration: float = Field(1.0, description="Chunk duration in seconds")
    language: Optional[str] = Field(None, description="Expected language")
    translate_to: Optional[str] = Field(
        None, description="Real-time translation target"
    )


# Helper function for logging requests
async def log_request(
    endpoint: str, processing_time: float, success: bool, error: str = None
):
    """Log STT requests for monitoring"""
    log_data = {
        "endpoint": endpoint,
        "processing_time": processing_time,
        "success": success,
        "timestamp": time.time(),
    }
    if error:
        log_data["error"] = error

    logger.info(f"STT Request: {log_data}")


@router.get("/health")
async def stt_health():
    """Check STT service health with detailed information"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{STT_SERVICE_URL}/health")

            if response.status_code == 200:
                health_data = response.json()

                # Add gateway-specific information
                health_data["gateway_info"] = {
                    "version": "2.0.0",
                    "endpoints_available": [
                        "/transcribe",
                        "/transcribe-base64",
                        "/transcribe-url",
                        "/transcribe-local",
                        "/transcribe-batch",
                        "/translate",
                    ],
                    "timestamp": time.time(),
                }

                return health_data
            else:
                raise HTTPException(
                    status_code=response.status_code, detail="STT service unhealthy"
                )

    except httpx.TimeoutException:
        logger.error("STT health check timeout")
        raise HTTPException(status_code=504, detail="STT service timeout")
    except Exception as e:
        logger.error(f"STT health check failed: {e}")
        raise HTTPException(status_code=503, detail="STT service unavailable")


@router.post("/transcribe")
async def transcribe_audio_file(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
    translate_to: Optional[str] = Form(None),
):
    """
    Transcribe uploaded audio file with optional online translation

    - **audio_file**: Audio file (WAV, MP3, M4A, FLAC, OGG, WebM, AAC)
    - **language**: Source language code (fa, en, auto) - optional
    - **task**: transcribe or translate (default: transcribe)
    - **translate_to**: Target language for translation (fa, en) - optional
    """

    start_time = time.time()

    try:
        # Validate file
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Validate file size at gateway level (basic check)
        content = await audio_file.read()
        max_size = 50 * 1024 * 1024  # 50MB at gateway level

        if len(content) > max_size:
            processing_time = time.time() - start_time
            await log_request("transcribe", processing_time, False, "File too large")
            raise HTTPException(
                status_code=413,
                detail=f"File size ({len(content) / 1024 / 1024:.1f}MB) exceeds limit (50MB)",
            )

        # Reset file position
        await audio_file.seek(0)

        # Prepare form data for STT service
        files = {"audio_file": (audio_file.filename, content, audio_file.content_type)}
        data = {}

        if language:
            data["language"] = language
        data["task"] = task
        if translate_to:
            data["translate_to"] = translate_to

        # Call STT service with extended timeout for translation
        timeout = 600.0 if translate_to else 300.0

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{STT_SERVICE_URL}/transcribe", files=files, data=data
            )

            processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                # Log successful transcription
                text_length = len(result.get("text", ""))
                logger.info(
                    f"File transcription completed: {audio_file.filename} -> "
                    f"{text_length} chars, {processing_time:.2f}s"
                )

                # Add translation info if available
                if result.get("translation"):
                    logger.info(
                        f"Translation included: {result['language']} -> {translate_to}"
                    )

                # Background logging
                background_tasks.add_task(
                    log_request, "transcribe", processing_time, True
                )

                return result
            else:
                error_detail = response.text
                logger.error(
                    f"STT service error: {response.status_code} - {error_detail}"
                )

                # Background logging
                background_tasks.add_task(
                    log_request, "transcribe", processing_time, False, error_detail
                )

                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"STT service error: {error_detail}",
                )

    except httpx.TimeoutException:
        processing_time = time.time() - start_time
        logger.error("STT file transcription timeout")
        background_tasks.add_task(
            log_request, "transcribe", processing_time, False, "Timeout"
        )
        raise HTTPException(status_code=504, detail="Transcription timeout")
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"File transcription failed: {e}")
        background_tasks.add_task(
            log_request, "transcribe", processing_time, False, str(e)
        )
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/transcribe-base64")
async def transcribe_base64_audio(
    audio_data: AudioData,
    task: str = "transcribe",
    translate_to: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Transcribe base64 encoded audio (optimized for mobile apps)

    - **audio_data**: Base64 encoded audio with metadata
    - **task**: transcribe or translate (default: transcribe)
    - **translate_to**: Target language for translation (fa, en) - optional
    """

    start_time = time.time()

    try:
        # Prepare JSON payload
        payload = audio_data.dict()
        params = {"task": task}
        if translate_to:
            params["translate_to"] = translate_to

        # Call STT service
        timeout = 600.0 if translate_to else 300.0

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{STT_SERVICE_URL}/transcribe-base64", json=payload, params=params
            )

            processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                # Log successful transcription
                text_length = len(result.get("text", ""))
                logger.info(
                    f"Base64 transcription completed: {audio_data.filename} -> "
                    f"{text_length} chars, {processing_time:.2f}s"
                )

                if result.get("translation"):
                    logger.info(
                        f"Translation included: {result['language']} -> {translate_to}"
                    )

                background_tasks.add_task(
                    log_request, "transcribe-base64", processing_time, True
                )

                return result
            else:
                error_detail = response.text
                logger.error(
                    f"STT base64 service error: {response.status_code} - {error_detail}"
                )

                background_tasks.add_task(
                    log_request,
                    "transcribe-base64",
                    processing_time,
                    False,
                    error_detail,
                )

                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"STT service error: {error_detail}",
                )

    except httpx.TimeoutException:
        processing_time = time.time() - start_time
        logger.error("STT base64 transcription timeout")
        background_tasks.add_task(
            log_request, "transcribe-base64", processing_time, False, "Timeout"
        )
        raise HTTPException(status_code=504, detail="Base64 transcription timeout")
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Base64 transcription failed: {e}")
        background_tasks.add_task(
            log_request, "transcribe-base64", processing_time, False, str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Base64 transcription failed: {str(e)}"
        )


@router.post("/transcribe-url")
async def transcribe_url_audio(
    audio_url: AudioURL,
    task: str = "transcribe",
    translate_to: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Transcribe audio from URL (web/streaming audio)

    - **audio_url**: Audio URL configuration
    - **task**: transcribe or translate (default: transcribe)
    - **translate_to**: Target language for translation (fa, en) - optional
    """

    start_time = time.time()

    try:
        # Prepare JSON payload
        payload = audio_url.dict()
        params = {"task": task}
        if translate_to:
            params["translate_to"] = translate_to

        # Call STT service with extended timeout for URL download + processing
        timeout = 900.0  # 15 minutes for URL download and processing

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{STT_SERVICE_URL}/transcribe-url", json=payload, params=params
            )

            processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                # Log successful transcription
                text_length = len(result.get("text", ""))
                logger.info(
                    f"URL transcription completed: {audio_url.url} -> "
                    f"{text_length} chars, {processing_time:.2f}s"
                )

                if result.get("translation"):
                    logger.info(
                        f"Translation included: {result['language']} -> {translate_to}"
                    )

                background_tasks.add_task(
                    log_request, "transcribe-url", processing_time, True
                )

                return result
            else:
                error_detail = response.text
                logger.error(
                    f"STT URL service error: {response.status_code} - {error_detail}"
                )

                background_tasks.add_task(
                    log_request, "transcribe-url", processing_time, False, error_detail
                )

                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"STT service error: {error_detail}",
                )

    except httpx.TimeoutException:
        processing_time = time.time() - start_time
        logger.error("STT URL transcription timeout")
        background_tasks.add_task(
            log_request, "transcribe-url", processing_time, False, "Timeout"
        )
        raise HTTPException(status_code=504, detail="URL transcription timeout")
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"URL transcription failed: {e}")
        background_tasks.add_task(
            log_request, "transcribe-url", processing_time, False, str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"URL transcription failed: {str(e)}"
        )


@router.post("/transcribe-local")
async def transcribe_local_audio(
    local_path: LocalAudioPath,
    task: str = "transcribe",
    translate_to: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Transcribe local audio file (server-side files)

    - **local_path**: Local file path configuration
    - **task**: transcribe or translate (default: transcribe)
    - **translate_to**: Target language for translation (fa, en) - optional
    """

    start_time = time.time()

    try:
        # Prepare JSON payload
        payload = local_path.dict()
        params = {"task": task}
        if translate_to:
            params["translate_to"] = translate_to

        # Call STT service
        timeout = 600.0 if translate_to else 300.0

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{STT_SERVICE_URL}/transcribe-local", json=payload, params=params
            )

            processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                # Log successful transcription
                text_length = len(result.get("text", ""))
                logger.info(
                    f"Local transcription completed: {local_path.file_path} -> "
                    f"{text_length} chars, {processing_time:.2f}s"
                )

                if result.get("translation"):
                    logger.info(
                        f"Translation included: {result['language']} -> {translate_to}"
                    )

                background_tasks.add_task(
                    log_request, "transcribe-local", processing_time, True
                )

                return result
            else:
                error_detail = response.text
                logger.error(
                    f"STT local service error: {response.status_code} - {error_detail}"
                )

                background_tasks.add_task(
                    log_request,
                    "transcribe-local",
                    processing_time,
                    False,
                    error_detail,
                )

                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"STT service error: {error_detail}",
                )

    except httpx.TimeoutException:
        processing_time = time.time() - start_time
        logger.error("STT local transcription timeout")
        background_tasks.add_task(
            log_request, "transcribe-local", processing_time, False, "Timeout"
        )
        raise HTTPException(status_code=504, detail="Local transcription timeout")
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Local transcription failed: {e}")
        background_tasks.add_task(
            log_request, "transcribe-local", processing_time, False, str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Local transcription failed: {str(e)}"
        )


@router.post("/translate")
async def translate_text(
    translation_request: TranslationRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Translate text using online translation services

    - **translation_request**: Translation configuration
    """

    start_time = time.time()

    try:
        # Call STT service translation endpoint
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{STT_SERVICE_URL}/translate", json=translation_request.dict()
            )

            processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                logger.info(
                    f"Translation completed: {translation_request.source_language} -> "
                    f"{translation_request.target_language}, {processing_time:.2f}s"
                )

                background_tasks.add_task(
                    log_request, "translate", processing_time, True
                )

                return result
            else:
                error_detail = response.text
                logger.error(
                    f"Translation service error: {response.status_code} - {error_detail}"
                )

                background_tasks.add_task(
                    log_request, "translate", processing_time, False, error_detail
                )

                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Translation error: {error_detail}",
                )

    except httpx.TimeoutException:
        processing_time = time.time() - start_time
        logger.error("Translation timeout")
        background_tasks.add_task(
            log_request, "translate", processing_time, False, "Timeout"
        )
        raise HTTPException(status_code=504, detail="Translation timeout")
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Translation failed: {e}")
        background_tasks.add_task(
            log_request, "translate", processing_time, False, str(e)
        )
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@router.post("/transcribe-batch")
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    translate_to: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Transcribe multiple audio files with optional translation

    - **files**: List of audio files (max 10 files, 100MB total)
    - **translate_to**: Target language for translation (fa, en) - optional
    """

    start_time = time.time()

    # Validate batch size
    if len(files) > 10:
        raise HTTPException(
            status_code=400, detail="Maximum 10 files allowed per batch"
        )

    try:
        # Prepare files for batch processing
        form_files = []
        total_size = 0

        for file in files:
            content = await file.read()
            total_size += len(content)

            # Check total size limit (increased for batch)
            if total_size > 200 * 1024 * 1024:  # 200MB total for batch
                raise HTTPException(
                    status_code=413, detail="Total batch size exceeds 200MB limit"
                )

            form_files.append(("files", (file.filename, content, file.content_type)))
            await file.seek(0)

        # Add translation parameter if specified
        data = {}
        if translate_to:
            data["translate_to"] = translate_to

        # Call STT service with extended timeout for batch processing
        timeout = 1800.0  # 30 minutes for batch

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{STT_SERVICE_URL}/transcribe-batch", files=form_files, data=data
            )

            processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                # Log batch results
                successful = result.get("summary", {}).get("successful_count", 0)
                total = result.get("summary", {}).get("total_files", len(files))

                logger.info(
                    f"Batch transcription completed: {successful}/{total} successful, "
                    f"{processing_time:.2f}s total"
                )

                if translate_to:
                    logger.info(f"Batch translation target: {translate_to}")

                background_tasks.add_task(
                    log_request, "transcribe-batch", processing_time, True
                )

                return result
            else:
                error_detail = response.text
                logger.error(
                    f"STT batch service error: {response.status_code} - {error_detail}"
                )

                background_tasks.add_task(
                    log_request,
                    "transcribe-batch",
                    processing_time,
                    False,
                    error_detail,
                )

                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Batch transcription error: {error_detail}",
                )

    except httpx.TimeoutException:
        processing_time = time.time() - start_time
        logger.error("STT batch transcription timeout")
        background_tasks.add_task(
            log_request, "transcribe-batch", processing_time, False, "Timeout"
        )
        raise HTTPException(status_code=504, detail="Batch transcription timeout")
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Batch transcription failed: {e}")
        background_tasks.add_task(
            log_request, "transcribe-batch", processing_time, False, str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Batch transcription failed: {str(e)}"
        )


@router.get("/languages")
async def get_supported_languages():
    """Get comprehensive language and feature information"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{STT_SERVICE_URL}/languages")

            if response.status_code == 200:
                result = response.json()

                # Add gateway-specific information
                result["gateway_features"] = {
                    "multiple_input_sources": True,
                    "online_translation": True,
                    "batch_processing": True,
                    "real_time_streaming": False,  # Future feature
                    "endpoints": {
                        "file_upload": "/api/v1/stt/transcribe",
                        "base64": "/api/v1/stt/transcribe-base64",
                        "url_download": "/api/v1/stt/transcribe-url",
                        "local_file": "/api/v1/stt/transcribe-local",
                        "batch": "/api/v1/stt/transcribe-batch",
                        "translation": "/api/v1/stt/translate",
                    },
                }

                return result
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to get language information",
                )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Language service timeout")
    except Exception as e:
        logger.error(f"Failed to get languages: {e}")
        raise HTTPException(status_code=503, detail="Language service unavailable")


# CORS preflight handler
@router.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests for all STT endpoints"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        },
    )
