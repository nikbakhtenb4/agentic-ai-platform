#!/usr/bin/env python3
"""
Enhanced Speech-to-Text Service
سرویس تبدیل گفتار به متن پیشرفته با پشتیبانی از منابع مختلف ورودی
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import whisper
import torch
import tempfile
import os
import httpx
import aiofiles
from typing import Optional, List, Union, Dict, Any
import logging
from pathlib import Path
import uvicorn
from pydantic import BaseModel, Field, HttpUrl
import base64
import time
import asyncio
from urllib.parse import urlparse
import mimetypes


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from utils.gpu_client import SimpleGPUClient, GPUClientError

    GPU_COORDINATION_AVAILABLE = True
except ImportError:
    GPU_COORDINATION_AVAILABLE = False
    logger.warning("⚠️ GPU Client not available - using direct GPU access")


# Response models
class STTResponse(BaseModel):
    text: str
    language: str
    confidence: float
    segments: list = []
    duration: float = 0.0
    processing_time: float = 0.0
    source_type: str = "unknown"  # file_upload, base64, url, local_path
    translation: Optional[dict] = None  # برای ترجمه آنلاین


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
    service: str = Field(
        "google", description="Translation service (google, microsoft, etc.)"
    )


class StreamingAudioConfig(BaseModel):
    """For real-time audio streaming (future feature)"""

    sample_rate: int = Field(16000, description="Audio sample rate")
    chunk_duration: float = Field(1.0, description="Chunk duration in seconds")
    language: Optional[str] = Field(None, description="Expected language")


class ModelInfoResponse(BaseModel):
    model_name: str
    supported_languages: List[str]
    device: str
    max_file_size_mb: int
    model_size: str


class FormatsResponse(BaseModel):
    supported_formats: List[str]
    max_file_size_mb: int


# STT Manager with enhanced capabilities
class EnhancedSTTManager:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.supported_languages = ["fa", "en", "auto"]
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE_MB", "25")) * 1024 * 1024
        self.max_url_file_size = (
            int(os.getenv("MAX_URL_FILE_SIZE_MB", "50")) * 1024 * 1024
        )
        self.supported_formats = [
            ".wav",
            ".mp3",
            ".m4a",
            ".flac",
            ".ogg",
            ".webm",
            ".aac",
        ]
        self.translation_services = {
            "google": "http://translate.googleapis.com/translate_a/single",
        }
        self.gpu_client = None
        self.gpu_coordination_enabled = GPU_COORDINATION_AVAILABLE
        self.coordinator_url = os.getenv(
            "GPU_COORDINATOR_URL", "http://gpu-coordinator:8080"
        )
        self.gpu_memory_requirement = float(os.getenv("STT_GPU_MEMORY_GB", "2.0"))

        # اضافه کردن retry mechanism بهتر
        self.gpu_retry_attempts = 3
        self.gpu_retry_delay = 5  # ثانیه

        logger.info(f"🎮 Enhanced STT Manager initialized - Device: {self.device}")
        if self.gpu_coordination_enabled:
            logger.info(f"🔄 GPU Coordination enabled - URL: {self.coordinator_url}")

    async def setup_gpu_coordination(self):
        """تنظیم GPU coordination"""
        if not self.gpu_coordination_enabled:
            logger.info("⚠️ GPU coordination not available")
            return False

        try:
            self.gpu_client = SimpleGPUClient(
                coordinator_url=self.coordinator_url, service_name="stt-service"
            )

            # بررسی در دسترس بودن coordinator
            if await self.gpu_client.is_coordinator_available():
                logger.info("✅ GPU Coordinator connected")
                return True
            else:
                logger.warning("⚠️ GPU Coordinator not reachable")
                return False

        except Exception as e:
            logger.error(f"❌ GPU coordination setup failed: {e}")
            return False

    async def request_gpu_with_retry(self, memory_gb: float = 2.0):
        """درخواست GPU با retry mechanism بهتر"""
        if not self.gpu_coordination_enabled:
            logger.info("🖥️ GPU coordination disabled, using direct GPU access")
            return True

        for attempt in range(self.gpu_retry_attempts):
            try:
                logger.info(
                    f"🔄 GPU request attempt {attempt + 1}/{self.gpu_retry_attempts}"
                )

                if not self.gpu_client:
                    await self.setup_gpu_coordination()

                if not self.gpu_client:
                    logger.warning("⚠️ GPU client not available")
                    return False

                # درخواست GPU
                allocation = await self.gpu_client.wait_for_gpu(
                    memory_gb=memory_gb,
                    priority="normal",
                    max_wait_time=30,  # کاهش wait time
                )

                if allocation.allocated:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(allocation.gpu_id)
                    self.device = f"cuda:{allocation.gpu_id}"
                    logger.info(f"✅ GPU {allocation.gpu_id} allocated for STT")
                    return True
                else:
                    logger.warning(f"⚠️ GPU allocation failed (attempt {attempt + 1})")

                    if attempt < self.gpu_retry_attempts - 1:
                        logger.info(
                            f"⏳ Waiting {self.gpu_retry_delay}s before next attempt..."
                        )
                        await asyncio.sleep(self.gpu_retry_delay)

            except Exception as e:
                logger.error(f"❌ GPU request attempt {attempt + 1} failed: {e}")
                if attempt < self.gpu_retry_attempts - 1:
                    await asyncio.sleep(self.gpu_retry_delay)

        logger.warning(
            "⚠️ All GPU allocation attempts failed, falling back to CPU/direct GPU"
        )
        return False

    async def load_model(self, model_size: str = "base"):
        """Load Whisper model with enhanced error handling"""
        try:
            logger.info(f"📥 Loading Whisper model: {model_size}")

            gpu_coordination_success = await self.request_gpu_with_retry(
                memory_gb=self.gpu_memory_requirement
            )

            if gpu_coordination_success:
                logger.info(f"🎮 Using coordinated GPU: {self.device}")
            else:
                logger.info(f"🖥️ Using direct GPU/CPU: {self.device}")

            # Load model with proper configuration
            self.model = whisper.load_model(
                model_size,
                device=self.device,
                download_root=os.getenv("WHISPER_MODEL_PATH", "/app/models"),
            )

            logger.info("✅ Whisper model loaded successfully")
            await self._test_model()

        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            # در صورت خطا، GPU را آزاد کنید
            if (
                self.gpu_client
                and hasattr(self.gpu_client, "is_gpu_allocated")
                and self.gpu_client.is_gpu_allocated()
            ):
                await self.gpu_client.release_gpu()
            raise

    async def _test_model(self):
        """Test model functionality"""
        try:
            # Create a test audio (1 second of silence)
            test_audio = whisper.pad_or_trim(torch.zeros(16000))
            if self.device.startswith("cuda"):
                test_audio = test_audio.cuda()

            # Quick test transcription
            _ = self.model.transcribe(
                test_audio.cpu().numpy(), fp16=False, verbose=False
            )
            logger.info("✅ Model test completed successfully")

        except Exception as e:
            logger.warning(f"⚠️ Model test failed: {e}")

    async def download_audio_from_url(self, url: str, max_size: int) -> str:
        """Download audio file from URL"""
        try:
            logger.info(f"📥 Downloading audio from: {url}")

            async with httpx.AsyncClient(timeout=60.0) as client:
                # First, get headers to check file size
                head_response = await client.head(url)

                if head_response.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot access URL: {head_response.status_code}",
                    )

                # Check content length
                content_length = head_response.headers.get("content-length")
                if content_length and int(content_length) > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large: {int(content_length) / 1024 / 1024:.1f}MB (max: {max_size / 1024 / 1024:.1f}MB)",
                    )

                # Check content type
                content_type = head_response.headers.get("content-type", "")
                if not content_type.startswith("audio/"):
                    logger.warning(f"⚠️ Content type may not be audio: {content_type}")

                # Download the file
                response = await client.get(url)

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Download failed: {response.status_code}",
                    )

                # Check actual size
                if len(response.content) > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Downloaded file too large: {len(response.content) / 1024 / 1024:.1f}MB",
                    )

                # Save to temporary file
                url_parsed = urlparse(url)
                filename = Path(url_parsed.path).name or "downloaded_audio"

                # Guess file extension from content type or URL
                if not Path(filename).suffix:
                    extension = mimetypes.guess_extension(content_type) or ".wav"
                    filename += extension

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(filename).suffix
                )
                temp_file.write(response.content)
                temp_file.close()

                logger.info(
                    f"✅ Downloaded {len(response.content) / 1024 / 1024:.1f}MB to {temp_file.name}"
                )

                return temp_file.name

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ URL download failed: {e}")
            raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")

    async def read_local_audio_file(self, file_path: str) -> str:
        """Read local audio file with validation"""
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=404, detail=f"File not found: {file_path}"
                )

            # Check if it's a file (not directory)
            if not os.path.isfile(file_path):
                raise HTTPException(
                    status_code=400, detail=f"Path is not a file: {file_path}"
                )

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {file_size / 1024 / 1024:.1f}MB (max: {self.max_file_size / 1024 / 1024:.1f}MB)",
                )

            # Check file extension
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in self.supported_formats:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported format: {file_extension}. Supported: {self.supported_formats}",
                )

            logger.info(
                f"📁 Reading local file: {file_path} ({file_size / 1024 / 1024:.1f}MB)"
            )

            return file_path

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Local file read failed: {e}")
            raise HTTPException(status_code=500, detail=f"File read error: {str(e)}")

    async def transcribe_audio_file(
        self,
        audio_file: UploadFile,
        language: Optional[str] = None,
        task: str = "transcribe",
        translate_to: Optional[str] = None,
    ) -> STTResponse:
        """Transcribe uploaded audio file"""
        return await self._transcribe_common(
            audio_source=audio_file,
            source_type="file_upload",
            language=language,
            task=task,
            translate_to=translate_to,
        )

    async def transcribe_base64_audio(
        self,
        audio_data: AudioData,
        task: str = "transcribe",
        translate_to: Optional[str] = None,
    ) -> STTResponse:
        """Transcribe base64 encoded audio"""
        return await self._transcribe_common(
            audio_source=audio_data,
            source_type="base64",
            language=audio_data.language,
            task=task,
            translate_to=translate_to,
        )

    async def transcribe_url_audio(
        self,
        audio_url: AudioURL,
        task: str = "transcribe",
        translate_to: Optional[str] = None,
    ) -> STTResponse:
        """Transcribe audio from URL"""
        return await self._transcribe_common(
            audio_source=audio_url,
            source_type="url",
            language=audio_url.language,
            task=task,
            translate_to=translate_to,
        )

    async def transcribe_local_audio(
        self,
        local_path: LocalAudioPath,
        task: str = "transcribe",
        translate_to: Optional[str] = None,
    ) -> STTResponse:
        """Transcribe local audio file"""
        return await self._transcribe_common(
            audio_source=local_path,
            source_type="local_path",
            language=local_path.language,
            task=task,
            translate_to=translate_to,
        )

    async def cleanup_gpu(self):
        """پاکسازی GPU coordination"""
        if (
            self.gpu_client
            and hasattr(self.gpu_client, "is_gpu_allocated")
            and self.gpu_client.is_gpu_allocated()
        ):
            await self.gpu_client.release_gpu()
            logger.info("✅ GPU released from coordination")

    async def _transcribe_common(
        self,
        audio_source: Union[UploadFile, AudioData, AudioURL, LocalAudioPath],
        source_type: str,
        language: Optional[str],
        task: str,
        translate_to: Optional[str] = None,
    ) -> STTResponse:
        """Common transcription logic for all input types"""
        start_time = time.time()

        if not self.model:
            raise HTTPException(status_code=500, detail="Model not loaded")

        temp_path = None
        cleanup_temp = False

        try:
            # Prepare audio file based on source type
            if source_type == "file_upload":
                # Handle uploaded file
                content = await audio_source.read()
                if len(content) > self.max_file_size:
                    raise HTTPException(status_code=413, detail="File too large")

                file_extension = Path(
                    audio_source.filename or "audio.wav"
                ).suffix.lower()
                if file_extension not in self.supported_formats:
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported format: {file_extension}"
                    )

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_extension
                )
                temp_file.write(content)
                temp_file.close()
                temp_path = temp_file.name
                cleanup_temp = True

            elif source_type == "base64":
                # Handle base64 audio
                try:
                    audio_bytes = base64.b64decode(audio_source.audio_data)
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid base64 data")

                if len(audio_bytes) > self.max_file_size:
                    raise HTTPException(status_code=413, detail="Audio data too large")

                file_extension = f".{audio_source.format.lower()}"
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_extension
                )
                temp_file.write(audio_bytes)
                temp_file.close()
                temp_path = temp_file.name
                cleanup_temp = True

            elif source_type == "url":
                # Handle URL download
                temp_path = await self.download_audio_from_url(
                    str(audio_source.url),
                    audio_source.max_file_size * 1024 * 1024
                    if hasattr(audio_source, "max_file_size")
                    else self.max_url_file_size,
                )
                cleanup_temp = True

            elif source_type == "local_path":
                # Handle local file
                temp_path = await self.read_local_audio_file(audio_source.file_path)
                cleanup_temp = False  # Don't delete original file

            # Validate language
            if language and language not in self.supported_languages:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported language: {language}. Supported: {self.supported_languages}",
                )

            # Prepare transcription options
            options = {
                "task": task,
                "fp16": torch.cuda.is_available() and self.device.startswith("cuda"),
                "verbose": False,
                "temperature": 0.0,
                "best_of": 1,
                "beam_size": 1,
            }

            if language and language != "auto":
                options["language"] = language

            # Transcribe audio
            logger.info(f"🎵 Transcribing from {source_type}")
            result = self.model.transcribe(temp_path, **options)

            processing_time = time.time() - start_time

            # Calculate confidence
            confidence = 0.0
            if "segments" in result and result["segments"]:
                confidence = sum(
                    1.0 - seg.get("no_speech_prob", 0.0) for seg in result["segments"]
                )
                confidence = confidence / len(result["segments"])

            # Prepare segments
            segments = []
            if "segments" in result:
                segments = [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip(),
                        "confidence": 1.0 - seg.get("no_speech_prob", 0.0),
                    }
                    for seg in result["segments"]
                ]

            # Prepare response
            response_data = STTResponse(
                text=result["text"].strip(),
                language=result.get("language", "unknown"),
                confidence=confidence,
                duration=result.get("duration", 0.0),
                processing_time=processing_time,
                segments=segments,
                source_type=source_type,
            )

            # Add translation if requested
            if translate_to and translate_to != response_data.language:
                try:
                    translation = await self.translate_text(
                        response_data.text,
                        source_language=response_data.language,
                        target_language=translate_to,
                    )
                    response_data.translation = translation
                    logger.info(
                        f"🌐 Translation added: {response_data.language} -> {translate_to}"
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Translation failed: {e}")
                    response_data.translation = {"error": str(e)}

            logger.info(
                f"✅ Transcription completed: {source_type} in {processing_time:.2f}s"
            )

            return response_data

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Transcription error: {str(e)}"
            )

        finally:
            # Cleanup temporary file if needed
            if cleanup_temp and temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    async def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
        service: str = "google",
    ) -> dict:
        """Translate text using online translation service"""
        try:
            if service == "google":
                return await self._translate_google(
                    text, source_language, target_language
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported translation service: {service}",
                )

        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise

    async def _translate_google(
        self, text: str, source_lang: str, target_lang: str
    ) -> dict:
        """Translate using Google Translate (unofficial API)"""
        try:
            # Convert language codes to Google format
            lang_map = {"fa": "fa", "en": "en", "auto": "auto"}
            sl = lang_map.get(source_lang, source_lang)
            tl = lang_map.get(target_lang, target_lang)

            url = "http://translate.googleapis.com/translate_a/single"
            params = {"client": "gtx", "sl": sl, "tl": tl, "dt": "t", "q": text}

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)

                if response.status_code == 200:
                    result = response.json()

                    # Extract translated text
                    if result and len(result) > 0 and len(result[0]) > 0:
                        translated_text = "".join(
                            [item[0] for item in result[0] if item[0]]
                        )

                        return {
                            "service": "google",
                            "source_language": source_lang,
                            "target_language": target_lang,
                            "original_text": text,
                            "translated_text": translated_text,
                            "success": True,
                        }
                    else:
                        raise Exception("Empty translation result")
                else:
                    raise Exception(f"Translation API error: {response.status_code}")

        except Exception as e:
            return {
                "service": "google",
                "source_language": source_lang,
                "target_language": target_lang,
                "original_text": text,
                "translated_text": None,
                "error": str(e),
                "success": False,
            }


# Initialize enhanced STT manager
stt_manager = None


# راه‌اندازی سرویس با مدیریت بهتر GPU
async def startup():
    """راه‌اندازی سرویس با مدیریت بهتر GPU"""
    global stt_manager

    try:
        logger.info("🚀 Starting Enhanced STT Service...")

        # ایجاد STT Manager
        stt_manager = EnhancedSTTManager()

        # بارگذاری مدل
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        await stt_manager.load_model(model_size)

        logger.info("✅ STT Service startup completed successfully")

    except Exception as e:
        logger.error(f"❌ STT Service startup failed: {e}")
        # ایجاد fallback manager
        stt_manager = EnhancedSTTManager()
        stt_manager.device = "cpu"
        logger.info("🖥️ Running in fallback mode")


# استفاده از lifespan به جای on_event (deprecated)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup()
    yield
    # Shutdown
    if stt_manager and stt_manager.gpu_client:
        try:
            await stt_manager.cleanup_gpu()
            logger.info("✅ GPU released successfully")
        except:
            pass


# تغییر تعریف app
app = FastAPI(
    title="Enhanced STT Service",
    description="Speech-to-Text Service with GPU Coordination",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
        "http://yourdomain.com",  # اضافه کردن دامین خودتون
        "*",  # در production محدود کنید
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Enhanced health check"""
    try:
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "gpu_name": torch.cuda.get_device_name(0)
                if torch.cuda.device_count() > 0
                else "Unknown",
                "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
                if torch.cuda.device_count() > 0
                else "Unknown",
            }

        gpu_coordination_info = {
            "enabled": stt_manager.gpu_coordination_enabled if stt_manager else False,
            "connected": stt_manager.gpu_client is not None if stt_manager else False,
            "allocated": stt_manager.gpu_client.is_gpu_allocated()
            if stt_manager
            and stt_manager.gpu_client
            and hasattr(stt_manager.gpu_client, "is_gpu_allocated")
            else False,
        }

        return {
            "status": "healthy",
            "service": "Enhanced STT Service",
            "version": "2.0.0",
            "device": stt_manager.device if stt_manager else "unknown",
            "model_loaded": stt_manager.model is not None if stt_manager else False,
            "supported_languages": stt_manager.supported_languages
            if stt_manager
            else [],
            "supported_formats": stt_manager.supported_formats if stt_manager else [],
            "max_file_size_mb": stt_manager.max_file_size / 1024 / 1024
            if stt_manager
            else 0,
            "max_url_file_size_mb": stt_manager.max_url_file_size / 1024 / 1024
            if stt_manager
            else 0,
            "gpu_info": gpu_info,
            "gpu_coordination_info": gpu_coordination_info,
            "features": [
                "file_upload",
                "base64_processing",
                "url_download",
                "local_file_access",
                "online_translation",
                "batch_processing",
                "multi_language_support",
                "gpu_acceleration",
            ],
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "Enhanced STT Service",
            "version": "2.0.0",
        }


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """دریافت اطلاعات مدل STT"""
    try:
        supported_languages = os.getenv("SUPPORTED_LANGUAGES", "fa,en").split(",")
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        max_file_size = int(os.getenv("MAX_FILE_SIZE_MB", "25"))

        return ModelInfoResponse(
            model_name=f"whisper-{model_size}",
            supported_languages=supported_languages,
            device=stt_manager.device if stt_manager else "unknown",
            max_file_size_mb=max_file_size,
            model_size=model_size,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"خطا در دریافت اطلاعات مدل: {str(e)}"
        )


@app.get("/formats", response_model=FormatsResponse)
async def get_supported_formats():
    """دریافت فرمت های صوتی پشتیبانی شده"""
    try:
        supported_formats = [
            "wav",
            "mp3",
            "m4a",
            "ogg",
            "flac",
            "aac",
            "wma",
            "mp4",
            "avi",
            "mkv",
            "mov",
        ]
        max_file_size = int(os.getenv("MAX_FILE_SIZE_MB", "25"))

        return FormatsResponse(
            supported_formats=supported_formats, max_file_size_mb=max_file_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در دریافت فرمت ها: {str(e)}")


@app.get("/gpu/status")
async def get_gpu_status():
    """وضعیت GPU برای STT"""
    try:
        gpu_allocated = False
        gpu_id = None

        if (
            stt_manager
            and stt_manager.gpu_client
            and hasattr(stt_manager.gpu_client, "is_gpu_allocated")
        ):
            gpu_allocated = stt_manager.gpu_client.is_gpu_allocated()
            if hasattr(stt_manager.gpu_client, "get_current_gpu_id"):
                gpu_id = stt_manager.gpu_client.get_current_gpu_id()

        return {
            "service": "stt-service",
            "gpu_coordination_enabled": stt_manager.gpu_coordination_enabled
            if stt_manager
            else False,
            "gpu_allocated": gpu_allocated,
            "gpu_id": gpu_id,
            "current_device": stt_manager.device if stt_manager else "unknown",
            "coordinator_url": stt_manager.coordinator_url if stt_manager else None,
            "model_loaded": stt_manager.model is not None if stt_manager else False,
        }
    except Exception as e:
        return {
            "service": "stt-service",
            "error": str(e),
            "gpu_coordination_enabled": False,
            "gpu_allocated": False,
        }


# File upload transcription
@app.post("/transcribe", response_model=STTResponse)
async def transcribe_audio_file(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
    translate_to: Optional[str] = Form(None),
):
    """
    Transcribe uploaded audio file with optional translation

    - **audio_file**: Audio file (WAV, MP3, M4A, etc.)
    - **language**: Source language code (fa, en, auto)
    - **task**: transcribe or translate
    - **translate_to**: Target language for translation (fa, en)
    """
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    return await stt_manager.transcribe_audio_file(
        audio_file, language, task, translate_to
    )


# Base64 transcription
@app.post("/transcribe-base64", response_model=STTResponse)
async def transcribe_base64(
    audio_data: AudioData, task: str = "transcribe", translate_to: Optional[str] = None
):
    """
    Transcribe base64 encoded audio (for mobile apps)

    - **audio_data**: Base64 encoded audio with metadata
    - **task**: transcribe or translate
    - **translate_to**: Target language for translation (fa, en)
    """
    return await stt_manager.transcribe_base64_audio(audio_data, task, translate_to)


# URL transcription
@app.post("/transcribe-url", response_model=STTResponse)
async def transcribe_url(
    audio_url: AudioURL, task: str = "transcribe", translate_to: Optional[str] = None
):
    """
    Transcribe audio from URL

    - **audio_url**: Audio URL with settings
    - **task**: transcribe or translate
    - **translate_to**: Target language for translation (fa, en)
    """
    return await stt_manager.transcribe_url_audio(audio_url, task, translate_to)


# Local file transcription
@app.post("/transcribe-local", response_model=STTResponse)
async def transcribe_local(
    local_path: LocalAudioPath,
    task: str = "transcribe",
    translate_to: Optional[str] = None,
):
    """
    Transcribe local audio file

    - **local_path**: Local file path with settings
    - **task**: transcribe or translate
    - **translate_to**: Target language for translation (fa, en)
    """
    return await stt_manager.transcribe_local_audio(local_path, task, translate_to)


# Translation endpoint
@app.post("/translate")
async def translate_text(translation_request: TranslationRequest):
    """
    Translate text using online services

    - **translation_request**: Translation parameters
    """
    try:
        result = await stt_manager.translate_text(
            text=translation_request.text,
            source_language=translation_request.source_language,
            target_language=translation_request.target_language,
            service=translation_request.service,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


# Batch transcription (enhanced)
@app.post("/transcribe-batch")
async def transcribe_batch(files: List[UploadFile] = File(...)):
    """Enhanced batch transcription with better error handling"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

    results = []
    total_start_time = time.time()

    for i, file in enumerate(files):
        try:
            logger.info(
                f"📦 Processing batch item {i + 1}/{len(files)}: {file.filename}"
            )
            result = await stt_manager.transcribe_audio_file(file)
            results.append(
                {
                    "index": i + 1,
                    "filename": file.filename,
                    "success": True,
                    "result": result.dict(),
                }
            )
        except Exception as e:
            logger.error(f"❌ Batch item {i + 1} failed: {file.filename} - {e}")
            results.append(
                {
                    "index": i + 1,
                    "filename": file.filename,
                    "success": False,
                    "error": str(e),
                }
            )

    total_time = time.time() - total_start_time
    successful_count = sum(1 for r in results if r.get("success", False))

    logger.info(
        f"📦 Batch completed: {successful_count}/{len(files)} successful in {total_time:.2f}s"
    )

    return {
        "results": results,
        "summary": {
            "total_files": len(files),
            "successful_count": successful_count,
            "failed_count": len(files) - successful_count,
            "success_rate": f"{successful_count / len(files) * 100:.1f}%",
            "total_processing_time": total_time,
        },
    }


# Enhanced languages endpoint
@app.get("/languages")
async def get_supported_languages():
    """Get comprehensive language and feature information"""
    return {
        "supported_languages": stt_manager.supported_languages if stt_manager else [],
        "language_names": {
            "fa": "Persian/Farsi (فارسی)",
            "en": "English",
            "auto": "Auto-detect",
        },
        "supported_formats": stt_manager.supported_formats if stt_manager else [],
        "max_file_sizes": {
            "upload_mb": stt_manager.max_file_size / 1024 / 1024 if stt_manager else 0,
            "url_download_mb": stt_manager.max_url_file_size / 1024 / 1024
            if stt_manager
            else 0,
        },
        "input_sources": [
            {
                "type": "file_upload",
                "endpoint": "/transcribe",
                "description": "Direct file upload",
            },
            {
                "type": "base64",
                "endpoint": "/transcribe-base64",
                "description": "Base64 encoded audio (mobile)",
            },
            {
                "type": "url",
                "endpoint": "/transcribe-url",
                "description": "Download from HTTP/HTTPS URL",
            },
            {
                "type": "local_path",
                "endpoint": "/transcribe-local",
                "description": "Local file system path",
            },
        ],
        "translation": {
            "supported": True,
            "services": ["google"],
            "endpoint": "/translate",
            "supported_languages": ["fa", "en"],
        },
        "features": [
            "multi_source_input",
            "online_translation",
            "batch_processing",
            "gpu_acceleration",
            "real_time_processing",
            "confidence_scoring",
            "segment_timestamps",
        ],
    }


# CORS preflight handler
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=False, log_level="info")
