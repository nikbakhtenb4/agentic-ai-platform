#!/usr/bin/env python3
"""
STT Service Test Script
تست سرویس تبدیل گفتار به متن با GPU Sharing
"""

import asyncio
import aiohttp
import aiofiles
import json
import time
import io
import wave
import numpy as np
from pathlib import Path
import logging
import sys

# تنظیم logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# تنظیمات
STT_SERVICE_URL = "http://localhost:8003"
GPU_COORDINATOR_URL = "http://localhost:8080"
TEST_AUDIO_DIR = "./test_audio"


class STTServiceTester:
    def __init__(self):
        self.stt_url = STT_SERVICE_URL
        self.coordinator_url = GPU_COORDINATOR_URL
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def generate_test_audio(self, duration=5, sample_rate=16000, frequency=440):
        """تولید فایل صوتی تست"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave_data = np.sin(frequency * 2 * np.pi * t) * 0.3

        # تبدیل به 16-bit PCM
        audio_data = (wave_data * 32767).astype(np.int16)

        # ایجاد فایل WAV در حافظه
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        buffer.seek(0)
        return buffer.getvalue()

    async def load_user_audio(self, file_path):
        """خواندن فایل صوتی ارائه‌شده توسط کاربر"""
        try:
            async with aiofiles.open(file_path, "rb") as f:
                audio_data = await f.read()
                logger.info(f"✅ فایل صوتی از مسیر {file_path} خوانده شد.")
                return audio_data
        except Exception as e:
            logger.error(f"❌ خطا در خواندن فایل صوتی: {e}")
            return None

    async def test_health_check(self):
        """تست health check"""
        try:
            async with self.session.get(f"{self.stt_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ STT Service Health: {data}")
                    return True
                else:
                    logger.error(f"❌ STT Health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ STT Health check error: {e}")
            return False

    async def test_stt_transcription(self, audio_data, model_size="base"):
        """تست تبدیل گفتار به متن"""
        try:
            # آپلود فایل صوتی
            form_data = aiohttp.FormData()
            form_data.add_field(
                "audio_file",
                audio_data,
                filename="user_audio.wav",
                content_type="audio/wav",
            )
            form_data.add_field("model_size", model_size)
            form_data.add_field("language", "auto")

            start_time = time.time()

            async with self.session.post(
                f"{self.stt_url}/transcribe", data=form_data
            ) as response:
                processing_time = time.time() - start_time

                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ STT Transcription successful:")
                    logger.info(f"   Model: {model_size}")
                    logger.info(f"   Processing time: {processing_time:.2f}s")
                    logger.info(
                        f"   GPU used: {result.get('gpu_info', {}).get('gpu_id', 'CPU')}"
                    )
                    logger.info(f"   Text: {result.get('text', 'No text')[:100]}...")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ STT Transcription failed: {response.status}")
                    logger.error(f"   Error: {error_text}")
                    return None

        except Exception as e:
            logger.error(f"❌ STT Transcription error: {e}")
            return None


async def main(audio_file_path=None):
    """تست اصلی"""
    print("🎤 Starting STT Service Tests...")

    async with STTServiceTester() as tester:
        # 1. تست Health Check
        logger.info("=" * 50)
        logger.info("1. Testing Health Checks")
        health_ok = await tester.test_health_check()
        if not health_ok:
            logger.error("❌ STT Service not available!")
            return

        # 2. تست تبدیل گفتار به متن با فایل کاربر یا صوت تست
        logger.info("=" * 50)
        logger.info("2. Testing Transcription with User Audio or Test Audio")
        if audio_file_path:
            audio_data = await tester.load_user_audio(audio_file_path)
            if audio_data is None:
                logger.error("❌ توقف تست به دلیل خطا در خواندن فایل صوتی.")
                return
        else:
            audio_data = tester.generate_test_audio(duration=3)
            logger.info("⚠️ فایل صوتی کاربر ارائه نشده، از صوت تست استفاده می‌شود.")

        result = await tester.test_stt_transcription(audio_data, "base")
        if result and "text" in result:
            print(f"\n🎉 متن تبدیل‌شده: {result['text']}\n")

    print("🎉 STT Service Tests Completed!")


if __name__ == "__main__":
    # دریافت مسیر فایل صوتی از آرگومان خط فرمان یا مقدار پیش‌فرض
    audio_file_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(audio_file_path))
