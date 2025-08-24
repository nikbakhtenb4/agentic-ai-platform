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
        """تولید فایل صوتی تست و ذخیره آن روی دیسک"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave_data = np.sin(frequency * 2 * np.pi * t) * 0.3

        # تبدیل به 16-bit PCM
        audio_data = (wave_data * 32767).astype(np.int16)

        # ذخیره فایل WAV روی دیسک
        output_path = "test_audio_output.wav"
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        # تولید بافر برای استفاده در تست‌ها
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        buffer.seek(0)
        logger.info(f"✅ فایل صوتی در مسیر {output_path} ذخیره شد.")
        return buffer.getvalue()

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

    async def test_gpu_coordinator_status(self):
        """تست وضعیت GPU Coordinator"""
        try:
            async with self.session.get(f"{self.coordinator_url}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"🎯 GPU Coordinator Status:")
                    logger.info(f"   GPUs: {len(data.get('gpus', {}))}")
                    logger.info(
                        f"   Queue: {data.get('queue', {}).get('pending_tasks', 0)} pending"
                    )
                    return data
                else:
                    logger.error(f"❌ GPU Coordinator status failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"❌ GPU Coordinator status error: {e}")
            return None

    async def test_stt_transcription(self, audio_data, model_size="base"):
        """تست تبدیل گفتار به متن"""
        try:
            # آپلود فایل صوتی
            form_data = aiohttp.FormData()
            form_data.add_field(
                "audio_file",
                audio_data,
                filename="test_audio.wav",
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

    async def test_multiple_models(self):
        """تست چندین مدل مختلف"""
        models = ["tiny", "base", "small"]
        results = {}

        for model in models:
            logger.info(f"🔄 Testing model: {model}")
            audio_data = self.generate_test_audio(duration=3)

            result = await self.test_stt_transcription(audio_data, model)
            results[model] = result

            # فاصله بین تست‌ها
            await asyncio.sleep(2)

        return results

    async def test_concurrent_requests(self, num_requests=3):
        """تست درخواست‌های همزمان"""
        logger.info(f"🔄 Testing {num_requests} concurrent requests...")

        tasks = []
        for i in range(num_requests):
            audio_data = self.generate_test_audio(duration=2, frequency=440 + i * 100)
            task = self.test_stt_transcription(audio_data, "base")
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if r and not isinstance(r, Exception))
        logger.info(
            f"✅ Concurrent test results: {successful}/{num_requests} successful"
        )

        return results

    async def test_gpu_allocation_info(self):
        """تست اطلاعات تخصیص GPU"""
        try:
            async with self.session.get(f"{self.stt_url}/gpu-info") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"🎯 STT GPU Info: {data}")
                    return data
                else:
                    logger.warning(f"⚠️ GPU info not available: {response.status}")
                    return None
        except Exception as e:
            logger.warning(f"⚠️ GPU info error: {e}")
            return None

    async def test_error_handling(self):
        """تست مدیریت خطا"""
        logger.info("🔄 Testing error handling...")

        # تست فایل خراب
        try:
            form_data = aiohttp.FormData()
            form_data.add_field(
                "audio",
                b"invalid audio data",
                filename="invalid.wav",
                content_type="audio/wav",
            )

            async with self.session.post(
                f"{self.stt_url}/transcribe", data=form_data
            ) as response:
                if response.status >= 400:
                    error_data = await response.json()
                    logger.info(
                        f"✅ Error handling working: {error_data.get('detail', 'No detail')}"
                    )
                else:
                    logger.warning(f"⚠️ Expected error but got: {response.status}")

        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")


async def main():
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

        # 2. تست GPU Coordinator
        logger.info("=" * 50)
        logger.info("2. Testing GPU Coordinator")
        await tester.test_gpu_coordinator_status()

        # 3. تست تبدیل گفتار به متن ساده
        logger.info("=" * 50)
        logger.info("3. Testing Basic Transcription")
        test_audio = tester.generate_test_audio(duration=3)
        await tester.test_stt_transcription(test_audio, "base")

        # 4. تست چندین مدل
        logger.info("=" * 50)
        logger.info("4. Testing Multiple Models")
        await tester.test_multiple_models()

        # 5. تست درخواست‌های همزمان
        logger.info("=" * 50)
        logger.info("5. Testing Concurrent Requests")
        await tester.test_concurrent_requests(3)

        # 6. تست اطلاعات GPU
        logger.info("=" * 50)
        logger.info("6. Testing GPU Allocation Info")
        await tester.test_gpu_allocation_info()

        # 7. تست مدیریت خطا
        logger.info("=" * 50)
        logger.info("7. Testing Error Handling")
        await tester.test_error_handling()

    print("🎉 STT Service Tests Completed!")


if __name__ == "__main__":
    asyncio.run(main())
