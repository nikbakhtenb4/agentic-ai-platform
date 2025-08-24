#!/usr/bin/env python3
"""
اسکریپت تست سریع برای سرویس STT
Quick Test Script for STT Service
"""

import requests
import json
import sys
import os
from pathlib import Path


def test_stt_service():
    """تست سریع سرویس STT"""

    base_url = "http://localhost:8003"

    print("🧪 تست سریع سرویس Speech-to-Text")
    print("=" * 50)

    # 1. تست Health Check
    print("1️⃣ تست وضعیت سرویس...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ سرویس فعال: {data['service']}")
            print(f"   🎮 دستگاه: {data['device']}")
            print(f"   🤖 مدل لود شده: {data['model_loaded']}")

            if data.get("gpu_coordination_info", {}).get("enabled"):
                print(f"   🔄 GPU Coordination: فعال")
            else:
                print(f"   🖥️ GPU Coordination: غیرفعال")
        else:
            print(f"   ❌ خطا: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ خطا در اتصال: {e}")
        return False

    print()

    # 2. تست فرمت‌های پشتیبانی شده
    print("2️⃣ بررسی فرمت‌های پشتیبانی شده...")
    try:
        response = requests.get(f"{base_url}/formats")
        if response.status_code == 200:
            data = response.json()
            formats = ", ".join(data["supported_formats"][:5])  # نمایش 5 فرمت اول
            print(f"   📁 فرمت‌های پشتیبانی شده: {formats}...")
            print(f"   📏 حداکثر سایز فایل: {data['max_file_size_mb']} MB")
        else:
            print(f"   ❌ خطا در دریافت فرمت‌ها: {response.status_code}")
    except Exception as e:
        print(f"   ❌ خطا: {e}")

    print()

    # 3. تست GPU Status
    print("3️⃣ وضعیت GPU...")
    try:
        response = requests.get(f"{base_url}/gpu/status")
        if response.status_code == 200:
            data = response.json()
            print(
                f"   🎮 GPU Coordination: {'فعال' if data['gpu_coordination_enabled'] else 'غیرفعال'}"
            )
            print(f"   🖥️ دستگاه فعلی: {data['current_device']}")
            print(f"   🤖 مدل لود شده: {'بله' if data['model_loaded'] else 'خیر'}")
            if data.get("model_size"):
                print(f"   📊 اندازه مدل: {data['model_size']}")
        else:
            print(f"   ❌ خطا: {response.status_code}")
    except Exception as e:
        print(f"   ❌ خطا: {e}")

    print()

    # 4. تست آپلود فایل (اگر فایل تست وجود داشته باشد)
    print("4️⃣ تست آپلود فایل...")

    # جستجو برای فایل‌های صوتی تست
    test_files = []
    current_dir = Path(".")

    # جستجو در مسیر فعلی
    for ext in [".wav", ".mp3", ".m4a", ".flac"]:
        test_files.extend(list(current_dir.glob(f"*{ext}")))
        test_files.extend(list(current_dir.glob(f"**/*{ext}")))

    if test_files:
        test_file = test_files[0]  # استفاده از اولین فایل پیدا شده
        print(f"   📁 فایل تست پیدا شده: {test_file}")

        try:
            with open(test_file, "rb") as audio_file:
                files = {"audio_file": (test_file.name, audio_file, "audio/wav")}
                data = {
                    "language": "auto",  # تشخیص خودکار زبان
                    "task": "transcribe",
                }

                print("   🎵 در حال پردازش...")
                response = requests.post(
                    f"{base_url}/transcribe", files=files, data=data, timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ تبدیل موفق!")
                    print(
                        f"   📝 متن: '{result['text'][:100]}{'...' if len(result['text']) > 100 else ''}'"
                    )
                    print(f"   🌍 زبان: {result['language']}")
                    print(f"   📊 اطمینان: {result['confidence']:.2f}")
                    print(f"   ⏱️ زمان پردازش: {result['processing_time']:.2f} ثانیه")
                else:
                    print(f"   ❌ خطا در تبدیل: {response.status_code}")
                    print(f"   📄 پیام: {response.text[:200]}")

        except Exception as e:
            print(f"   ❌ خطا در آپلود: {e}")
    else:
        print("   ⚠️ فایل صوتی برای تست پیدا نشد")
        print("   💡 برای تست کامل، یک فایل .wav یا .mp3 در همین پوشه قرار دهید")

    print()

    # 5. تست ترجمه
    print("5️⃣ تست سرویس ترجمه...")
    try:
        payload = {
            "text": "سلام، چطور هستید؟",
            "source_language": "fa",
            "target_language": "en",
            "service": "google",
        }

        response = requests.post(f"{base_url}/translate", json=payload, timeout=15)

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"   ✅ ترجمه موفق")
                print(f"   📝 اصلی: {result['original_text']}")
                print(f"   🔄 ترجمه: {result['translated_text']}")
            else:
                print(f"   ❌ خطا در ترجمه: {result.get('error')}")
        else:
            print(f"   ❌ خطا در سرویس ترجمه: {response.status_code}")

    except Exception as e:
        print(f"   ❌ خطا در ترجمه: {e}")

    print()
    print("🎉 تست کامل شد!")
    print("\n💡 نکات:")
    print("   - برای تست بیشتر، فایل صوتی در همین پوشه قرار دهید")
    print("   - برای مشاهده API های کامل: http://localhost:8003/docs")
    print("   - برای تست دستی: http://localhost:8003/health")

    return True


def create_sample_audio():
    """ایجاد فایل صوتی نمونه برای تست"""
    try:
        import numpy as np
        import wave

        print("🎵 ایجاد فایل صوتی نمونه...")

        # تنظیمات
        duration = 3  # 3 ثانیه
        sample_rate = 16000
        frequency = 440  # فرکانس A4

        # تولید sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (np.sin(2 * np.pi * frequency * t) * 32767 * 0.3).astype(np.int16)

        # ذخیره فایل
        filename = "salam.wav"
        with wave.open(filename, "w") as wav_file:
            wav_file.setnchannels(1)  # مونو
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        print(f"   ✅ فایل تست ایجاد شد: {filename}")
        return filename

    except ImportError:
        print("   ❌ numpy یا wave در دسترس نیست")
        print("   💡 برای ایجاد فایل تست: pip install numpy")
        return None
    except Exception as e:
        print(f"   ❌ خطا در ایجاد فایل: {e}")
        return None


if __name__ == "__main__":
    print("🚀 شروع تست سرویس STT...")
    print()

    # اگر آرگومان --create-sample داده شده باشد
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        create_sample_audio()
        print()

    # اجرای تست اصلی
    success = test_stt_service()

    if success:
        print("\n✨ همه چیز به نظر درست کار می‌کنه!")
    else:
        print("\n❌ مشکلی در سرویس وجود داره. لاگ‌ها رو بررسی کن.")
        sys.exit(1)
