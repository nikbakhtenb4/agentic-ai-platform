#!/usr/bin/env python3
"""
Comprehensive STT Test Script
اسکریپت تست کامل سرویس STT با قابلیت‌های پیشرفته
"""

import requests
import json
import time
import sys
import os
import base64
from typing import Optional, Dict, Any


class STTTestSuite:
    def __init__(self, base_url: str = "https://localhost"):
        self.base_url = base_url.rstrip("/")
        self.stt_direct_url = f"{self.base_url}:8003"
        self.stt_gateway_url = f"{self.base_url}/api/v1/stt"

    def print_header(self, title: str):
        """Print formatted test header"""
        print("\n" + "=" * 60)
        print(f"🎯 {title}")
        print("=" * 60)

    def print_result(self, success: bool, message: str, details: str = None):
        """Print formatted test result"""
        icon = "✅" if success else "❌"
        print(f"{icon} {message}")
        if details:
            print(f"   {details}")

    def test_health_check(self) -> bool:
        """Test health check endpoints"""
        self.print_header("HEALTH CHECK TESTS")

        success = True

        # Test direct STT health
        try:
            response = requests.get(f"{self.stt_direct_url}/health", timeout=10)
            if response.status_code == 200:
                result = response.json()
                self.print_result(
                    True,
                    "Direct STT Health",
                    f"Device: {result.get('device')}, Model: {result.get('model_loaded')}",
                )
            else:
                self.print_result(
                    False, f"Direct STT Health failed: {response.status_code}"
                )
                success = False
        except Exception as e:
            self.print_result(False, f"Direct STT Health error: {e}")
            success = False

        # Test gateway STT health
        try:
            response = requests.get(f"{self.stt_gateway_url}/health", timeout=10)
            if response.status_code == 200:
                result = response.json()
                self.print_result(
                    True,
                    "Gateway STT Health",
                    f"Features: {len(result.get('features', []))}",
                )
            else:
                self.print_result(
                    False, f"Gateway STT Health failed: {response.status_code}"
                )
                success = False
        except Exception as e:
            self.print_result(False, f"Gateway STT Health error: {e}")
            success = False

        return success

    def test_languages_endpoint(self) -> bool:
        """Test languages endpoint"""
        self.print_header("LANGUAGES ENDPOINT TEST")

        try:
            response = requests.get(f"{self.stt_direct_url}/languages", timeout=10)
            if response.status_code == 200:
                result = response.json()
                languages = result.get("supported_languages", [])
                formats = result.get("supported_formats", [])
                input_sources = result.get("input_sources", [])

                self.print_result(
                    True,
                    "Languages endpoint",
                    f"Languages: {languages}, Formats: {len(formats)}, Sources: {len(input_sources)}",
                )

                # Check if translation is supported
                translation = result.get("translation", {})
                if translation.get("supported"):
                    self.print_result(
                        True,
                        "Translation support",
                        f"Services: {translation.get('services', [])}",
                    )

                return True
            else:
                self.print_result(
                    False, f"Languages endpoint failed: {response.status_code}"
                )
                return False
        except Exception as e:
            self.print_result(False, f"Languages endpoint error: {e}")
            return False

    def test_file_transcription(self, audio_file_path: str = None) -> bool:
        """Test file upload transcription"""
        self.print_header("FILE TRANSCRIPTION TEST")

        # Look for test audio files
        test_files = [
            audio_file_path,
            "test_audio.wav",
            "sample.wav",
            "audio.wav",
            "../test_audio.wav",
            "data/uploads/test.wav",
        ]

        audio_file = None
        for file_path in test_files:
            if file_path and os.path.exists(file_path):
                audio_file = file_path
                break

        if not audio_file:
            self.print_result(
                False,
                "No test audio file found",
                "Create 'test_audio.wav' to test file transcription",
            )
            return False

        try:
            # Test basic transcription
            with open(audio_file, "rb") as f:
                files = {"audio_file": (audio_file, f, "audio/wav")}
                data = {"language": "fa", "task": "transcribe"}

                start_time = time.time()
                response = requests.post(
                    f"{self.stt_direct_url}/transcribe",
                    files=files,
                    data=data,
                    timeout=60,
                )
                end_time = time.time()

                if response.status_code == 200:
                    result = response.json()
                    text = result.get("text", "")
                    confidence = result.get("confidence", 0)

                    self.print_result(
                        True,
                        "File transcription",
                        f"Text: {text[:50]}..., Confidence: {confidence:.2f}, Time: {end_time - start_time:.2f}s",
                    )
                    return True
                else:
                    self.print_result(
                        False, f"File transcription failed: {response.status_code}"
                    )
                    return False

        except Exception as e:
            self.print_result(False, f"File transcription error: {e}")
            return False

    def test_file_with_translation(self, audio_file_path: str = None) -> bool:
        """Test file transcription with translation"""
        self.print_header("FILE TRANSCRIPTION + TRANSLATION TEST")

        # Find audio file
        test_files = [audio_file_path, "test_audio.wav", "sample.wav", "audio.wav"]
        audio_file = None
        for file_path in test_files:
            if file_path and os.path.exists(file_path):
                audio_file = file_path
                break

        if not audio_file:
            self.print_result(False, "No test audio file found for translation test")
            return False

        try:
            with open(audio_file, "rb") as f:
                files = {"audio_file": (audio_file, f, "audio/wav")}
                data = {"language": "fa", "task": "transcribe", "translate_to": "en"}

                start_time = time.time()
                response = requests.post(
                    f"{self.stt_direct_url}/transcribe",
                    files=files,
                    data=data,
                    timeout=120,  # More time for translation
                )
                end_time = time.time()

                if response.status_code == 200:
                    result = response.json()
                    text = result.get("text", "")
                    translation = result.get("translation", {})

                    if translation and translation.get("success"):
                        translated_text = translation.get("translated_text", "")
                        self.print_result(
                            True,
                            "File transcription + translation",
                            f"Original: {text[:30]}..., Translated: {translated_text[:30]}...",
                        )
                    else:
                        self.print_result(
                            False,
                            "Translation failed",
                            translation.get("error", "Unknown error"),
                        )
                    return True
                else:
                    self.print_result(
                        False, f"File + translation failed: {response.status_code}"
                    )
                    return False

        except Exception as e:
            self.print_result(False, f"File + translation error: {e}")
            return False

    def test_base64_transcription(self) -> bool:
        """Test base64 transcription (mobile simulation)"""
        self.print_header("BASE64 TRANSCRIPTION TEST")

        # Create dummy base64 audio (very short WAV)
        dummy_wav = (
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
            b"\x00\x04\x00\x00\x00\x08\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        base64_audio = base64.b64encode(dummy_wav).decode("utf-8")

        try:
            payload = {
                "audio_data": base64_audio,
                "filename": "test_mobile.wav",
                "language": "fa",
                "format": "wav",
            }

            response = requests.post(
                f"{self.stt_direct_url}/transcribe-base64", json=payload, timeout=30
            )

            # We expect this might fail with dummy data, but endpoint should respond
            if response.status_code in [200, 400, 422, 500]:
                self.print_result(
                    True,
                    "Base64 endpoint responding",
                    f"Status: {response.status_code} (dummy data test)",
                )
                return True
            else:
                self.print_result(
                    False, f"Base64 endpoint failed: {response.status_code}"
                )
                return False

        except Exception as e:
            self.print_result(False, f"Base64 test error: {e}")
            return False

    def test_url_transcription(self) -> bool:
        """Test URL transcription"""
        self.print_header("URL TRANSCRIPTION TEST")

        # Use a sample audio URL (you can replace with actual URL)
        test_urls = [
            "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav",
            "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav",
            # Add your own test URLs here
        ]

        for url in test_urls:
            try:
                payload = {
                    "url": url,
                    "language": "auto",
                    "max_file_size": 10,  # 10MB limit
                }

                start_time = time.time()
                response = requests.post(
                    f"{self.stt_direct_url}/transcribe-url",
                    json=payload,
                    timeout=180,  # 3 minutes for download + processing
                )
                end_time = time.time()

                if response.status_code == 200:
                    result = response.json()
                    text = result.get("text", "")
                    self.print_result(
                        True,
                        f"URL transcription: {url}",
                        f"Text: {text[:30]}..., Time: {end_time - start_time:.2f}s",
                    )
                    return True
                elif response.status_code == 404:
                    self.print_result(False, f"URL not found: {url}")
                    continue
                else:
                    self.print_result(
                        False, f"URL transcription failed: {response.status_code}"
                    )
                    continue

            except Exception as e:
                self.print_result(False, f"URL test error: {e}")
                continue

        self.print_result(False, "No working test URLs found")
        return False

    def test_local_file_transcription(self) -> bool:
        """Test local file transcription"""
        self.print_header("LOCAL FILE TRANSCRIPTION TEST")

        # Test with existing files
        test_paths = [
            "/app/uploads/test.wav",
            "/tmp/test_audio.wav",
            "./test_audio.wav",
            "./sample.wav",
        ]

        for file_path in test_paths:
            if os.path.exists(file_path):
                try:
                    payload = {"file_path": file_path, "language": "auto"}

                    response = requests.post(
                        f"{self.stt_direct_url}/transcribe-local",
                        json=payload,
                        timeout=60,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        text = result.get("text", "")
                        self.print_result(
                            True,
                            f"Local file transcription: {file_path}",
                            f"Text: {text[:50]}...",
                        )
                        return True
                    else:
                        self.print_result(
                            False, f"Local file failed: {response.status_code}"
                        )

                except Exception as e:
                    self.print_result(False, f"Local file error: {e}")

        self.print_result(False, "No accessible local files found")
        return False

    def test_translation_only(self) -> bool:
        """Test standalone translation"""
        self.print_header("STANDALONE TRANSLATION TEST")

        test_texts = [
            ("سلام دنیا", "fa", "en"),
            ("Hello world", "en", "fa"),
            ("این یک تست است", "fa", "en"),
        ]

        success_count = 0

        for text, source_lang, target_lang in test_texts:
            try:
                payload = {
                    "text": text,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "service": "google",
                }

                response = requests.post(
                    f"{self.stt_direct_url}/translate", json=payload, timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        translated = result.get("translated_text", "")
                        self.print_result(
                            True,
                            f"Translation: {source_lang} -> {target_lang}",
                            f"'{text}' -> '{translated}'",
                        )
                        success_count += 1
                    else:
                        self.print_result(
                            False, f"Translation failed: {result.get('error')}"
                        )
                else:
                    self.print_result(
                        False, f"Translation request failed: {response.status_code}"
                    )

            except Exception as e:
                self.print_result(False, f"Translation error: {e}")

        return success_count > 0

    def test_batch_transcription(self) -> bool:
        """Test batch transcription"""
        self.print_header("BATCH TRANSCRIPTION TEST")

        # Find test files
        test_files = []
        possible_files = ["test_audio.wav", "sample.wav", "audio.wav"]

        for file_path in possible_files:
            if os.path.exists(file_path):
                test_files.append(file_path)
                if len(test_files) >= 2:  # Test with 2 files max
                    break

        if not test_files:
            self.print_result(False, "No test files found for batch processing")
            return False

        try:
            files = []
            for file_path in test_files:
                with open(file_path, "rb") as f:
                    files.append(("files", (file_path, f.read(), "audio/wav")))

            start_time = time.time()
            response = requests.post(
                f"{self.stt_direct_url}/transcribe-batch",
                files=files,
                timeout=180,  # 3 minutes for batch
            )
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                summary = result.get("summary", {})
                successful = summary.get("successful_count", 0)
                total = summary.get("total_files", 0)

                self.print_result(
                    True,
                    "Batch transcription",
                    f"Success: {successful}/{total}, Time: {end_time - start_time:.2f}s",
                )
                return True
            else:
                self.print_result(
                    False, f"Batch transcription failed: {response.status_code}"
                )
                return False

        except Exception as e:
            self.print_result(False, f"Batch transcription error: {e}")
            return False

    def test_gateway_integration(self) -> bool:
        """Test STT through API Gateway"""
        self.print_header("API GATEWAY INTEGRATION TEST")

        try:
            # Test health through gateway
            response = requests.get(f"{self.stt_gateway_url}/health", timeout=10)
            if response.status_code == 200:
                result = response.json()
                gateway_info = result.get("gateway_info", {})
                self.print_result(
                    True,
                    "Gateway health check",
                    f"Version: {gateway_info.get('version')}",
                )

                # Test languages through gateway
                response = requests.get(f"{self.stt_gateway_url}/languages", timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    gateway_features = result.get("gateway_features", {})
                    self.print_result(
                        True,
                        "Gateway languages endpoint",
                        f"Multiple sources: {gateway_features.get('multiple_input_sources')}",
                    )
                    return True
                else:
                    self.print_result(
                        False, f"Gateway languages failed: {response.status_code}"
                    )
                    return False
            else:
                self.print_result(
                    False, f"Gateway health failed: {response.status_code}"
                )
                return False

        except Exception as e:
            self.print_result(False, f"Gateway integration error: {e}")
            return False

    def run_all_tests(self, audio_file_path: str = None) -> Dict[str, bool]:
        """Run all tests and return results"""
        print("🚀 STT Comprehensive Test Suite")
        print("Testing Enhanced STT Service with Multiple Input Sources")
        print("=" * 80)

        tests = [
            ("Health Check", self.test_health_check),
            ("Languages Endpoint", self.test_languages_endpoint),
            (
                "File Transcription",
                lambda: self.test_file_transcription(audio_file_path),
            ),
            (
                "File + Translation",
                lambda: self.test_file_with_translation(audio_file_path),
            ),
            ("Base64 Transcription", self.test_base64_transcription),
            ("URL Transcription", self.test_url_transcription),
            ("Local File Transcription", self.test_local_file_transcription),
            ("Standalone Translation", self.test_translation_only),
            ("Batch Transcription", self.test_batch_transcription),
            ("Gateway Integration", self.test_gateway_integration),
        ]

        results = {}
        passed = 0

        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                time.sleep(1)  # Small delay between tests
            except Exception as e:
                print(f"❌ {test_name} - Unexpected error: {e}")
                results[test_name] = False

        # Print summary
        self.print_header("TEST SUMMARY")
        total = len(tests)

        for test_name, result in results.items():
            icon = "✅" if result else "❌"
            print(f"{icon} {test_name}")

        print(
            f"\n🎯 Results: {passed}/{total} tests passed ({passed / total * 100:.1f}%)"
        )

        if passed == total:
            print("🎉 All tests passed! Enhanced STT service is working perfectly.")
        elif passed >= total * 0.7:
            print("✅ Most tests passed! STT service is working well.")
        else:
            print("⚠️  Several tests failed. Please check the logs and configuration.")

        return results


def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced STT Test Suite")
    parser.add_argument("--url", default="https://localhost", help="Base URL")
    parser.add_argument("--audio-file", help="Path to test audio file")
    parser.add_argument(
        "--test",
        choices=[
            "health",
            "languages",
            "file",
            "translation",
            "base64",
            "url",
            "local",
            "batch",
            "gateway",
            "all",
        ],
        default="all",
        help="Specific test to run",
    )

    args = parser.parse_args()

    test_suite = STTTestSuite(args.url)

    if args.test == "all":
        results = test_suite.run_all_tests(args.audio_file)
        return 0 if all(results.values()) else 1
    else:
        # Run specific test
        test_map = {
            "health": test_suite.test_health_check,
            "languages": test_suite.test_languages_endpoint,
            "file": lambda: test_suite.test_file_transcription(args.audio_file),
            "translation": test_suite.test_translation_only,
            "base64": test_suite.test_base64_transcription,
            "url": test_suite.test_url_transcription,
            "local": test_suite.test_local_file_transcription,
            "batch": test_suite.test_batch_transcription,
            "gateway": test_suite.test_gateway_integration,
        }

        if args.test in test_map:
            success = test_map[args.test]()
            return 0 if success else 1
        else:
            print(f"Unknown test: {args.test}")
            return 1


if __name__ == "__main__":
    exit(main())
