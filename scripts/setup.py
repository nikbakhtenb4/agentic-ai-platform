#!/usr/bin/env python3
"""
Agentic AI Platform Setup Script - Enhanced Version (Windows Fixed)
راه‌اندازی اولیه پلتفرم با LLM و STT Services + GPU Sharing
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path
import platform
import locale


def set_windows_encoding():
    """تنظیم encoding برای Windows"""
    if platform.system() == "Windows":
        # Set console encoding to UTF-8
        try:
            os.system("chcp 65001 >nul 2>&1")
            # Set environment variables for proper encoding
            os.environ["PYTHONIOENCODING"] = "utf-8"
            os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"

            # Try to set locale to UTF-8
            try:
                locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
            except:
                try:
                    locale.setlocale(locale.LC_ALL, "C.UTF-8")
                except:
                    pass

            print("✅ Windows encoding configured for UTF-8")
            return True
        except Exception as e:
            print(f"⚠️  Could not set Windows encoding: {e}")
            return False
    return True


def print_banner():
    """نمایش بنر شروع"""
    banner = """
    ╔══════════════════════════════════════╗
    ║        Agentic AI Platform           ║
    ║   LLM + STT + GPU Sharing Setup     ║
    ╚══════════════════════════════════════╝
    """
    print(banner)


def load_environment():
    """Load environment variables from .env file"""
    print("🔧 Loading environment variables...")

    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        return False

    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

        print("✅ Environment variables loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        return False


def check_requirements():
    """بررسی وجود Docker و Docker Compose"""
    print("🔍 Checking requirements...")

    try:
        result = subprocess.run(
            ["docker", "--version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        docker_version = result.stdout.strip()
        print(f"✅ Docker: {docker_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not installed or not in PATH")
        print("💡 Install Docker Desktop: https://docs.docker.com/get-docker/")
        return False

    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        compose_version = result.stdout.strip()
        print(f"✅ Docker Compose: {compose_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker Compose is not installed or not in PATH")
        print("💡 Install Docker Compose: https://docs.docker.com/compose/install/")
        return False

    # Check Docker daemon
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="ignore",
        )
        if result.returncode != 0:
            print("❌ Docker daemon is not running")
            print("💡 Start Docker Desktop or Docker service")
            return False
        print("✅ Docker daemon is running")
    except subprocess.TimeoutExpired:
        print("⚠️  Docker daemon check timed out")
    except Exception as e:
        print(f"⚠️  Docker daemon check failed: {e}")

    return True


def check_gpu_support():
    """بررسی پشتیبانی GPU و تنظیمات CUDA"""
    print("🎮 Checking GPU support and CUDA configuration...")

    gpu_available = False

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")

            # استخراج اطلاعات GPU
            lines = result.stdout.split("\n")
            for line in lines:
                if "MiB" in line and "/" in line:
                    print(f"   💾 GPU Memory: {line.strip()}")
                    break

            gpu_available = True
        else:
            print("⚠️  No NVIDIA GPU detected")
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found")

    # بررسی Docker GPU runtime
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if "nvidia" in result.stdout.lower():
            print("✅ Docker GPU runtime available")
        else:
            print("⚠️  Docker GPU runtime not detected")
            if gpu_available:
                print("   💡 Install nvidia-docker for GPU acceleration")
                print(
                    "   💡 Guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
                )
    except:
        pass

    if gpu_available:
        print("🚀 GPU acceleration will be used for both LLM and STT services")
        print("🔄 GPU Coordinator will manage resource sharing")
    else:
        print("🖥️  CPU-only mode will be used")

    return gpu_available


def run_docker_command_safe(cmd, timeout=600, ignore_errors=False):
    """
    اجرای دستورات Docker با handling بهتر برای Windows
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["DOCKER_BUILDKIT"] = "1"

    # برای Windows از shell=True استفاده می‌کنیم
    shell_mode = platform.system() == "Windows"

    try:
        # برای جلوگیری از timeout، از Popen استفاده می‌کنیم
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # ترکیب stderr با stdout
            text=True,
            encoding="utf-8",
            errors="ignore",  # ignore encoding errors
            shell=shell_mode,
            env=env,
            bufsize=1,
            universal_newlines=True,
        )

        # خواندن output به صورت real-time
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # نمایش progress برای build های طولانی
                if any(
                    keyword in output.lower()
                    for keyword in ["downloading", "extracting", "pulling"]
                ):
                    print(f"   📥 {output.strip()[:80]}...")
                elif "error" in output.lower() and not ignore_errors:
                    print(f"   ❌ {output.strip()}")

        # انتظار برای پایان process
        return_code = process.poll()

        result_output = "\n".join(output_lines)

        if return_code == 0:
            return True, result_output
        else:
            return False, result_output

    except Exception as e:
        return False, f"Command execution failed: {e}"


def build_services():
    """ساخت سرویس‌ها با error handling بهتر برای Windows"""
    print("🏗️  Building services with GPU sharing support...")

    # لیست سرویس‌ها برای build
    services_to_build = [
        ("test-service", "🧪 Test Service"),
        ("gpu-coordinator", "🔄 GPU Coordinator"),
        ("llm-service", "🤖 LLM Service"),
        ("stt-service", "🎙️  STT Service"),
        ("api-gateway", "🌐 API Gateway"),
    ]

    successful_builds = []
    failed_builds = []

    for service_name, display_name in services_to_build:
        print(f"\n{display_name}...")

        # بررسی وجود Dockerfile
        dockerfile_paths = [
            f"services/{service_name}/Dockerfile",
            f"services/audio-service/stt/Dockerfile"
            if service_name == "stt-service"
            else None,
        ]

        dockerfile_path = None
        for path in dockerfile_paths:
            if path and Path(path).exists():
                dockerfile_path = Path(path)
                break

        if not dockerfile_path:
            print(f"⚠️  Dockerfile not found for {service_name}, skipping...")
            continue

        # Build service با Windows-safe approach
        build_success = False

        print(f"   🔨 Building {service_name}...")

        # Strategy 1: Normal build
        success, output = run_docker_command_safe(
            ["docker", "compose", "build", service_name], timeout=900
        )

        if success:
            print(f"✅ {display_name} built successfully")
            successful_builds.append(service_name)
            build_success = True
        else:
            print(f"⚠️  {display_name} build failed, trying with --no-cache...")

            # Strategy 2: Build with --no-cache
            success, output = run_docker_command_safe(
                ["docker", "compose", "build", service_name, "--no-cache"], timeout=1200
            )

            if success:
                print(f"✅ {display_name} built successfully (no-cache)")
                successful_builds.append(service_name)
                build_success = True
            else:
                print(f"❌ {display_name} build failed")
                # نمایش خطا (فقط آخرین خطوط)
                error_lines = output.split("\n")[-5:]
                for line in error_lines:
                    if line.strip() and "error" in line.lower():
                        print(f"   🔍 {line.strip()}")

        if not build_success:
            failed_builds.append((service_name, display_name))

    # Summary
    print(f"\n📊 Build Summary:")
    print(f"   ✅ Successful: {len(successful_builds)} services")
    print(f"   ❌ Failed: {len(failed_builds)} services")

    if successful_builds:
        print(f"   Built services: {', '.join(successful_builds)}")

    if failed_builds:
        print(f"   Failed services: {', '.join([s[0] for s in failed_builds])}")
        print("\n💡 Manual build suggestions:")
        for service_name, display_name in failed_builds:
            print(f"   docker compose build {service_name} --no-cache --progress=plain")

        # If critical services failed, offer to continue anyway
        critical_services = ["api-gateway", "llm-service"]
        critical_failed = [s for s in failed_builds if s[0] in critical_services]

        if critical_failed:
            print(f"\n⚠️  Critical services failed: {[s[0] for s in critical_failed]}")
            response = input("Continue with partial setup? (y/N): ").lower().strip()
            if response != "y":
                return False

    return len(successful_builds) > 0


def start_services():
    """راه‌اندازی سرویس‌ها"""
    print("🚀 Starting services...")

    # ابتدا infrastructure services
    infrastructure_services = ["postgres", "redis"]

    print("🗄️  Starting infrastructure services...")
    for service in infrastructure_services:
        success, output = run_docker_command_safe(
            ["docker", "compose", "up", "-d", service], timeout=120
        )

        if success:
            print(f"✅ {service.title()} started")
            time.sleep(2)
        else:
            print(f"⚠️  {service.title()} start failed")

    # سپس GPU Coordinator
    print("🔄 Starting GPU Coordinator...")
    success, output = run_docker_command_safe(
        ["docker", "compose", "up", "-d", "gpu-coordinator"], timeout=120
    )

    if success:
        print("✅ GPU Coordinator started")
        time.sleep(5)  # انتظار برای آماده شدن
    else:
        print("⚠️  GPU Coordinator start failed, continuing...")

    # شروع همه سرویس‌ها
    print("🌐 Starting all services...")
    success, output = run_docker_command_safe(
        ["docker", "compose", "up", "-d"], timeout=180
    )

    if success:
        print("✅ All services startup command executed")

        # نمایش وضعیت containers
        time.sleep(3)
        success, ps_output = run_docker_command_safe(
            ["docker", "compose", "ps"], timeout=30
        )

        if success:
            print("📊 Container status:")
            print(ps_output)

        return True
    else:
        print("❌ Failed to start services")
        print("🔍 Checking for failed containers...")
        return False


def check_llm_model():
    """بررسی وجود مدل LLM"""
    print("🤖 Checking LLM model...")

    # مسیر مدل از environment variable یا default
    model_path_env = os.getenv("MODEL_PATH", "/app/models/gpt2-fa")
    local_model_path = model_path_env.replace("/app/", "data/")

    model_path = Path(local_model_path)
    if not model_path.exists():
        print(f"❌ LLM model directory not found!")
        print(f"   Expected path: {model_path.absolute()}")
        print("   Please ensure the GPT2-FA model is placed in the correct directory")
        print("   💡 Download from: https://huggingface.co/YOUR_MODEL_NAME")
        return False

    # بررسی فایل‌های مورد نیاز مدل
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "vocab.json",
    ]

    missing_files = []
    existing_files = []

    for file_name in required_files:
        file_path = model_path / file_name
        if file_path.exists():
            existing_files.append(file_name)
        else:
            missing_files.append(file_name)

    print(f"   ✅ Found {len(existing_files)} model files: {', '.join(existing_files)}")

    if missing_files:
        print(f"   ⚠️  Missing files: {', '.join(missing_files)}")
        print("   The model might still work, but please verify completeness")
    else:
        print("✅ All essential LLM model files found")

    # بررسی حجم مدل
    try:
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"📦 LLM Model size: {size_mb:.1f} MB")

        if size_mb < 10:
            print("⚠️  LLM Model size seems small - please verify model integrity")
        elif size_mb > 5000:
            print("💾 Large model detected - ensure sufficient system memory")

    except Exception as e:
        print(f"⚠️  Could not calculate LLM model size: {e}")

    return True


def check_stt_model():
    """بررسی وجود و آمادگی مدل STT (Whisper)"""
    print("🎙️  Checking STT model configuration...")

    # مسیر مدل STT
    stt_model_path_env = os.getenv("WHISPER_MODEL_PATH", "/app/models/stt")
    local_stt_path = stt_model_path_env.replace("/app/", "data/")

    stt_model_path = Path(local_stt_path)
    whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "medium")

    if stt_model_path.exists():
        # بررسی فایل‌های Whisper موجود
        whisper_files = list(stt_model_path.glob("*.pt"))
        if whisper_files:
            print(f"✅ Found {len(whisper_files)} Whisper model files")
            for file in whisper_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   📁 {file.name}: {size_mb:.1f} MB")
        else:
            print("📥 STT model directory exists but no .pt files found")
            print("   Whisper models will be downloaded automatically on first use")
    else:
        print("📥 STT model directory not found")
        print(
            f"   Will be created and Whisper '{whisper_model_size}' model will be downloaded"
        )

    print(f"🔧 Configured Whisper model size: {whisper_model_size}")

    # تخمین حجم دانلود
    model_sizes = {"tiny": 39, "base": 74, "small": 244, "medium": 769, "large": 1550}

    estimated_size = model_sizes.get(whisper_model_size, 500)
    print(f"📊 Estimated download size: ~{estimated_size}MB")

    # بررسی تنظیمات STT
    max_file_size = os.getenv("MAX_FILE_SIZE_MB", "25")
    supported_langs = os.getenv("SUPPORTED_LANGUAGES", "fa,en")
    print(f"📊 Max file size: {max_file_size}MB")
    print(f"🌐 Supported languages: {supported_langs}")

    return True


def create_directories():
    """ایجاد دایرکتوری‌های مورد نیاز"""
    print("📁 Creating required directories...")

    directories = [
        "data/models/llm",
        "data/models/stt",
        "data/vectors",
        "data/cache",
        "data/logs",
        "data/uploads",
        "monitoring/grafana/dashboards",
        "monitoring/grafana/provisioning/datasources",
        "services/api-gateway/middleware",
        "services/api-gateway/routes",
        "services/test-service",
        "services/llm-service/models",
        "services/llm-service/utils",
        "services/audio-service/stt",
        "services/gpu-coordinator",
        "shared/database",
        "shared/models",
        "shared/utils",
    ]

    created_count = 0
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            created_count += 1
        except Exception as e:
            print(f"⚠️  Could not create {directory}: {e}")

    print(f"✅ Created {created_count}/{len(directories)} directories")


def setup_environment():
    """تنظیم و validation environment variables"""
    print("🔧 Validating environment configuration...")

    # بررسی کلیدهای مهم environment
    important_keys = {
        "MODEL_PATH": "/app/models/gpt2-fa",
        "LLM_SERVICE_PORT": "8002",
        "STT_SERVICE_PORT": "8003",
        "GPU_COORDINATOR_PORT": "8080",
        "WHISPER_MODEL_SIZE": "medium",
        "CUDA_VISIBLE_DEVICES": "0",
        "MAX_MEMORY_MB": "6144",
    }

    print("🔍 Environment variables status:")
    all_set = True

    for key, default_value in important_keys.items():
        value = os.getenv(key)
        if value:
            print(f"   ✅ {key}: {value}")
        else:
            print(f"   ⚠️  {key}: not set, using default ({default_value})")
            os.environ[key] = default_value
            all_set = False

    if all_set:
        print("✅ All environment variables properly configured")
    else:
        print("⚠️  Some environment variables were missing - defaults applied")

    return True


def check_system_resources():
    """بررسی منابع سیستم"""
    print("💾 Checking system resources...")

    try:
        import psutil

        # بررسی RAM
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)

        print(f"🐏 Total RAM: {memory_gb:.1f} GB")
        print(f"🐏 Available RAM: {available_gb:.1f} GB")

        if memory_gb < 8:
            print("⚠️  Warning: Less than 8GB RAM detected")
            print("   Both LLM and STT services will compete for memory")
            print("   💡 Consider using CPU-only mode or lighter models")
        elif memory_gb >= 16:
            print("✅ Excellent RAM for both LLM and STT processing")
        else:
            print("✅ Adequate RAM - GPU Coordinator will manage resources")

        # بررسی CPU
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
        print(f"🔧 CPU cores: {cpu_count} physical, {cpu_logical} logical")

        if cpu_count < 4:
            print("⚠️  Consider using GPU acceleration for better performance")

        # بررسی فضای دیسک
        try:
            current_dir = os.path.abspath(".")
            disk = psutil.disk_usage(current_dir)
            free_gb = disk.free / (1024**3)
            total_gb = disk.total / (1024**3)

            print(f"💿 Disk space: {free_gb:.1f} GB free of {total_gb:.1f} GB total")

            if free_gb < 10:
                print("⚠️  Warning: Low disk space")
                print("   Whisper models and LLM models require significant space")
                print(
                    "   💡 Consider freeing up at least 10GB for models and containers"
                )
            elif free_gb < 5:
                print("❌ Critical: Very low disk space")
                print("   Setup may fail due to insufficient space")
                return False

        except Exception as disk_error:
            print(f"⚠️  Could not check disk space: {disk_error}")

    except ImportError:
        print("⚠️  psutil not available - skipping detailed resource check")
        print("   💡 Install with: pip install psutil")
    except Exception as e:
        print(f"⚠️  Error checking system resources: {e}")

    return True


def wait_for_services():
    """انتظار برای آماده شدن سرویس‌ها"""
    print("⏳ Waiting for services to be ready...")

    services = {
        "GPU Coordinator": ("http://localhost:8080/health", 30),
        "API Gateway": ("http://localhost:8000/health", 30),
        "LLM Service": ("http://localhost:8002/health", 90),
        "STT Service": ("http://localhost:8003/health", 90),
        "Test Service": ("http://localhost:8001/health", 30),
        "Prometheus": ("http://localhost:9090/-/ready", 30),
        "Grafana": ("http://localhost:3000/api/health", 30),
    }

    ready_services = []
    failed_services = []

    for service_name, (url, max_attempts) in services.items():
        print(f"🔍 Checking {service_name}...")

        service_ready = False
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {service_name} is ready")
                    ready_services.append(service_name)
                    service_ready = True
                    break
            except requests.exceptions.RequestException:
                pass

            if attempt < max_attempts - 1:
                if service_name in ["LLM Service", "STT Service"] and attempt % 15 == 0:
                    print(f"   💭 Loading AI model... ({attempt}/{max_attempts})")
                elif attempt % 10 == 0:
                    print(f"   ⏳ Waiting... ({attempt}/{max_attempts})")
                time.sleep(2)

        if not service_ready:
            print(f"⚠️  {service_name} not ready after {max_attempts} attempts")
            failed_services.append(service_name)

    print(f"\n📊 Service Status Summary:")
    print(f"   ✅ Ready: {len(ready_services)} services")
    print(f"   ⚠️  Not ready: {len(failed_services)} services")

    if ready_services:
        print(f"   Ready services: {', '.join(ready_services)}")

    if failed_services:
        print(f"   Failed services: {', '.join(failed_services)}")
        print("   💡 Check logs with: docker compose logs [service-name]")


def test_gpu_coordinator():
    """تست GPU Coordinator"""
    print("🔄 Testing GPU Coordinator...")

    try:
        # تست health endpoint
        response = requests.get("http://localhost:8080/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ GPU Coordinator health check passed")
            print(f"   🎮 GPU Available: {health_data.get('gpu_available', False)}")
        else:
            print(f"⚠️  GPU Coordinator health check returned: {response.status_code}")
            return False

        # تست status endpoint
        response = requests.get("http://localhost:8080/status", timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ GPU Status retrieved")
            print(
                f"   📊 Available Memory: {status_data.get('available_memory_mb', 'Unknown')} MB"
            )
            print(f"   🔧 Device Count: {status_data.get('device_count', 'Unknown')}")
        else:
            print("⚠️  Could not retrieve GPU status")

        # تست queue status
        response = requests.get("http://localhost:8080/queue", timeout=10)
        if response.status_code == 200:
            queue_data = response.json()
            print("✅ Queue status retrieved")
            print(
                f"   📋 Pending requests: {queue_data.get('pending_requests', {}).get('total', 0)}"
            )

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ GPU Coordinator test failed: {e}")
        return False


def test_llm_service():
    """تست LLM Service"""
    print("🤖 Testing LLM Service...")

    try:
        # تست health endpoint
        response = requests.get("http://localhost:8002/health", timeout=10)
        if response.status_code == 200:
            print("✅ LLM Health check passed")
        else:
            print(f"⚠️  LLM Health check returned: {response.status_code}")
            return False

        # تست model info endpoint
        response = requests.get("http://localhost:8002/model/info", timeout=10)
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ LLM Model loaded: {model_info.get('model_name', 'Unknown')}")
            print(f"   📊 Max length: {model_info.get('max_length', 'Unknown')}")
            print(f"   🎮 Device: {model_info.get('device', 'Unknown')}")
        else:
            print("⚠️  Could not retrieve LLM model info")

        # تست ساده تولید متن
        test_payload = {"text": "سلام", "max_length": 50, "temperature": 0.7}

        print("🔬 Testing LLM text generation...")
        response = requests.post(
            "http://localhost:8002/generate", json=test_payload, timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("generated_text", "")[:100]
            print(f"✅ LLM text generation test passed")
            print(f"   💬 Sample output: {generated_text}...")
        else:
            print(f"⚠️  LLM text generation test failed: {response.status_code}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ LLM Service test failed: {e}")
        return False


def test_stt_service():
    """تست STT Service"""
    print("🎙️  Testing STT Service...")

    try:
        # تست health endpoint
        response = requests.get("http://localhost:8003/health", timeout=10)
        if response.status_code == 200:
            print("✅ STT Health check passed")
        else:
            print(f"⚠️  STT Health check returned: {response.status_code}")
            return False

        # تست model info endpoint
        response = requests.get("http://localhost:8003/model/info", timeout=10)
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ STT Model info: {model_info.get('model_name', 'Unknown')}")
            print(
                f"   🌐 Supported languages: {model_info.get('supported_languages', 'Unknown')}"
            )
            print(f"   🎮 Device: {model_info.get('device', 'Unknown')}")
        else:
            print("⚠️  Could not retrieve STT model info")

        # تست supported formats
        response = requests.get("http://localhost:8003/formats", timeout=10)
        if response.status_code == 200:
            formats = response.json()
            supported_formats = formats.get("supported_formats", [])
            print(f"✅ Supported audio formats: {supported_formats}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ STT Service test failed: {e}")
        return False


def test_endpoints():
    """تست endpoint های اصلی"""
    print("🧪 Testing main endpoints...")

    endpoints = [
        ("Root", "http://localhost/"),
        ("API Info", "http://localhost/api/info"),
        ("Health Check", "http://localhost/health"),
        ("Test Service", "http://localhost/test/ping"),
        ("GPU Coordinator via Gateway", "http://localhost/api/gpu/status"),
        ("LLM via Gateway", "http://localhost/api/llm/health"),
        ("STT via Gateway", "http://localhost/api/stt/health"),
    ]

    working_endpoints = []
    failed_endpoints = []

    for name, url in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: OK")
                working_endpoints.append(name)
            else:
                print(f"⚠️  {name}: Status {response.status_code}")
                failed_endpoints.append(name)
        except requests.exceptions.RequestException as e:
            print(f"❌ {name}: Failed - {type(e).__name__}")
            failed_endpoints.append(name)

    print(f"\n📊 Endpoint Test Summary:")
    print(f"   ✅ Working: {len(working_endpoints)}/{len(endpoints)}")
    if failed_endpoints:
        print(f"   ❌ Failed: {', '.join(failed_endpoints)}")


def show_urls():
    """نمایش URL های مهم"""
    print("\n🌐 Available URLs:")
    print("─" * 60)
    print("🏠 Main Platform:     http://localhost")
    print("🔧 API Gateway:       http://localhost:8000")
    print("🔄 GPU Coordinator:   http://localhost:8080")
    print("🤖 LLM Service:       http://localhost:8002")
    print("🎙️  STT Service:       http://localhost:8003")
    print("🧪 Test Service:      http://localhost:8001")
    print("📊 Prometheus:        http://localhost:9090")
    print("📈 Grafana:          http://localhost:3000")
    print("   └─ Username: admin")
    print("   └─ Password: admin")
    print("─" * 60)


def show_ai_examples():
    """نمایش نمونه‌های استفاده از AI Services"""
    print("\n🤖 AI Services Examples:")
    print("─" * 60)
    print("🔄 GPU Status:")
    print("   curl http://localhost:8080/status")
    print("")
    print("📋 LLM Model Info:")
    print("   curl http://localhost:8002/model/info")
    print("")
    print("💬 LLM Text Generation:")
    print("   curl -X POST http://localhost:8002/generate \\")
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"text": "سلام", "max_length": 100}\'')
    print("")
    print("🎙️  STT Model Info:")
    print("   curl http://localhost:8003/model/info")
    print("")
    print("🔊 STT Audio Transcription:")
    print("   curl -X POST http://localhost:8003/transcribe \\")
    print('        -F "audio=@audio_file.wav" \\')
    print('        -F "language=fa"')
    print("")
    print("🔗 Via API Gateway:")
    print("   curl -X POST http://localhost/api/llm/generate \\")
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"text": "سلام", "max_length": 100}\'')
    print("─" * 60)


def show_troubleshooting():
    """نمایش راهنمای troubleshooting"""
    print("\n🔧 Troubleshooting Guide:")
    print("─" * 60)
    print("📊 Check container status:")
    print("   docker compose ps")
    print("")
    print("📋 View logs:")
    print("   docker compose logs -f [service-name]")
    print("   docker compose logs -f llm-service")
    print("   docker compose logs -f stt-service")
    print("   docker compose logs -f gpu-coordinator")
    print("")
    print("🔄 Restart specific service:")
    print("   docker compose restart [service-name]")
    print("")
    print("🛠️  Rebuild and restart:")
    print("   docker compose down")
    print("   docker compose build [service-name] --no-cache")
    print("   docker compose up -d")
    print("")
    print("💾 Check resources:")
    print("   docker stats")
    print("   nvidia-smi  # for GPU usage")
    print("")
    print("🧹 Clean up (if needed):")
    print("   docker compose down -v")
    print("   docker system prune -f")
    print("")
    print("🪟 Windows specific issues:")
    print("   • Run PowerShell/CMD as Administrator")
    print("   • Ensure Docker Desktop is running")
    print("   • Check Windows firewall settings")
    print("   • Restart Docker Desktop if needed")
    print("─" * 60)


def show_next_steps():
    """نمایش مراحل بعدی"""
    print("\n🎯 Next Steps:")
    print("─" * 60)
    print("1. Test AI capabilities:")
    print("   • LLM: curl http://localhost:8002/model/info")
    print("   • STT: curl http://localhost:8003/model/info")
    print("   • GPU: curl http://localhost:8080/status")
    print("")
    print("2. Monitor system:")
    print("   • Container health: docker compose ps")
    print("   • Resource usage: docker stats")
    print("   • GPU usage: nvidia-smi")
    print("")
    print("3. Scale and optimize:")
    print("   • Adjust memory limits in .env")
    print("   • Monitor GPU sharing efficiency")
    print("   • Add more AI models as needed")
    print("")
    print("4. Production preparation:")
    print("   • Change default passwords")
    print("   • Configure SSL/TLS")
    print("   • Set up proper monitoring")
    print("   • Implement backup strategies")
    print("")
    print("5. Windows optimization:")
    print("   • Consider WSL2 for better performance")
    print("   • Monitor Windows resource usage")
    print("   • Keep Docker Desktop updated")
    print("─" * 60)


def create_windows_env_file():
    """ایجاد فایل .env سازگار با Windows"""
    print("📝 Creating Windows-compatible .env file...")

    env_content = """# Agentic AI Platform Configuration - Windows Compatible
# GPU and Memory Settings
CUDA_VISIBLE_DEVICES=0
MAX_MEMORY_MB=6144
GPU_MEMORY_FRACTION=0.8

# Model Paths (Windows compatible)
MODEL_PATH=/app/models/gpt2-fa
WHISPER_MODEL_PATH=/app/models/stt
WHISPER_MODEL_SIZE=medium

# Service Ports
API_GATEWAY_PORT=8000
LLM_SERVICE_PORT=8002
STT_SERVICE_PORT=8003
GPU_COORDINATOR_PORT=8080
TEST_SERVICE_PORT=8001

# Database Configuration
POSTGRES_DB=agentic_platform
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres123
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis123

# STT Configuration
MAX_FILE_SIZE_MB=25
SUPPORTED_LANGUAGES=fa,en
AUDIO_TEMP_DIR=/tmp/audio

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Windows Specific
DOCKER_BUILDKIT=1
PYTHONIOENCODING=utf-8
COMPOSE_CONVERT_WINDOWS_PATHS=1

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
"""

    env_file = Path(".env")
    if not env_file.exists():
        try:
            with open(env_file, "w", encoding="utf-8") as f:
                f.write(env_content)
            print("✅ Created .env file with Windows-compatible settings")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("✅ .env file already exists")
        return True


def main():
    """تابع اصلی"""
    # تنظیم encoding برای Windows
    set_windows_encoding()

    print_banner()

    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        create_windows_env_file()

    # Load environment variables
    if not load_environment():
        print("❌ Environment setup failed")
        sys.exit(1)

    # بررسی requirements
    if not check_requirements():
        print("❌ Please install required tools first")
        print("💡 Make sure Docker Desktop is running")
        sys.exit(1)

    # بررسی پشتیبانی GPU
    gpu_available = check_gpu_support()

    # بررسی منابع سیستم
    if not check_system_resources():
        print("❌ Insufficient system resources")
        print("💡 Consider closing other applications to free up resources")
        response = input("Continue anyway? (y/N): ").lower().strip()
        if response != "y":
            sys.exit(1)

    # بررسی مدل‌های AI
    if not check_llm_model():
        print("❌ Please ensure LLM model is properly installed")
        print("💡 Check the data/models/llm/gpt2-fa/ directory")
        response = input("Continue without LLM model? (y/N): ").lower().strip()
        if response != "y":
            sys.exit(1)

    check_stt_model()  # STT مدل اختیاری است

    # ایجاد دایرکتوری‌ها
    create_directories()

    # تنظیم environment
    if not setup_environment():
        sys.exit(1)

    print(f"\n🖥️  Platform: {platform.system()} {platform.release()}")
    if platform.system() == "Windows":
        print("🪟 Windows-specific optimizations applied")

    # ساخت سرویس‌ها
    if not build_services():
        print("\n💡 Build failed. Troubleshooting suggestions:")
        print("   1. Check your internet connection")
        print("   2. Restart Docker Desktop")
        print("   3. Run as Administrator")
        print("   4. Clear Docker cache: docker system prune -f")
        print("   5. Manual build: docker compose build --no-cache")

        response = input("\nContinue with partial setup? (y/N): ").lower().strip()
        if response != "y":
            print("💡 Fix the build issues and run the script again")
            sys.exit(1)

    # راه‌اندازی سرویس‌ها
    if not start_services():
        print("❌ Service startup failed")
        print("💡 Check Docker Desktop and try again")
        print("💡 Run: docker compose logs to see detailed errors")

    # انتظار برای آماده شدن
    wait_for_services()

    # تست سرویس‌ها
    print("\n🧪 Running service tests...")
    test_gpu_coordinator()
    test_llm_service()
    test_stt_service()

    # تست endpoint ها
    test_endpoints()

    # نمایش اطلاعات
    show_urls()
    show_ai_examples()
    show_troubleshooting()
    show_next_steps()

    print("\n🎉 Setup completed!")
    print("Your Agentic AI Platform with GPU Sharing is running!")
    print("🤖 LLM Service: Ready for text generation")
    print("🎙️  STT Service: Ready for speech recognition")
    print("🔄 GPU Coordinator: Managing resource sharing")

    # Final status check
    try:
        success, ps_output = run_docker_command_safe(
            ["docker", "compose", "ps"], timeout=30
        )

        if success:
            running_containers = ps_output.count("Up")
            total_lines = len([line for line in ps_output.split("\n") if line.strip()])
            total_containers = max(0, total_lines - 1)  # exclude header

            if running_containers > 0:
                print(
                    f"\n📊 Final Status: {running_containers}/{total_containers} containers running"
                )

            # Show any failed containers
            if "Exit" in ps_output or "Restarting" in ps_output:
                print("⚠️  Some containers may have issues:")
                print("   Run: docker compose logs [service-name] for details")

    except Exception:
        print("⚠️  Could not get final container status")

    print("\n💡 For Windows users:")
    print("   • Keep Docker Desktop running")
    print("   • Monitor resource usage in Task Manager")
    print("   • Consider WSL2 for better Docker performance")


if __name__ == "__main__":
    main()
