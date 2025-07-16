#!/usr/bin/env python3
"""
Agentic AI Platform Setup Script
راه‌اندازی اولیه پلتفرم با LLM Service
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def print_banner():
    """نمایش بنر شروع"""
    banner = """
    ╔══════════════════════════════════════╗
    ║        Agentic AI Platform           ║
    ║    Setup & Deployment with LLM       ║
    ╚══════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """بررسی وجود Docker و Docker Compose"""
    print("🔍 Checking requirements...")
    
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not installed or not in PATH")
        return False
    
    try:
        subprocess.run(["docker", "compose", "version"], check=True, capture_output=True)
        print("✅ Docker Compose is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker Compose is not installed or not in PATH")
        return False
    
    return True

def check_gpu_support():
    """بررسی پشتیبانی GPU"""
    print("🎮 Checking GPU support...")
    
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print("💡 LLM Service will use GPU acceleration")
            return True
        else:
            print("⚠️  No NVIDIA GPU detected - using CPU")
            return False
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - using CPU")
        return False

def check_llm_model():
    """بررسی وجود مدل LLM"""
    print("🤖 Checking LLM model...")
    
    model_path = Path("data/models/llm/gpt2-fa")
    if not model_path.exists():
        print("❌ LLM model directory not found!")
        print(f"   Expected path: {model_path.absolute()}")
        print("   Please ensure the GPT2-FA model is placed in the correct directory")
        return False
    
    # بررسی فایل‌های مورد نیاز مدل
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "vocab.json"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = model_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"⚠️  Some model files might be missing: {', '.join(missing_files)}")
        print("   The model might still work, but please verify completeness")
    else:
        print("✅ LLM model files found")
    
    # بررسی حجم مدل
    try:
        total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"📦 Model size: {size_mb:.1f} MB")
        
        if size_mb < 10:
            print("⚠️  Model size seems small - please verify model integrity")
        
    except Exception as e:
        print(f"⚠️  Could not calculate model size: {e}")
    
    return True

def create_directories():
    """ایجاد دایرکتوری‌های مورد نیاز"""
    print("📁 Creating required directories...")
    
    directories = [
        "data/models/llm",
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
        "shared/database",
        "shared/models",
        "shared/utils"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def setup_environment():
    """تنظیم فایل environment"""
    print("🔧 Setting up environment...")
    
    if not os.path.exists(".env"):
        print("❌ .env file not found! Please create it first.")
        return False
    
    print("✅ Environment file exists")
    return True

def check_system_resources():
    """بررسی منابع سیستم"""
    print("💾 Checking system resources...")
    
    try:
        import psutil
        
        # بررسی RAM
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"🐏 Available RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️  Warning: Less than 4GB RAM detected")
            print("   LLM Service might face memory issues")
        elif memory_gb >= 8:
            print("✅ Sufficient RAM for LLM processing")
        else:
            print("⚠️  Moderate RAM - consider closing other applications")
        
        # بررسی CPU
        cpu_count = psutil.cpu_count()
        print(f"🔧 CPU cores: {cpu_count}")
        
        if cpu_count < 4:
            print("⚠️  Warning: Less than 4 CPU cores detected")
            print("   LLM processing might be slow")
        
        # بررسی فضای دیسک (با handling برای ویندوز)
        try:
            # در ویندوز ممکن است نیاز به مسیر مطلق باشد
            current_dir = os.path.abspath('.')
            disk = psutil.disk_usage(current_dir)
            free_gb = disk.free / (1024**3)
            print(f"💿 Free disk space: {free_gb:.1f} GB")
            
            if free_gb < 5:
                print("⚠️  Warning: Low disk space")
                print("   Consider freeing up space for model caching")
        except Exception as disk_error:
            print(f"⚠️  Could not check disk space: {disk_error}")
            print("   Please ensure sufficient disk space manually")
            
    except ImportError:
        print("⚠️  psutil not available - skipping resource check")
        print("   Install with: pip install psutil")
    except Exception as e:
        print(f"⚠️  Error checking system resources: {e}")
        print("   Continuing with setup...")

def build_services():
    """ساخت سرویس‌ها"""
    print("🏗️  Building services...")
    
    # تنظیم encoding برای ویندوز
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # ابتدا LLM Service را build کنیم (ممکن است زمان بیشتری ببرد)
    print("🤖 Building LLM Service...")
    try:
        # برای نمایش لایو لاگ‌ها
        process = subprocess.Popen(
            ["docker", "compose", "build", "llm-service", "--progress", "plain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        # نمایش خروجی در real-time
        for line in process.stdout:
            if "Step" in line or "ERROR" in line or "FAILED" in line or "✓" in line:
                print(f"   {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            print("✅ LLM Service built successfully")
        else:
            print("❌ LLM Service build failed")
            print("💡 Trying alternative build method...")
            
            # تلاش مجدد با روش ساده‌تر
            result = subprocess.run(
                ["docker", "compose", "build", "llm-service"],
                check=True,
                env=env
            )
            print("✅ LLM Service built successfully (alternative method)")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ LLM Service build failed")
        print("🔍 Checking for common issues...")
        
        # بررسی فایل Dockerfile
        dockerfile_path = Path("services/llm-service/Dockerfile")
        if not dockerfile_path.exists():
            print("❌ Dockerfile not found for LLM Service")
            return False
        
        # بررسی requirements.txt
        req_path = Path("services/llm-service/requirements.txt")
        if not req_path.exists():
            print("❌ requirements.txt not found for LLM Service")
            return False
            
        print("💡 Try building manually with: docker compose build llm-service --no-cache")
        return False
    
    # سپس بقیه سرویس‌ها را build کنیم
    print("🔧 Building other services...")
    try:
        result = subprocess.run(
            ["docker", "compose", "build"],
            check=True,
            env=env
        )
        print("✅ All services built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed")
        print("💡 Try building manually with: docker compose build --no-cache")
        return False

def start_services():
    """راه‌اندازی سرویس‌ها"""
    print("🚀 Starting services...")
    
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            check=True
        )
        print("✅ Services started")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start services: {e}")
        return False

def wait_for_services():
    """انتظار برای آماده شدن سرویس‌ها"""
    print("⏳ Waiting for services to be ready...")
    
    services = {
        "API Gateway": "http://localhost:8000/health",
        "LLM Service": "http://localhost:8002/health", 
        "Test Service": "http://localhost:8001/health",
        "Prometheus": "http://localhost:9090/-/ready",
        "Grafana": "http://localhost:3000/api/health"
    }
    
    for service_name, url in services.items():
        print(f"🔍 Checking {service_name}...")
        
        # برای LLM Service زمان بیشتری در نظر می‌گیریم
        max_attempts = 60 if service_name == "LLM Service" else 30
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {service_name} is ready")
                    break
            except requests.exceptions.RequestException:
                pass
            
            if attempt < max_attempts - 1:
                if service_name == "LLM Service" and attempt % 10 == 0:
                    print(f"   💭 Still loading model... ({attempt}/{max_attempts})")
                time.sleep(2)
            else:
                print(f"⚠️  {service_name} might not be ready (timeout)")

def test_llm_service():
    """تست LLM Service"""
    print("🧪 Testing LLM Service...")
    
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
            print(f"✅ Model loaded: {model_info.get('model_name', 'Unknown')}")
            print(f"   📊 Max length: {model_info.get('max_length', 'Unknown')}")
        else:
            print("⚠️  Could not retrieve model info")
        
        # تست ساده تولید متن
        test_payload = {
            "text": "سلام",
            "max_length": 50,
            "temperature": 0.7
        }
        
        print("🔬 Testing text generation...")
        response = requests.post(
            "http://localhost:8002/generate",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('generated_text', '')[:100]  # نمایش 100 کاراکتر اول
            print(f"✅ Text generation test passed")
            print(f"   💬 Sample output: {generated_text}...")
        else:
            print(f"⚠️  Text generation test failed: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text[:200]}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ LLM Service test failed: {e}")
        return False

def test_endpoints():
    """تست endpoint های اصلی"""
    print("🧪 Testing endpoints...")
    
    endpoints = [
        ("Root", "http://localhost/"),
        ("API Info", "http://localhost/api/info"),
        ("Health Check", "http://localhost/health"),
        ("Test Service", "http://localhost/test/ping"),
        ("Connection Test", "http://localhost/api/test-connection"),
        ("LLM via Gateway", "http://localhost/api/llm/health")
    ]
    
    for name, url in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: OK")
            else:
                print(f"⚠️  {name}: Status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ {name}: Failed - {e}")

def show_urls():
    """نمایش URL های مهم"""
    print("\n🌐 Available URLs:")
    print("─" * 50)
    print("🏠 Main Platform:     http://localhost")
    print("🔧 API Gateway:       http://localhost:8000")
    print("🤖 LLM Service:       http://localhost:8002")
    print("🧪 Test Service:      http://localhost:8001") 
    print("📊 Prometheus:        http://localhost:9090")
    print("📈 Grafana:          http://localhost:3000")
    print("   └─ Username: admin")
    print("   └─ Password: admin")
    print("─" * 50)

def show_llm_examples():
    """نمایش نمونه‌های استفاده از LLM"""
    print("\n🤖 LLM Service Examples:")
    print("─" * 50)
    print("📋 Model Info:")
    print("   curl http://localhost:8002/model/info")
    print("")
    print("💬 Text Generation:")
    print('   curl -X POST http://localhost:8002/generate \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"text": "سلام", "max_length": 100}\'')
    print("")
    print("🔗 Via API Gateway:")
    print('   curl -X POST http://localhost/api/llm/generate \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"text": "سلام", "max_length": 100}\'')
    print("─" * 50)

def show_next_steps():
    """نمایش مراحل بعدی"""
    print("\n🎯 Next Steps:")
    print("─" * 50)
    print("1. Check all services: docker compose ps")
    print("2. View logs: docker compose logs -f llm-service")
    print("3. Test LLM: curl http://localhost:8002/model/info")
    print("4. Monitor resources: docker stats")
    print("5. Stop services: docker compose down")
    print("6. Add more AI capabilities!")
    print("─" * 50)

def debug_docker_issues():
    """بررسی و Debug مشکلات Docker"""
    print("🔍 Debugging Docker issues...")
    
    # بررسی وضعیت Docker
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker daemon is not running")
            print("   Please start Docker Desktop")
            return False
    except Exception:
        print("❌ Cannot communicate with Docker")
        return False
    
    # بررسی فایل‌های مورد نیاز
    required_files = [
        "docker-compose.yml",
        "services/llm-service/Dockerfile",
        "services/llm-service/requirements.txt",
        "services/llm-service/main.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ Docker and required files are OK")
    return True

def main():
    """تابع اصلی"""
    print_banner()
    
    # بررسی requirements
    if not check_requirements():
        print("❌ Please install required tools first")
        sys.exit(1)
    
    # بررسی پشتیبانی GPU
    check_gpu_support()
    
    # بررسی منابع سیستم
    check_system_resources()
    
    # بررسی مدل LLM
    if not check_llm_model():
        print("❌ Please ensure LLM model is properly installed")
        sys.exit(1)
    
    # ایجاد دایرکتوری‌ها
    create_directories()
    
    # تنظیم environment
    if not setup_environment():
        sys.exit(1)
    
    # Debug Docker issues
    if not debug_docker_issues():
        print("❌ Please fix Docker issues first")
        sys.exit(1)
    
    # ساخت سرویس‌ها
    if not build_services():
        print("\n💡 Build failed. You can try manual build:")
        print("   docker compose build llm-service --no-cache")
        print("   docker compose build --no-cache")
        sys.exit(1)
    
    # راه‌اندازی سرویس‌ها
    if not start_services():
        sys.exit(1)
    
    # انتظار برای آماده شدن
    wait_for_services()
    
    # تست LLM Service
    test_llm_service()
    
    # تست endpoint ها
    test_endpoints()
    
    # نمایش اطلاعات
    show_urls()
    show_llm_examples()
    show_next_steps()
    
    print("\n🎉 Setup completed successfully!")
    print("Your Agentic AI Platform with LLM Service is running!")

if __name__ == "__main__":
    main()