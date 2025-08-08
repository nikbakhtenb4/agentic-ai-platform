# # scripts/debug_docker
# # python debug_docker.py
# """
# Docker Build Debug Script
# اسکریپت تشخیص و رفع مشکلات Build در پروژه Agentic AI
# """

# import os
# import sys
# import subprocess
# import json
# from pathlib import Path


# def print_banner():
#     """نمایش بنر"""
#     banner = """
#     ╔══════════════════════════════════════╗
#     ║       Docker Build Debugger          ║
#     ║    Find & Fix Build Issues           ║
#     ╚══════════════════════════════════════╝
#     """
#     print(banner)


# def check_docker_status():
#     """بررسی وضعیت Docker"""
#     print("🐳 Checking Docker status...")

#     try:
#         # بررسی Docker daemon
#         result = subprocess.run(["docker", "info"], capture_output=True, text=True)
#         if result.returncode == 0:
#             print("✅ Docker daemon is running")

#             # نمایش اطلاعات مهم Docker
#             lines = result.stdout.split("\n")
#             for line in lines:
#                 if "Server Version:" in line:
#                     print(f"   📋 {line.strip()}")
#                 elif "Storage Driver:" in line:
#                     print(f"   💾 {line.strip()}")
#                 elif "Logging Driver:" in line:
#                     print(f"   📝 {line.strip()}")
#                 elif "Runtimes:" in line:
#                     print(f"   🏃 {line.strip()}")

#             return True
#         else:
#             print("❌ Docker daemon is not running")
#             print("   💡 Start Docker Desktop and try again")
#             return False

#     except FileNotFoundError:
#         print("❌ Docker command not found")
#         print("   💡 Install Docker and add it to PATH")
#         return False
#     except Exception as e:
#         print(f"❌ Error checking Docker: {e}")
#         return False


# def check_compose_file():
#     """بررسی فایل docker-compose.yml"""
#     print("\n📄 Checking docker-compose.yml...")

#     compose_file = Path("docker-compose.yml")
#     if not compose_file.exists():
#         print("❌ docker-compose.yml not found")
#         return False

#     try:
#         # تست syntax
#         result = subprocess.run(
#             ["docker", "compose", "config", "--quiet"], capture_output=True, text=True
#         )

#         if result.returncode == 0:
#             print("✅ docker-compose.yml syntax is correct")

#             # نمایش services موجود
#             config_result = subprocess.run(
#                 ["docker", "compose", "config", "--services"],
#                 capture_output=True,
#                 text=True,
#             )

#             if config_result.returncode == 0:
#                 services = config_result.stdout.strip().split("\n")
#                 print(f"   📋 Found {len(services)} services:")
#                 for service in services:
#                     if service.strip():
#                         print(f"      • {service}")

#             return True
#         else:
#             print("❌ docker-compose.yml has syntax errors:")
#             print(f"   {result.stderr}")
#             return False

#     except Exception as e:
#         print(f"❌ Error checking compose file: {e}")
#         return False


# def check_service_files():
#     """بررسی فایل‌های مورد نیاز هر سرویس"""
#     print("\n📁 Checking service files...")

#     services_config = {
#         "gpu-coordinator": {
#             "dockerfile": "services/gpu-coordinator/Dockerfile",
#             "main": "services/gpu-coordinator/main.py",
#             "requirements": "services/gpu-coordinator/requirements.txt",
#         },
#         "llm-service": {
#             "dockerfile": "services/llm-service/Dockerfile",
#             "main": "services/llm-service/main.py",
#             "requirements": "services/llm-service/requirements.txt",
#         },
#         "stt-service": {
#             "dockerfile": "services/audio-service/stt/Dockerfile",
#             "main": "services/audio-service/stt/main.py",
#             "requirements": "services/audio-service/stt/requirements.txt",
#         },
#         "api-gateway": {
#             "dockerfile": "services/api-gateway/Dockerfile",
#             "main": "services/api-gateway/main.py",
#             "requirements": "services/api-gateway/requirements.txt",
#         },
#         "auth-service": {
#             "dockerfile": "services/auth-service/Dockerfile",
#             "main": "services/auth-service/main.py",
#             "requirements": "services/auth-service/requirements.txt",
#         },
#         "test-service": {
#             "dockerfile": "services/test-service/Dockerfile",
#             "main": "services/test-service/main.py",
#             "requirements": "services/test-service/requirements.txt",
#         },
#     }

#     issues_found = []

#     for service_name, files in services_config.items():
#         print(f"\n🔍 Checking {service_name}:")
#         service_issues = []

#         for file_type, file_path in files.items():
#             if Path(file_path).exists():
#                 print(f"   ✅ {file_type}: {file_path}")

#                 # بررسی محتوای فایل requirements.txt
#                 if file_type == "requirements" and Path(file_path).exists():
#                     try:
#                         with open(file_path, "r") as f:
#                             content = f.read().strip()
#                             if not content:
#                                 print(f"      ⚠️  Empty requirements.txt")
#                             else:
#                                 lines = content.split("\n")
#                                 print(f"      📦 {len(lines)} packages listed")
#                     except Exception as e:
#                         print(f"      ⚠️  Could not read requirements: {e}")

#             else:
#                 print(f"   ❌ {file_type}: {file_path} (MISSING)")
#                 service_issues.append(f"{file_type}: {file_path}")

#         if service_issues:
#             issues_found.append({"service": service_name, "issues": service_issues})

#     if issues_found:
#         print(f"\n❌ Found issues in {len(issues_found)} services:")
#         for issue in issues_found:
#             print(f"   • {issue['service']}: {', '.join(issue['issues'])}")
#         return False
#     else:
#         print("\n✅ All service files found")
#         return True


# def check_shared_directory():
#     """بررسی دایرکتوری shared"""
#     print("\n📂 Checking shared directory...")

#     shared_path = Path("shared")
#     if not shared_path.exists():
#         print("❌ shared/ directory not found")
#         print("   💡 Create with: mkdir -p shared/{config,database,models,utils}")
#         return False

#     # بررسی زیردایرکتوری‌های مهم
#     subdirs = ["config", "database", "models", "utils"]
#     missing_dirs = []

#     for subdir in subdirs:
#         subdir_path = shared_path / subdir
#         if subdir_path.exists():
#             print(f"   ✅ shared/{subdir}/")
#         else:
#             print(f"   ⚠️  shared/{subdir}/ (missing)")
#             missing_dirs.append(subdir)

#     if missing_dirs:
#         print(f"   💡 Create missing directories:")
#         for dir_name in missing_dirs:
#             print(f"      mkdir -p shared/{dir_name}")

#     return True


# def check_data_directory():
#     """بررسی دایرکتوری data و مدل‌ها"""
#     print("\n💾 Checking data directory and models...")

#     data_path = Path("data")
#     if not data_path.exists():
#         print("❌ data/ directory not found")
#         print("   💡 Create with: mkdir -p data/{models/{llm,stt},cache,logs,uploads}")
#         return False

#     # بررسی مدل LLM
#     llm_model_path = data_path / "models" / "llm" / "gpt2-fa"
#     if llm_model_path.exists():
#         print("   ✅ LLM model directory found")

#         # بررسی فایل‌های مدل
#         model_files = [
#             "config.json",
#             "pytorch_model.bin",
#             "tokenizer.json",
#             "vocab.json",
#         ]
#         for model_file in model_files:
#             if (llm_model_path / model_file).exists():
#                 print(f"      ✅ {model_file}")
#             else:
#                 print(f"      ⚠️  {model_file} (missing)")
#     else:
#         print("   ❌ LLM model directory not found")
#         print("      Expected: data/models/llm/gpt2-fa/")

#     # بررسی مدل STT
#     stt_model_path = data_path / "models" / "stt"
#     if stt_model_path.exists():
#         print("   ✅ STT model directory found")
#         whisper_files = list(stt_model_path.glob("*.pt"))
#         if whisper_files:
#             print(f"      📁 Found {len(whisper_files)} Whisper model files")
#         else:
#             print("      📥 No Whisper models found (will download on first use)")
#     else:
#         print("   ⚠️  STT model directory not found")
#         print("      Will be created automatically")

#     return True


# def check_environment_file():
#     """بررسی فایل .env"""
#     print("\n🔧 Checking .env file...")

#     env_file = Path(".env")
#     if not env_file.exists():
#         print("❌ .env file not found")
#         print("   💡 Create .env file with required variables")
#         return False

#     try:
#         with open(env_file, "r") as f:
#             content = f.read()

#         # متغیرهای مهم
#         important_vars = [
#             "MODEL_PATH",
#             "LLM_SERVICE_PORT",
#             "STT_SERVICE_PORT",
#             "GPU_COORDINATOR_PORT",
#             "REDIS_URL",
#             "DATABASE_URL",
#         ]

#         missing_vars = []
#         for var in important_vars:
#             if f"{var}=" in content:
#                 print(f"   ✅ {var}")
#             else:
#                 print(f"   ⚠️  {var} (not found)")
#                 missing_vars.append(var)

#         if missing_vars:
#             print(f"   💡 Add missing variables to .env:")
#             for var in missing_vars:
#                 print(f"      {var}=your_value_here")

#         return True

#     except Exception as e:
#         print(f"❌ Error reading .env file: {e}")
#         return False


# def test_individual_build(service_name):
#     """تست build یک سرویس خاص"""
#     print(f"\n🔨 Testing build for {service_name}...")

#     try:
#         # تلاش برای build
#         process = subprocess.Popen(
#             [
#                 "docker",
#                 "compose",
#                 "build",
#                 service_name,
#                 "--no-cache",
#                 "--progress",
#                 "plain",
#             ],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             universal_newlines=True,
#             encoding="utf-8",
#             errors="replace",
#         )

#         build_log = []
#         for line in process.stdout:
#             line = line.strip()
#             build_log.append(line)

#             # نمایش خطوط مهم
#             if any(
#                 keyword in line.lower()
#                 for keyword in ["error", "failed", "step", "downloading", "installing"]
#             ):
#                 print(f"   📝 {line}")

#         process.wait()

#         if process.returncode == 0:
#             print(f"✅ {service_name} build successful")
#             return True, build_log
#         else:
#             print(f"❌ {service_name} build failed")

#             # نمایش آخرین خطاها
#             error_lines = [line for line in build_log if "error" in line.lower()]
#             if error_lines:
#                 print("   🔍 Error details:")
#                 for error in error_lines[-3:]:  # آخرین 3 خطا
#                     print(f"      {error}")

#             return False, build_log

#     except Exception as e:
#         print(f"❌ Build test failed: {e}")
#         return False, []


# def analyze_build_logs(service_name, build_log):
#     """تجزیه و تحلیل لاگ‌های build"""
#     print(f"\n🔍 Analyzing build logs for {service_name}...")

#     common_issues = {
#         "permission denied": "Docker permission issue - try running as admin/sudo",
#         "no space left on device": "Disk space full - clean up Docker images",
#         "network timeout": "Internet connection issue - check network",
#         "package not found": "Python package issue - check requirements.txt",
#         "dockerfile not found": "Dockerfile path issue - check docker-compose.yml",
#         "context not found": "Build context issue - check file paths",
#         "pip install failed": "Python dependency issue - check package versions",
#         "cuda not found": "CUDA/GPU issue - check nvidia-docker setup",
#     }

#     issues_found = []
#     for line in build_log:
#         line_lower = line.lower()
#         for issue_key, solution in common_issues.items():
#             if issue_key in line_lower:
#                 issues_found.append(
#                     {"issue": issue_key, "solution": solution, "line": line.strip()}
#                 )

#     if issues_found:
#         print(f"   🎯 Found {len(issues_found)} potential issues:")
#         for issue in issues_found:
#             print(f"      • Issue: {issue['issue']}")
#             print(f"        Solution: {issue['solution']}")
#             print(f"        Log: {issue['line'][:100]}...")
#             print()
#     else:
#         print("   ✅ No common issues detected")

#     return issues_found


# def fix_common_issues():
#     """رفع مشکلات رایج"""
#     print("\n🔧 Fixing common issues...")

#     fixes_applied = []

#     # ایجاد دایرکتوری‌های مورد نیاز
#     required_dirs = [
#         "data/models/llm",
#         "data/models/stt",
#         "data/cache",
#         "data/logs",
#         "shared/config",
#         "shared/database",
#         "shared/models",
#         "shared/utils",
#     ]

#     for dir_path in required_dirs:
#         path = Path(dir_path)
#         if not path.exists():
#             try:
#                 path.mkdir(parents=True, exist_ok=True)
#                 print(f"   ✅ Created directory: {dir_path}")
#                 fixes_applied.append(f"Created {dir_path}")
#             except Exception as e:
#                 print(f"   ❌ Failed to create {dir_path}: {e}")

#     # ایجاد فایل __init__.py در shared
#     shared_init = Path("shared/__init__.py")
#     if not shared_init.exists():
#         try:
#             shared_init.touch()
#             print("   ✅ Created shared/__init__.py")
#             fixes_applied.append("Created shared/__init__.py")
#         except Exception as e:
#             print(f"   ❌ Failed to create shared/__init__.py: {e}")

#     # پاک‌سازی Docker cache
#     try:
#         subprocess.run(["docker", "system", "prune", "-f"], capture_output=True)
#         print("   ✅ Cleaned Docker system cache")
#         fixes_applied.append("Cleaned Docker cache")
#     except Exception as e:
#         print(f"   ⚠️  Could not clean Docker cache: {e}")

#     print(f"\n✅ Applied {len(fixes_applied)} fixes:")
#     for fix in fixes_applied:
#         print(f"   • {fix}")

#     return fixes_applied


# def generate_build_report():
#     """تولید گزارش کامل build"""
#     print("\n📊 Generating build report...")

#     services = [
#         "gpu-coordinator",
#         "llm-service",
#         "stt-service",
#         "api-gateway",
#         "auth-service",
#         "test-service",
#     ]

#     report = {
#         "timestamp": subprocess.run(
#             ["date"], capture_output=True, text=True
#         ).stdout.strip(),
#         "docker_info": {},
#         "services": {},
#     }

#     # اطلاعات Docker
#     try:
#         docker_info = subprocess.run(
#             ["docker", "info", "--format", "json"], capture_output=True, text=True
#         )
#         if docker_info.returncode == 0:
#             report["docker_info"] = json.loads(docker_info.stdout)
#     except:
#         pass

#     # تست build هر سرویس
#     for service in services:
#         print(f"   🔨 Testing {service}...")
#         success, build_log = test_individual_build(service)

#         issues = analyze_build_logs(service, build_log) if not success else []

#         report["services"][service] = {
#             "build_success": success,
#             "issues_count": len(issues),
#             "issues": issues,
#             "log_lines": len(build_log),
#         }

#     # ذخیره گزارش
#     try:
#         with open("build_report.json", "w") as f:
#             json.dump(report, f, indent=2)
#         print("   ✅ Build report saved to build_report.json")
#     except Exception as e:
#         print(f"   ⚠️  Could not save report: {e}")

#     # خلاصه گزارش
#     successful_builds = sum(
#         1 for s in report["services"].values() if s["build_success"]
#     )
#     total_services = len(report["services"])

#     print(
#         f"\n📋 Build Summary: {successful_builds}/{total_services} services built successfully"
#     )

#     for service_name, service_data in report["services"].items():
#         status = "✅" if service_data["build_success"] else "❌"
#         issues_text = (
#             f" ({service_data['issues_count']} issues)"
#             if service_data["issues_count"] > 0
#             else ""
#         )
#         print(f"   {status} {service_name}{issues_text}")

#     return report


# def show_manual_commands():
#     """نمایش دستورات manual برای build"""
#     print("\n🔧 Manual Build Commands:")
#     print("─" * 60)

#     services = [
#         ("gpu-coordinator", "services/gpu-coordinator/"),
#         ("llm-service", "services/llm-service/"),
#         ("stt-service", "services/audio-service/stt/"),
#         ("api-gateway", "services/api-gateway/"),
#         ("auth-service", "services/auth-service/"),
#         ("test-service", "services/test-service/"),
#     ]

#     print("🏗️  Individual service builds:")
#     for service_name, service_path in services:
#         print(f"   docker compose build {service_name} --no-cache")

#     print("\n🧹 Clean build (if issues persist):")
#     print("   docker system prune -a")
#     print("   docker compose build --no-cache")

#     print("\n🔍 Debug specific service:")
#     print("   docker compose build [service-name] --progress=plain --no-cache")

#     print("\n📋 Check service status:")
#     print("   docker compose config --services")
#     print("   docker compose ps")

#     print("\n📝 View logs:")
#     print("   docker compose logs [service-name]")
#     print("─" * 60)


# def interactive_debug():
#     """حالت تعاملی debug"""
#     print("\n🎯 Interactive Debug Mode")
#     print("─" * 40)

#     while True:
#         print("\nSelect an option:")
#         print("1. Check specific service files")
#         print("2. Test build specific service")
#         print("3. View docker-compose config")
#         print("4. Clean Docker system")
#         print("5. Fix common issues")
#         print("6. Generate full report")
#         print("7. Show manual commands")
#         print("8. Exit")

#         try:
#             choice = input("\nEnter choice (1-8): ").strip()

#             if choice == "1":
#                 service = input("Enter service name: ").strip()
#                 if service:
#                     # بررسی فایل‌های سرویس خاص
#                     print(f"\n🔍 Checking {service} files...")
#                     check_service_files()

#             elif choice == "2":
#                 service = input("Enter service name: ").strip()
#                 if service:
#                     test_individual_build(service)

#             elif choice == "3":
#                 print("\n📄 Docker Compose Configuration:")
#                 try:
#                     result = subprocess.run(
#                         ["docker", "compose", "config"], capture_output=True, text=True
#                     )
#                     if result.returncode == 0:
#                         print(
#                             result.stdout[:1000] + "..."
#                             if len(result.stdout) > 1000
#                             else result.stdout
#                         )
#                     else:
#                         print(f"❌ Error: {result.stderr}")
#                 except Exception as e:
#                     print(f"❌ Error: {e}")

#             elif choice == "4":
#                 print("\n🧹 Cleaning Docker system...")
#                 try:
#                     subprocess.run(
#                         ["docker", "system", "prune", "-a", "-f"], check=True
#                     )
#                     print("✅ Docker system cleaned")
#                 except Exception as e:
#                     print(f"❌ Error: {e}")

#             elif choice == "5":
#                 fix_common_issues()

#             elif choice == "6":
#                 generate_build_report()

#             elif choice == "7":
#                 show_manual_commands()

#             elif choice == "8":
#                 print("👋 Exiting debug mode")
#                 break

#             else:
#                 print("❌ Invalid choice")

#         except KeyboardInterrupt:
#             print("\n👋 Exiting debug mode")
#             break
#         except Exception as e:
#             print(f"❌ Error: {e}")


# def main():
#     """تابع اصلی"""
#     print_banner()

#     # بررسی‌های اولیه
#     if not check_docker_status():
#         sys.exit(1)

#     if not check_compose_file():
#         print("💡 Fix docker-compose.yml issues first")
#         sys.exit(1)

#     # بررسی‌های جزئی
#     check_service_files()
#     check_shared_directory()
#     check_data_directory()
#     check_environment_file()

#     # رفع مشکلات رایج
#     fix_common_issues()

#     print("\n" + "=" * 60)
#     print("🎯 BUILD TESTING PHASE")
#     print("=" * 60)

#     # تولید گزارش کامل
#     report = generate_build_report()

#     # نمایش دستورات manual
#     show_manual_commands()

#     # پیشنهاد حالت تعاملی
#     print("\n💡 Tips:")
#     print("   • Run individual builds first: docker compose build [service-name]")
#     print(
#         "   • Check logs if build fails: docker compose build [service] --progress=plain"
#     )
#     print("   • Clean Docker cache if needed: docker system prune -a")
#     print("   • Ensure all required files exist in their proper locations")

#     try:
#         response = input("\nRun interactive debug mode? (y/n): ").lower().strip()
#         if response in ["y", "yes"]:
#             interactive_debug()
#     except KeyboardInterrupt:
#         print("\n👋 Goodbye!")


# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n👋 Debug cancelled by user")
#         sys.exit(1)
#     except Exception as e:
#         print(f"\n❌ Unexpected error: {e}")
#         sys.exit(1)
