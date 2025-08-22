#!/usr/bin/env python3
"""
Enhanced Test LLM Service GPU Integration
Includes better error handling and diagnostics
"""

import asyncio
import httpx
import time
import os
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for better output formatting"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_section(title, icon="🔍"):
    """Print formatted section header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{icon} {title}{Colors.END}")
    print("=" * 50)


def print_success(message, icon="✅"):
    """Print success message"""
    print(f"   {Colors.GREEN}{icon} {message}{Colors.END}")


def print_error(message, icon="❌"):
    """Print error message"""
    print(f"   {Colors.RED}{icon} {message}{Colors.END}")


def print_warning(message, icon="⚠️"):
    """Print warning message"""
    print(f"   {Colors.YELLOW}{icon} {message}{Colors.END}")


def print_info(message, icon="ℹ️"):
    """Print info message"""
    print(f"   {Colors.BLUE}{icon} {message}{Colors.END}")


async def test_service_connectivity():
    """Test basic connectivity to both services"""
    print_section("Service Connectivity Test", "🌐")

    base_url_gpu = "http://localhost:8080"
    base_url_llm = "http://localhost:8002"

    results = {"gpu": False, "llm": False}

    async with httpx.AsyncClient(timeout=10) as client:
        # Test GPU Coordinator
        try:
            response = await client.get(f"{base_url_gpu}/health")
            if response.status_code == 200:
                data = response.json()
                print_success(f"GPU Coordinator: {data.get('status', 'Unknown')}")
                print_info(f"Total GPUs: {data.get('total_gpus', 0)}")
                results["gpu"] = True
            else:
                print_error(f"GPU Coordinator failed: HTTP {response.status_code}")
        except Exception as e:
            print_error(f"GPU Coordinator unreachable: {e}")

        # Test LLM Service
        try:
            response = await client.get(f"{base_url_llm}/health")
            if response.status_code == 200:
                data = response.json()
                print_success(f"LLM Service: {data.get('status', 'Unknown')}")
                print_info(f"Model Loaded: {data.get('model_loaded', False)}")
                print_info(f"GPU Available: {data.get('gpu_available', False)}")
                print_info(f"Uptime: {data.get('uptime', 0):.1f}s")
                results["llm"] = True
            else:
                print_error(f"LLM Service failed: HTTP {response.status_code}")
        except Exception as e:
            print_error(f"LLM Service unreachable: {e}")

    return results


async def test_gpu_coordination_status():
    """Test GPU coordination status endpoint"""
    print_section("GPU Coordination Status", "🔗")

    base_url_llm = "http://localhost:8002"

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            response = await client.get(f"{base_url_llm}/gpu/status")
            if response.status_code == 200:
                data = response.json()
                print_success("GPU coordination status retrieved")
                print_info(
                    f"Coordination Available: {data.get('coordination_available', False)}"
                )
                print_info(f"GPU Allocated: {data.get('gpu_allocated', False)}")
                if data.get("gpu_id"):
                    print_info(f"GPU ID: {data['gpu_id']}")
                print_info(f"Coordinator URL: {data.get('coordinator_url', 'N/A')}")
                print_info(f"Service Name: {data.get('service_name', 'N/A')}")
                return True
            else:
                print_error(f"GPU status check failed: HTTP {response.status_code}")
                if response.status_code == 404:
                    print_warning("Endpoint not found - check if route is implemented")
                return False
        except Exception as e:
            print_error(f"GPU status check error: {e}")
            return False


async def test_model_info():
    """Test model information endpoint"""
    print_section("Model Information", "📚")

    base_url_llm = "http://localhost:8002"

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            response = await client.get(f"{base_url_llm}/model/info")
            if response.status_code == 200:
                data = response.json()
                print_success("Model information retrieved")
                print_info(f"Model Name: {data.get('model_name', 'Unknown')}")
                print_info(f"Model Path: {data.get('model_path', 'Unknown')}")
                print_info(f"Model Loaded: {data.get('is_loaded', False)}")
                print_info(f"GPU Available: {data.get('gpu_available', False)}")

                if "parameters" in data:
                    params = data["parameters"]
                    total_params = params.get("total_parameters", 0)
                    if total_params > 0:
                        print_info(f"Parameters: {total_params:,}")
                        print_info(f"Device: {params.get('device', 'Unknown')}")
                        print_info(f"Model Type: {params.get('model_type', 'Unknown')}")
                    else:
                        print_warning(
                            "Model parameters show 0 - model may not be loaded"
                        )

                if "memory_usage" in data:
                    mem = data["memory_usage"]
                    print_info(
                        f"Memory: {mem.get('used_gb', 0):.1f}GB used, {mem.get('percent', 0):.1f}%"
                    )

                return data.get("is_loaded", False)
            else:
                print_error(f"Model info failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Model info error: {e}")
            return False


async def test_text_generation(model_loaded=False):
    """Test text generation functionality"""
    print_section("Text Generation Test", "🔤")

    base_url_llm = "http://localhost:8002"

    if not model_loaded:
        print_warning("Model not loaded - generation will likely fail")

    test_prompts = [
        {"text": "سلام، چطوری؟", "description": "Simple Persian greeting"},
        {"text": "Hello, how are you?", "description": "Simple English greeting"},
        {"text": "Tell me a story", "description": "Story request"},
    ]

    async with httpx.AsyncClient(timeout=60) as client:
        for i, prompt_info in enumerate(test_prompts, 1):
            print_info(f"Test {i}: {prompt_info['description']}")

            generation_request = {
                "text": prompt_info["text"],
                "max_length": 50,
                "temperature": 0.7,
                "top_p": 0.9,
            }

            try:
                print_info(f"Input: '{prompt_info['text']}'")
                start_time = time.time()

                response = await client.post(
                    f"{base_url_llm}/generate", json=generation_request, timeout=60
                )

                total_time = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    print_success("Generation successful!")

                    generated = result.get("generated_text", "")
                    # Truncate long outputs
                    display_text = (
                        generated[:100] + "..." if len(generated) > 100 else generated
                    )
                    print_info(f"Output: {display_text}")
                    print_info(
                        f"Generation Time: {result.get('generation_time', 0):.2f}s"
                    )
                    print_info(f"Total Time: {total_time:.2f}s")
                    print_info(f"GPU Used: {result.get('gpu_used', False)}")
                    print_info(f"Tokens: {result.get('token_count', 0)}")
                    print_info(f"Model: {result.get('model_name', 'Unknown')}")

                    return True
                else:
                    print_error(f"Generation failed: HTTP {response.status_code}")
                    try:
                        error_detail = response.json()
                        print_error(
                            f"Error: {error_detail.get('detail', 'Unknown error')}"
                        )
                    except:
                        print_error(f"Error: {response.text}")

                    if response.status_code == 503:
                        print_warning("Service unavailable - likely model not loaded")

                    # Try next prompt
                    continue

            except httpx.TimeoutException:
                print_error(f"Generation timeout after 60s")
            except Exception as e:
                print_error(f"Generation error: {e}")

        return False


async def test_gpu_coordinator_details():
    """Test detailed GPU coordinator information"""
    print_section("GPU Coordinator Details", "🎮")

    base_url_gpu = "http://localhost:8080"

    async with httpx.AsyncClient(timeout=15) as client:
        # Test status endpoint
        try:
            response = await client.get(f"{base_url_gpu}/status")
            if response.status_code == 200:
                data = response.json()
                print_success("GPU Coordinator status retrieved")

                if "gpus" in data:
                    print_info("GPU Status:")
                    for gpu_id, gpu_info in data["gpus"].items():
                        memory_used = gpu_info.get("memory_used_gb", 0)
                        running_tasks = gpu_info.get("running_tasks", 0)
                        print_info(
                            f"  GPU {gpu_id}: {running_tasks} tasks, {memory_used:.1f}GB used"
                        )

                if "queue" in data:
                    queue = data["queue"]
                    pending = queue.get("pending_tasks", 0)
                    running = queue.get("running_tasks", 0)
                    print_info(f"Queue: {pending} pending, {running} running")

            else:
                print_warning(f"GPU status endpoint failed: {response.status_code}")
        except Exception as e:
            print_error(f"GPU status error: {e}")

        # Test queue endpoint
        try:
            response = await client.get(f"{base_url_gpu}/queue")
            if response.status_code == 200:
                data = response.json()
                queue_length = data.get("queue_length", 0)
                if queue_length > 0:
                    print_warning(f"Queue has {queue_length} pending tasks")
                else:
                    print_success("Queue is empty")
            else:
                print_warning(f"Queue endpoint failed: {response.status_code}")
        except Exception as e:
            print_error(f"Queue check error: {e}")


async def test_metrics():
    """Test metrics endpoint"""
    print_section("Metrics Test", "📊")

    base_url_llm = "http://localhost:8002"

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            response = await client.get(f"{base_url_llm}/metrics")
            if response.status_code == 200:
                metrics = response.text
                print_success("Metrics retrieved")

                # Parse key metrics
                lines = metrics.split("\n")
                key_metrics = []
                for line in lines:
                    if (
                        any(
                            keyword in line
                            for keyword in [
                                "llm_requests_total",
                                "llm_gpu_utilization",
                                "llm_tokens_generated_total",
                                "llm_active_connections",
                            ]
                        )
                        and not line.startswith("#")
                        and line.strip()
                    ):
                        key_metrics.append(line.strip())

                if key_metrics:
                    print_info("Key Metrics:")
                    for metric in key_metrics[:8]:  # Show first 8 metrics
                        print_info(f"  {metric}")
                else:
                    print_warning("No key metrics found")

                return True
            else:
                print_error(f"Metrics failed: HTTP {response.status_code}")
        except Exception as e:
            print_error(f"Metrics error: {e}")

    return False


async def diagnostic_info():
    """Print diagnostic information"""
    print_section("Diagnostic Information", "🔍")

    # Check if running in Docker
    if Path("/.dockerenv").exists():
        print_info("Running inside Docker container")
    else:
        print_info("Running on host system")

    # Check environment variables
    env_vars = [
        "GPU_COORDINATOR_URL",
        "MODEL_PATH",
        "LLM_GPU_MEMORY_GB",
        "CUDA_VISIBLE_DEVICES",
    ]

    print_info("Environment Variables:")
    for var in env_vars:
        value = os.getenv(var, "Not set")
        print_info(f"  {var}: {value}")

    # Check Python path
    print_info(f"Python executable: {sys.executable}")

    # Try to import key libraries
    print_info("Library availability:")
    libraries = ["torch", "transformers", "httpx", "fastapi"]
    for lib in libraries:
        try:
            __import__(lib)
            print_success(f"  {lib}: Available")
        except ImportError:
            print_error(f"  {lib}: Not available")


async def main():
    """Main test function"""
    print(
        f"{Colors.BOLD}{Colors.PURPLE}🚀 Enhanced LLM + GPU Integration Test Suite{Colors.END}"
    )
    print("=" * 60)

    # Import os here
    import os

    try:
        # 1. Diagnostic info
        await diagnostic_info()

        # 2. Test connectivity
        connectivity = await test_service_connectivity()

        if not connectivity["llm"]:
            print_error("\n❌ LLM Service not accessible - stopping tests")
            print_info("Troubleshooting steps:")
            print_info(
                "1. Check if LLM service is running: docker ps | grep llm-service"
            )
            print_info("2. Check LLM service logs: docker logs agentic-llm-service")
            print_info("3. Verify port 8002 is accessible")
            return False

        # 3. Test GPU coordination status
        await test_gpu_coordination_status()

        # 4. Test model info
        model_loaded = await test_model_info()

        # 5. Test GPU coordinator details (if available)
        if connectivity["gpu"]:
            await test_gpu_coordinator_details()
        else:
            print_warning("Skipping GPU coordinator details - service not accessible")

        # 6. Test text generation
        generation_success = await test_text_generation(model_loaded)

        # 7. Test metrics
        await test_metrics()

        # Final summary
        print_section("Test Summary", "📝")

        if connectivity["gpu"]:
            print_success("GPU Coordinator: Connected")
        else:
            print_warning("GPU Coordinator: Not accessible")

        if connectivity["llm"]:
            print_success("LLM Service: Connected")
        else:
            print_error("LLM Service: Not accessible")

        if model_loaded:
            print_success("Model: Loaded")
        else:
            print_warning("Model: Not loaded or failed to load")

        if generation_success:
            print_success("Text Generation: Working")
        else:
            print_warning("Text Generation: Failed")

        # Recommendations
        print_section("Recommendations", "💡")

        if not model_loaded:
            print_info("Model loading issues:")
            print_info("1. Check MODEL_PATH environment variable")
            print_info("2. Verify model files exist in the specified path")
            print_info("3. Check LLM service logs for detailed error messages")
            print_info("4. Ensure sufficient memory/disk space")

        if not connectivity["gpu"]:
            print_info("GPU Coordinator issues:")
            print_info("1. Check if GPU coordinator is running")
            print_info("2. Verify network connectivity between services")
            print_info("3. Check GPU coordinator logs")

        if not generation_success and model_loaded:
            print_info("Generation issues:")
            print_info("1. Check request format and parameters")
            print_info("2. Verify model is compatible with generation requests")
            print_info("3. Check for memory or GPU allocation issues")

        print(f"\n{Colors.BOLD}{Colors.GREEN}✅ Test suite completed!{Colors.END}")
        return True

    except Exception as e:
        print_error(f"Test suite failed with error: {e}")
        return False


def quick_connectivity_test():
    """Quick synchronous connectivity test"""
    print_section("Quick Connectivity Test", "⚡")

    try:
        import requests

        # Test LLM service
        try:
            response = requests.get("http://localhost:8002/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print_success(f"LLM Service: {data.get('status', 'Unknown')}")
                print_info(f"Model loaded: {data.get('model_loaded', False)}")
                return True
            else:
                print_error(f"LLM Service failed: HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            print_error("LLM Service: Connection refused")
        except Exception as e:
            print_error(f"LLM Service error: {e}")

        # Test GPU coordinator
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print_success(f"GPU Coordinator: {data.get('status', 'Unknown')}")
            else:
                print_warning(f"GPU Coordinator: HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            print_warning("GPU Coordinator: Connection refused")
        except Exception as e:
            print_warning(f"GPU Coordinator: {e}")

        return False

    except ImportError:
        print_error("requests library not available")
        return False


if __name__ == "__main__":
    print(f"{Colors.BOLD}{Colors.CYAN}🧪 LLM + GPU Integration Test Suite{Colors.END}")
    print("=" * 60)

    # Quick test first
    if quick_connectivity_test():
        print_info("\n🔄 Running comprehensive integration tests...")
        try:
            success = asyncio.run(main())
            if success:
                print(
                    f"\n{Colors.BOLD}{Colors.GREEN}🎉 All tests completed successfully!{Colors.END}"
                )
            else:
                print(
                    f"\n{Colors.BOLD}{Colors.YELLOW}⚠️  Tests completed with issues{Colors.END}"
                )
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⏸️  Tests interrupted by user{Colors.END}")
        except Exception as e:
            print(f"\n{Colors.RED}💥 Test suite crashed: {e}{Colors.END}")
    else:
        print_error("\n❌ Basic connectivity failed")
        print_info("\nTroubleshooting steps:")
        print_info("1. Check if services are running:")
        print_info("   docker ps | grep -E '(llm-service|gpu-coordinator)'")
        print_info("2. Check service logs:")
        print_info("   docker logs agentic-llm-service")
        print_info("   docker logs agentic-gpu-coordinator")
        print_info("3. Verify ports are accessible:")
        print_info("   curl http://localhost:8002/health")
        print_info("   curl http://localhost:8080/health")
        print_info("4. Check Docker network configuration")

        sys.exit(1)
