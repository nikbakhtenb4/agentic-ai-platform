import requests
import json


def test_gpu():
    print("🧪 Testing GPU Coordinator...")
    print("=" * 40)

    base_url = "http://localhost:8080"

    # Test 1: Health
    print("\\n📍 1. Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data['status']}")
            print(f"   🎮 GPU Available: {data['gpu_available']}")
            print(f"   📊 Total GPUs: {data['total_gpus']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Test 2: Status
    print("\\n📍 2. System Status")
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            sys_info = data.get("system", {})
            gpus = data.get("gpus", {})
            print(f"   💾 Total Memory: {sys_info.get('total_memory_gb', 0):.1f} GB")
            print(f"   💾 Free Memory: {sys_info.get('free_memory_gb', 0):.1f} GB")
            print(f"   ⚡ GPU Count: {len(gpus)}")
        else:
            print(f"   ⚠️ Status failed: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ Status error: {e}")

    # Test 3: GPU Request
    print("\\n📍 3. GPU Request/Release")
    try:
        request_data = {
            "service_name": "test-service",
            "estimated_memory": 1.0,
            "priority": "normal",
        }

        response = requests.post(f"{base_url}/request", json=request_data, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Allocated: {data.get('allocated')}")

            if data.get("allocated"):
                task_id = data.get("task_id")
                gpu_id = data.get("gpu_id")
                print(f"   🎮 GPU {gpu_id} allocated (Task: {task_id})")

                # Release
                import time

                time.sleep(1)
                release_response = requests.post(
                    f"{base_url}/release/{task_id}", timeout=5
                )
                if release_response.status_code == 200:
                    print(f"   ✅ GPU released successfully")
                else:
                    print(f"   ⚠️ Release failed: {release_response.status_code}")
            else:
                print(f"   ⚠️ Not allocated: {data.get('message')}")
        else:
            print(f"   ❌ Request failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Request error: {e}")

    print("\\n✅ Test completed!")
    return True


if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("❌ Installing requests...")
        import subprocess

        subprocess.run(["pip", "install", "requests"])
        import requests

    test_gpu()
