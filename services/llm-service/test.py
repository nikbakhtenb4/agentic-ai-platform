#!/usr/bin/env python3
"""
Test script for LLM Service
تست سرویس LLM برای بررسی عملکرد
"""

import asyncio
import json
import time
import httpx
from typing import Dict, Any


class LLMServiceTester:
    def __init__(self, base_url: str = "https://localhost:8002"):
        self.base_url = base_url.rstrip("/")

    async def test_health(self) -> Dict[str, Any]:
        """Test health endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return {"status": "✅ PASS", "data": response.json()}
        except Exception as e:
            return {"status": "❌ FAIL", "error": str(e)}

    async def test_model_info(self) -> Dict[str, Any]:
        """Test model info endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/model/info")
                response.raise_for_status()
                return {"status": "✅ PASS", "data": response.json()}
        except Exception as e:
            return {"status": "❌ FAIL", "error": str(e)}

    async def test_generation(self, text: str = "سلام، چطوری؟") -> Dict[str, Any]:
        """Test text generation endpoint"""
        try:
            payload = {
                "text": text,
                "max_length": 50,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "num_return_sequences": 1,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                return {"status": "✅ PASS", "data": response.json()}
        except Exception as e:
            return {"status": "❌ FAIL", "error": str(e)}

    async def test_metrics(self) -> Dict[str, Any]:
        """Test metrics endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/metrics")
                response.raise_for_status()
                return {"status": "✅ PASS", "metrics_length": len(response.text)}
        except Exception as e:
            return {"status": "❌ FAIL", "error": str(e)}

    async def run_all_tests(self):
        """Run all tests"""
        print("🤖 Testing LLM Service")
        print("=" * 50)

        # Test 1: Health Check
        print("1️⃣ Testing Health Check...")
        health_result = await self.test_health()
        print(f"   {health_result['status']}")
        if health_result["status"] == "✅ PASS":
            data = health_result["data"]
            print(f"   📊 Status: {data.get('status')}")
            print(f"   🤖 Model Loaded: {data.get('model_loaded')}")
            print(f"   🎮 GPU Available: {data.get('gpu_available')}")
            print(
                f"   💾 Memory Usage: {data.get('memory_usage', {}).get('used_gb', 0):.1f}GB"
            )
            print(f"   ⏱️ Uptime: {data.get('uptime', 0):.1f}s")
        else:
            print(f"   ❌ Error: {health_result.get('error')}")
        print()

        # Test 2: Model Info
        print("2️⃣ Testing Model Info...")
        model_result = await self.test_model_info()
        print(f"   {model_result['status']}")
        if model_result["status"] == "✅ PASS":
            data = model_result["data"]
            print(f"   📝 Model Name: {data.get('model_name')}")
            print(f"   📂 Model Path: {data.get('model_path')}")
            print(f"   ✅ Is Loaded: {data.get('is_loaded')}")
            params = data.get("parameters", {})
            print(f"   🔢 Total Parameters: {params.get('total_parameters', 0):,}")
            print(f"   📊 Status: {params.get('status')}")
        else:
            print(f"   ❌ Error: {model_result.get('error')}")
        print()

        # Test 3: Text Generation
        print("3️⃣ Testing Text Generation...")
        gen_result = await self.test_generation()
        print(f"   {gen_result['status']}")
        if gen_result["status"] == "✅ PASS":
            data = gen_result["data"]
            print(f"   📝 Generated Text: {data.get('generated_text', '')[:100]}...")
            print(f"   🤖 Model: {data.get('model_name')}")
            print(f"   🎮 GPU Used: {data.get('gpu_used')}")
            print(f"   🔢 Token Count: {data.get('token_count')}")
            print(f"   ⏱️ Generation Time: {data.get('generation_time', 0):.2f}s")
        else:
            print(f"   ❌ Error: {gen_result.get('error')}")
        print()

        # Test 4: Metrics
        print("4️⃣ Testing Metrics...")
        metrics_result = await self.test_metrics()
        print(f"   {metrics_result['status']}")
        if metrics_result["status"] == "✅ PASS":
            print(
                f"   📊 Metrics Length: {metrics_result.get('metrics_length')} characters"
            )
        else:
            print(f"   ❌ Error: {metrics_result.get('error')}")
        print()

        # Summary
        tests = [health_result, model_result, gen_result, metrics_result]
        passed = sum(1 for test in tests if test["status"] == "✅ PASS")
        print("📊 Test Summary")
        print("=" * 50)
        print(f"✅ Passed: {passed}/4")
        print(f"❌ Failed: {4 - passed}/4")

        if passed == 4:
            print("🎉 All tests passed! LLM Service is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the errors above.")

        return passed == 4


async def main():
    """Main test function"""
    tester = LLMServiceTester()

    print("🚀 Starting LLM Service Tests...")
    print()

    # Wait a bit for service to be ready
    print("⏳ Waiting for service to be ready...")
    await asyncio.sleep(2)

    success = await tester.run_all_tests()

    if success:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
