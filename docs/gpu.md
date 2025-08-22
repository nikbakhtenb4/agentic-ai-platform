powershell:
(base) PS D:\Project\Server-OpenAI\agentic-ai-platform\services\gpu-coordinator> .\start_gpu_coordinator.ps1

docker run -it --name gpu-coord-container gpu-coord:test bash
python test_gpu.py
