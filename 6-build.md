مرحله 2: Build تک تک

# شروع با GPU Coordinator:

<!-- cd services/gpu-coordinator -->

docker build -f services/gpu-coordinator/Dockerfile -t gpu-coordinator .

# اگر موفق بود، ادامه با بقیه:

cd ../llm-service  
docker build -t llm-service . --progress=plain

cd ../audio-service/stt
docker build -t stt-service . --progress=plain
docker build -t stt-service .

cd ../../../api-gateway
docker build -t api-gateway . --progress=plain

--------------wheels------------

(base) PS D:\Project\Server-OpenAI\agentic-ai-platform\services\audio-service\stt> docker build -t stt-service .

(base) PS D:\Project\Server-OpenAI\agentic-ai-platform> docker build -f services/gpu-coordinator/Dockerfile -t gpu-coordinator .
PS D:\Project\Server-OpenAI\agentic-ai-platform> docker compose build llm-service

(base) PS D:\Project\Server-OpenAI\agentic-ai-platform> conda activate agentic-platform

--------------wheels------------

<!-- pip install markupsafe>=2.0
pip download torch==2.5.1 torchvision torchaudio --platform linux_x86_64 --python-version 3.11 --index-url https://download.pytorch.org/whl/cu118 -d wheels/ --only-binary=:all:


pip download torch==2.5.1 torchvision torchaudio  --platform linux_x86_64  --python-version 3.11  --index-url https://download.pytorch.org/whl/cu118   -d wheels/    --only-binary=:all: markupsafe>=2.0 jinja2>=3.1.4
(agentic-platform) PS D:\Project\Server-OpenAI\agentic-ai-platform> docker compose build gpu-coordinator --no-cache -->

-----------------build---------------
conda activate agentic-platform  
(agentic-platform) PS D:\Project\Server-OpenAI\agentic-ai-platform>
docker build -f services/gpu-coordinator/Dockerfile -t gpu-coordinator .
docker build -f services/llm-service/Dockerfile -t llm-service .  
docker build --progress=plain --no-cache -f services/audio-service/stt/Dockerfile -t stt .
docker build -f services/test-service/Dockerfile -t test-service .

pip install requests  
python scripts/setup.py
