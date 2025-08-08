services/audio-service/stt/
├── main.py # FastAPI application
├── Dockerfile # Container setup
├── requirements.txt # Dependencies
├── config/
│ └── stt_config.yaml # Configuration
├── models/
│ └── whisper_handler.py # Whisper model management
├── utils/
│ ├── audio_processor.py # Audio processing
│ ├── gpu_coordinator.py # Shared GPU management
│ └── cache_manager.py # Caching system
└── schemas/
└── stt_schemas.py # Pydantic models

🚀 ویژگی‌های پیاده‌سازی شده:

Whisper Integration: استفاده از مدل Whisper برای STT
Multi-language: پشتیبانی فارسی و انگلیسی
Audio Processing: پردازش پیشرفته صوتی
Batch Processing: پردازش چندین فایل همزمان
GPU Acceleration: بهره‌گیری از GPU
Docker & Kubernetes: آماده برای deployment
Error Handling: مدیریت خطا و logging کامل

📋 API Endpoints:
POST /stt/transcribe - تبدیل تک فایل صوتی
POST /stt/transcribe-batch - تبدیل چندین فایل
GET /stt/languages - زبان‌های پشتیبانی شده  
GET /stt/health - وضعیت سرویس
تغییرات برای پشتیبانی موبایل/وب:
✅ قابلیت‌های اضافه شده:

CORS Configuration: برای دسترسی از مرورگر
Base64 Audio Support: برای ارسال صدا از موبایل اپ
Enhanced File Upload: پشتیبانی فرمت‌های بیشتر
Better Error Handling: پیام‌های خطای واضح‌تر
File Size Validation: چک سایز فایل برای موبایل
Options Handler: پشتیبانی CORS preflight

🔗 API Endpoints برای موبایل/وب:
bash# برای وب (فایل آپلود)
POST /stt/transcribe

# برای موبایل (Base64)

POST /stt/transcribe-base64

# برای پردازش چندتایی

POST /stt/transcribe-batch

# اطلاعات سرویس

GET /stt/languages
GET /stt/health
📱 نمونه استفاده از موبایل:
javascript// React Native / Mobile App
const uploadAudio = async (audioBase64, filename) => {
const response = await fetch('/api/stt/transcribe-base64', {
method: 'POST',
headers: {
'Content-Type': 'application/json',
},
body: JSON.stringify({
audio_data: audioBase64,
filename: filename,
language: 'fa', // یا 'en'
format: 'wav'
})
});

return await response.json();
};
🌐 نمونه استفاده از وب:
javascript// Web App
const uploadAudioFile = async (file, language) => {
const formData = new FormData();
formData.append('audio_file', file);
formData.append('language', language);

const response = await fetch('/api/stt/transcribe', {
method: 'POST',
body: formData
});

return await response.json();
};

---

# راهنمای کامل راه‌اندازی سیستم Agentic AI Platform با STT

## 📋 فهرست مطالب

1. [معرفی سیستم](#معرفی-سیستم)
2. [پیش‌نیازها](#پیش‌نیازها)
3. [ساختار پروژه](#ساختار-پروژه)
4. [راه‌اندازی](#راه‌اندازی)
5. [سرویس‌های موجود](#سرویس‌های-موجود)
6. [تست و استفاده](#تست-و-استفاده)
7. [مانیتورینگ](#مانیتورینگ)
8. [عیب‌یابی](#عیب‌یابی)

## 🚀 معرفی سیستم

سیستم Agentic AI Platform یک پلتفرم کامل هوش مصنوعی است که شامل سرویس‌های زیر می‌باشد:

### ✨ ویژگی‌های کلیدی

- **LLM Service**: سرویس تولید متن با مدل GPT2-FA فارسی
- **STT Service**: سرویس تبدیل گفتار به متن با Whisper
- **GPU Coordinator**: مدیریت مشترک منابع GPU بین سرویس‌ها
- **API Gateway**: مدیریت مرکزی API ها
- **Monitoring**: مانیتورینگ کامل با Prometheus و Grafana

### 🎯 قابلیت‌های STT Service

- پشتیبانی از زبان‌های فارسی و انگلیسی
- پردازش فایل‌های صوتی متنوع (WAV, MP3, M4A, FLAC, OGG, WebM)
- پردازش Base64 برای اپلیکیشن‌های موبایل
- پردازش دسته‌ای (Batch Processing)
- کش کردن نتایج برای بهبود کارایی
- پردازش و بهبود کیفیت صدا
- مدیریت GPU مشترک با سایر سرویس‌ها

## 🛠 پیش‌نیازها

### سیستم عامل

- Windows 10/11، Linux، یا macOS
- حداقل 16GB RAM (32GB توصیه می‌شود)
- حداقل 50GB فضای خالی دیسک
- GPU NVIDIA با حداقل 6GB VRAM (اختیاری اما توصیه می‌شود)

### نرم‌افزارهای مورد نیاز

- Docker Desktop
- Docker Compose V2
- Python 3.11+ (برای تست)
- Git
- NVIDIA Container Toolkit (برای استفاده از GPU)

### مدل‌های هوش مصنوعی

- مدل GPT2-FA فارسی در مسیر `data/models/llm/gpt2-fa/`
- مدل Whisper (به صورت خودکار دانلود می‌شود)

## 📁 ساختار پروژه

```
agentic-ai-platform/
├── docker-compose.yml
├── nginx.conf
├── .env
├── setup.py
├── data/
│   ├── models/
│   │   ├── llm/gpt2-fa/
│   │   └── stt/
│   ├── cache/
│   ├── logs/
│   └── uploads/
├── services/
│   ├── api-gateway/
│   ├── llm-service/
│   ├── audio-service/stt/
│   ├── gpu-coordinator/
│   └── test-service/
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── shared/
│   └── database/init.sql
└── tests/
    └── stt_test_client.py
```

## 🚀 راه‌اندازی

### گام 1: کلون و آماده‌سازی

```bash
# کلون پروژه
git clone <repository-url>
cd agentic-ai-platform

# ایجاد دایرکتوری‌های مورد نیاز
mkdir -p data/models/llm/gpt2-fa
mkdir -p data/models/stt
mkdir -p data/cache/stt
mkdir -p data/logs
mkdir -p data/uploads

# کپی مدل GPT2-FA
# فایل‌های مدل را در data/models/llm/gpt2-fa/ قرار دهید
```

### گام 2: تنظیم محیط

```bash
# کپی فایل environment
cp .env.example .env

# ویرایش متغیرهای محیطی در صورت نیاز
# nano .env
```

### گام 3: راه‌اندازی با Docker

```bash
# بیلد و اجرای سرویس‌ها
docker compose up --build -d

# مشاهده وضعیت سرویس‌ها
docker compose ps

# مشاهده لاگ‌ها
docker compose logs -f
```

### گام 4: تست سلامت

```bash
# تست سلامت کلی
curl https://localhost/health

# تست سلامت STT
curl https://localhost/stt/health

# تست سلامت LLM
curl https://localhost/llm/health
```

## 🔧 سرویس‌های موجود

### 1. API Gateway (Port: 8000)

- **نقش**: مدیریت مرکزی درخواست‌ها
- **URL**: https://localhost:8000
- **Endpoints**:
  - `/health` - بررسی سلامت
  - `/api/info` - اطلاعات API
  - `/api/v1/llm/*` - سرویس‌های LLM
  - `/api/v1/stt/*` - سرویس‌های STT

### 2. LLM Service (Port: 8002)

- **نقش**: تولید متن با مدل GPT2-FA
- **URL**: https://localhost:8002
- **ویژگی‌ها**: تولید متن، گفتگو، پردازش دسته‌ای

### 3. STT Service (Port: 8003)

- **نقش**: تبدیل گفتار به متن
- **URL**: https://localhost:8003
- **ویژگی‌ها**:
  - پشتیبانی از فرمت‌های مختلف صوتی
  - پردازش Base64 برای موبایل
  - پردازش دسته‌ای
  - کش کردن نتایج

### 4. GPU Coordinator (Port: 8080)

- **نقش**: مدیریت مشترک GPU
- **URL**: https://localhost:8080
- **ویژگی‌ها**: تخصیص هوشمند GPU، مدیریت صف

### 5. Monitoring

- **Prometheus**: https://localhost:9090
- **Grafana**: https://localhost:3000 (admin/admin)

## 🧪 تست و استفاده

### تست STT Service

#### 1. تست ساده با cURL

```bash
# آپلود فایل صوتی
curl -X POST https://localhost/api/v1/stt/transcribe \
  -F "audio_file=@test_audio.wav" \
  -F "language=fa" \
  -F "task=transcribe"

# بررسی زبان‌های پشتیبانی شده
curl https://localhost/api/v1/stt/languages
```

#### 2. تست با Python Client

```bash
# نصب وابستگی‌ها
pip install aiohttp aiofiles

# اجرای تست
cd tests
python stt_test_client.py --file test_audio.wav --language fa

# تست کارایی
python stt_test_client.py --file test_audio.wav --performance 5

# تست دسته‌ای
python stt_test_client.py --batch --files audio1.wav audio2.wav audio3.wav
```

#### 3. تست Base64 (شبیه‌سازی موبایل)

```bash
python stt_test_client.py --file test_audio.wav --base64 --language fa
```

### تست LLM Service

```bash
# تولید متن
curl -X POST https://localhost/api/v1/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "سلام", "max_length": 100, "temperature": 0.7}'

# گفتگو
curl -X POST https://localhost/api/v1/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "سلام، حالت چطوره؟"}'
```

## 📊 مانیتورینگ

### Grafana Dashboard

1. بروید به https://localhost:3000
2. وارد شوید با admin/admin
3. Dashboard های آماده را مشاهده کنید

### متریک‌های کلیدی

- **STT Metrics**:
  - تعداد درخواست‌ها
  - زمان پردازش
  - نرخ موفقیت
  - استفاده از GPU
- **LLM Metrics**:
  - تعداد تولید متن
  - زمان پاسخ
  - استفاده از حافظه
- **GPU Metrics**:
  - استفاده از GPU
  - حافظه اشغال شده
  - صف انتظار

## 🔧 تنظیمات پیشرفته

### تنظیم STT Service

فایل `services/audio-service/stt/config/stt_config.yaml`:

```yaml
stt_service:
  model:
    size: "base" # tiny, base, small, medium, large
    device: "auto" # auto, cpu, cuda
  audio:
    max_file_size: 25 # MB
    max_duration: 600 # seconds
  performance:
    max_concurrent_requests: 5
    timeout: 300
  cache:
    enabled: true
    ttl: 3600 # 1 hour
```

### تنظیم متغیرهای محیطی

فایل `.env`:

```env
# STT Configuration
WHISPER_MODEL_SIZE=base
MAX_FILE_SIZE_MB=25
MAX_CONCURRENT_REQUESTS=5
CACHE_ENABLED=true
CACHE_TTL=3600

# GPU Configuration
MAX_GPU_MEMORY_MB=6144
```

## 🚨 عیب‌یابی

### مشکلات رایج

#### 1. مدل LLM لود نمی‌شود

```bash
# بررسی وجود فایل‌های مدل
ls -la data/models/llm/gpt2-fa/

# بررسی لاگ‌های LLM Service
docker compose logs llm-service
```

#### 2. GPU تشخیص داده نمی‌شود

```bash
# بررسی GPU
nvidia-smi

# بررسی NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# بررسی لاگ‌های GPU Coordinator
docker compose logs gpu-coordinator
```

#### 3. STT Service آهسته است

```bash
# بررسی استفاده از منابع
docker stats

# کاهش همزمان درخواست‌ها
# در .env: MAX_CONCURRENT_REQUESTS=2

# استفاده از مدل کوچکتر
# در .env: WHISPER_MODEL_SIZE=tiny
```

#### 4. خطاهای حافظه

```bash
# بررسی حافظه موجود
free -h

# کاهش حافظه محدود شده برای سرویس‌ها
# در docker-compose.yml، کاهش memory limits
```

### لاگ‌ها و دیباگ

```bash
# مشاهده لاگ‌های زنده
docker compose logs -f stt-service

# مشاهده لاگ‌های خاص
docker compose logs stt-service | grep ERROR

# بررسی وضعیت سرویس‌ها
docker compose ps

# ری‌استارت سرویس خاص
docker compose restart stt-service
```

## 📚 API Documentation

### STT Service Endpoints

#### POST /transcribe

آپلود و تبدیل فایل صوتی به متن

```bash
curl -X POST https://localhost/api/v1/stt/transcribe \
  -F "audio_file=@audio.wav" \
  -F "language=fa" \
  -F "task=transcribe"
```

**Response:**

```json
{
  "text": "متن تبدیل شده از گفتار",
  "language": "fa",
  "confidence": 0.95,
  "duration": 15.5,
  "processing_time": 2.3,
  "segments": [...]
}
```

#### POST /transcribe-base64

تبدیل صدای Base64 (برای موبایل)

```bash
curl -X POST https://localhost/api/v1/stt/transcribe-base64 \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "UklGRnbiAQBXQVZFZm...",
    "filename": "audio.wav",
    "language": "fa",
    "format": "wav"
  }'
```

#### POST /transcribe-batch

پردازش دسته‌ای چندین فایل

```bash
curl -X POST https://localhost/api/v1/stt/transcribe-batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav"
```

#### GET /languages

دریافت زبان‌های پشتیبانی شده

```bash
curl https://localhost/api/v1/stt/languages
```

## 🔒 امنیت

### تنظیمات امنیتی

- تغییر پسوردهای پیش‌فرض در production
- محدودیت CORS برای production
- استفاده از HTTPS
- محدودیت rate limiting
- احراز هویت و اختیارات کاربران

### Production Checklist

- [ ] تغییر JWT_SECRET_KEY
- [ ] تغییر پسورد پایگاه داده
- [ ] تنظیم CORS محدود
- [ ] فعال‌سازی HTTPS
- [ ] تنظیم backup خودکار
- [ ] مانیتورینگ و alerting
- [ ] تست load testing
- [ ] تنظیم log rotation

## 🚀 Performance Tuning

### بهینه‌سازی STT

- استفاده از مدل مناسب (base برای تعادل سرعت/دقت)
- تنظیم cache TTL
- کاهش max_concurrent_requests در صورت محدودیت منابع
- استفاده از GPU برای سرعت بیشتر

### بهینه‌سازی GPU

- مدیریت priority برای درخواست‌های مختلف
- تنظیم memory limits مناسب
- monitoring استفاده از GPU

### Scale Up/Out

- اضافه کردن replica برای سرویس‌ها
- استفاده از load balancer
- database clustering
- microservices architecture

---

## 🆘 پشتیبانی

برای سوالات و مشکلات:

1. بررسی این راهنما
2. مطالعه لاگ‌های سیستم
3. بررسی issues در repository
4. تماس با تیم توسعه

**نکته**: این سیستم برای محیط development طراحی شده و برای production نیاز به تنظیمات امنیتی اضافی دارد.

---

## 📄 لایسنس

این پروژه تحت لایسنس MIT منتشر شده است.
خلاصه پیاده‌سازی کامل
سیستم Agentic AI Platform با سرویس STT به صورت کامل پیاده‌سازی شد! در ادامه خلاصه‌ای از آنچه انجام دادیم:
✅ آنچه پیاده‌سازی شد:
🎯 سرویس STT (Speech-to-Text)

Enhanced STT Service با Whisper مدل
پشتیبانی از فارسی و انگلیسی
پردازش Base64 برای موبایل
پردازش دسته‌ای (Batch Processing)
کش کردن نتایج برای بهبود کارایی
پردازش صوتی پیشرفته (noise reduction، normalization)
مدیریت خطا و retry logic

🎮 مدیریت مشترک GPU

GPU Coordinator برای تخصیص هوشمند GPU
Priority-based scheduling
Resource management بین LLM و STT
Monitoring و metrics برای GPU

🔧 بهبودهای زیرساختی

Docker Compose بروزرسانی شده
Nginx با پشتیبانی STT
API Gateway با routing های STT
Configuration management پیشرفته
Monitoring کامل با Prometheus

📁 فایل‌های ایجاد شده:

STT Service Core:

services/audio-service/stt/main.py - سرویس اصلی
services/audio-service/stt/Dockerfile - داکر
services/audio-service/stt/requirements.txt - وابستگی‌ها

Configuration:

services/audio-service/stt/config/stt_config.yaml
services/audio-service/stt/utils/audio_processor.py

GPU Management:

utils/gpu_coordinator.py - مدیریت GPU مشترک

Infrastructure:

docker-compose.yml - بروزرسانی شده
nginx.conf - پیکربندی جدید
Prometheus configs

Testing:

tests/stt_test_client.py - کلاینت تست کامل

Documentation:

راهنمای کامل راه‌اندازی و استفاده

🚀 قابلیت‌های کلیدی:
STT Service Features:
✅ Multi-format Support - WAV, MP3, M4A, FLAC, OGG, WebM
✅ Multilingual - فارسی، انگلیسی، تشخیص خودکار
✅ Mobile Ready - پشتیبانی Base64
✅ Batch Processing - پردازش همزمان چندین فایل
✅ Advanced Audio Processing - noise reduction، normalization
✅ Caching System - کش هوشمند نتایج
✅ GPU Acceleration - استفاده بهینه از GPU
✅ Monitoring & Metrics - متریک‌های کامل
GPU Coordination:
✅ Shared GPU Management - تخصیص هوشمند بین سرویس‌ها
✅ Priority Scheduling - اولویت‌بندی درخواست‌ها
✅ Resource Monitoring - نظارت بر استفاده از منابع
✅ Automatic Cleanup - پاکسازی خودکار
🔧 نحوه استفاده:
راه‌اندازی:

# 🎯 نمونه‌های کاربردی سرویس STT پیشرفته

## 🌟 منابع ورودی مختلف

### 1. آپلود فایل از وب اپلیکیشن

```javascript
// Frontend JavaScript
async function uploadAudioFile(file, language = "fa", translateTo = null) {
  const formData = new FormData();
  formData.append("audio_file", file);
  formData.append("language", language);
  formData.append("task", "transcribe");

  if (translateTo) {
    formData.append("translate_to", translateTo);
  }

  try {
    const response = await fetch("/api/v1/stt/transcribe", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    console.log("Transcribed text:", result.text);
    console.log("Language:", result.language);
    console.log("Confidence:", result.confidence);

    if (result.translation) {
      console.log("Translation:", result.translation.translated_text);
    }

    return result;
  } catch (error) {
    console.error("Transcription failed:", error);
  }
}

// استفاده
const fileInput = document.getElementById("audioFile");
const file = fileInput.files[0];
uploadAudioFile(file, "fa", "en"); // فارسی به انگلیسی
```

### 2. ارسال صوت از موبایل اپ (Base64)

```javascript
// React Native / Mobile App
async function transcribeAudioFromMobile(
  audioBase64,
  filename,
  language = "fa"
) {
  const payload = {
    audio_data: audioBase64,
    filename: filename,
    language: language,
    format: "wav",
  };

  try {
    const response = await fetch("/api/v1/stt/transcribe-base64", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const result = await response.json();

    console.log("Mobile transcription:", result.text);
    console.log("Processing time:", result.processing_time);

    return result;
  } catch (error) {
    console.error("Mobile transcription failed:", error);
  }
}

// تبدیل فایل صوتی به Base64
function audioToBase64(audioFile) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(",")[1]; // حذف header
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(audioFile);
  });
}

// استفاده
const audioFile = recordedAudio; // فایل ضبط شده
const base64Audio = await audioToBase64(audioFile);
transcribeAudioFromMobile(base64Audio, "recorded_audio.wav", "fa");
```

### 3. دانلود و تبدیل از URL

```python
# Python Example
import requests

def transcribe_from_url(audio_url, language='auto', translate_to=None):
    """تبدیل فایل صوتی از URL"""

    payload = {
        "url": audio_url,
        "language": language,
        "max_file_size": 50  # 50MB
    }

    params = {}
    if translate_to:
        params["translate_to"] = translate_to

    try:
        response = requests.post(
            "https://localhost/api/v1/stt/transcribe-url",
            json=payload,
            params=params,
            timeout=300  # 5 minutes
        )

        if response.status_code == 200:
            result = response.json()

            print(f"URL: {audio_url}")
            print(f"Text: {result['text']}")
            print(f"Language: {result['language']}")
            print(f"Duration: {result['duration']:.2f}s")

            if result.get('translation'):
                print(f"Translation: {result['translation']['translated_text']}")

            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

# استفاده
# تبدیل از یک پادکست فارسی و ترجمه به انگلیسی
result = transcribe_from_url(
    "https://example.com/persian_podcast.mp3",
    language="fa",
    translate_to="en"
)
```

### 4. تبدیل فایل لوکال سرور

```bash
# cURL Example
curl -X POST https://localhost/api/v1/stt/transcribe-local \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/app/uploads/meeting_recording.wav",
    "language": "fa"
  }'

# Response:
# {
#   "text": "متن جلسه تبدیل شده...",
#   "language": "fa",
#   "confidence": 0.95,
#   "duration": 1800.0,
#   "processing_time": 45.2,
#   "source_type": "local_path"
# }
```

## 🌐 ترجمه آنلاین

### 1. ترجمه همراه با تبدیل گفتار

```python
def transcribe_and_translate(audio_file_path, source_lang="fa", target_lang="en"):
    """تبدیل گفتار + ترجمه یکجا"""

    with open(audio_file_path, 'rb') as f:
        files = {'audio_file': f}
        data = {
            'language': source_lang,
            'task': 'transcribe',
            'translate_to': target_lang
        }

        response = requests.post(
            "https://localhost:8003/transcribe",
            files=files,
            data=data
        )

        if response.status_code == 200:
            result = response.json()

            print("Original text:", result['text'])

            if result.get('translation') and result['translation']['success']:
                print("Translated text:", result['translation']['translated_text'])
            else:
                print("Translation failed:", result.get('translation', {}).get('error'))

            return result

# استفاده
transcribe_and_translate("persian_speech.wav", "fa", "en")
```

### 2. ترجمه مستقل متن

```javascript
// Standalone Translation
async function translateText(text, sourceLang, targetLang) {
  const payload = {
    text: text,
    source_language: sourceLang,
    target_language: targetLang,
    service: "google",
  };

  try {
    const response = await fetch("/api/v1/stt/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await response.json();

    if (result.success) {
      return result.translated_text;
    } else {
      throw new Error(result.error);
    }
  } catch (error) {
    console.error("Translation error:", error);
    return null;
  }
}

// استفاده
const persianText = "این یک متن فارسی است";
const englishText = await translateText(persianText, "fa", "en");
console.log(englishText); // "This is a Persian text"
```

## 📦 پردازش دسته‌ای

### 1. پردازش چندین فایل همزمان

```python
def batch_transcribe_files(file_paths, translate_to=None):
    """پردازش دسته‌ای چندین فایل"""

    files = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            files.append(('files', (file_path, f.read(), 'audio/wav')))

    data = {}
    if translate_to:
        data['translate_to'] = translate_to

    response = requests.post(
        "https://localhost:8003/transcribe-batch",
        files=files,
        data=data,
        timeout=1800  # 30 minutes
    )

    if response.status_code == 200:
        result = response.json()

        print(f"Batch Results:")
        print(f"Total: {result['summary']['total_files']}")
        print(f"Successful: {result['summary']['successful_count']}")
        print(f"Failed: {result['summary']['failed_count']}")
        print(f"Success Rate: {result['summary']['success_rate']}")

        for item in result['results']:
            if item['success']:
                print(f"✅ {item['filename']}: {item['result']['text'][:50]}...")
            else:
                print(f"❌ {item['filename']}: {item['error']}")

        return result
    else:
        print(f"Batch processing failed: {response.status_code}")
        return None

# استفاده
audio_files = [
    "meeting1.wav",
    "interview1.wav",
    "lecture1.wav"
]
batch_transcribe_files(audio_files, translate_to="en")
```

## 🎙️ سناریوهای کاربردی واقعی

### 1. پلتفرم آموزش آنلاین

```python
class OnlineLearningPlatform:
    def __init__(self):
        self.stt_base_url = "https://localhost:8003"

    def process_lecture_recording(self, lecture_file, instructor_language="fa"):
        """پردازش ضبط جلسه درس"""

        # تبدیل گفتار به متن + ترجمه
        with open(lecture_file, 'rb') as f:
            files = {'audio_file': f}
            data = {
                'language': instructor_language,
                'translate_to': 'en'  # برای دانشجویان بین‌المللی
            }

            response = requests.post(
                f"{self.stt_base_url}/transcribe",
                files=files,
                data=data
            )

            if response.status_code == 200:
                result = response.json()

                # ذخیره متن فارسی
                persian_transcript = result['text']

                # ذخیره ترجمه انگلیسی
                english_translation = None
                if result.get('translation', {}).get('success'):
                    english_translation = result['translation']['translated_text']

                # ذخیره timestamps برای subtitle
                segments = result.get('segments', [])

                return {
                    'persian_text': persian_transcript,
                    'english_text': english_translation,
                    'segments': segments,
                    'duration': result['duration']
                }

    def create_subtitles(self, segments, language="fa"):
        """ایجاد فایل زیرنویس"""

        srt_content = ""
        for i, segment in enumerate(segments):
            start_time = self.seconds_to_srt_time(segment['start'])
            end_time = self.seconds_to_srt_time(segment['end'])
            text = segment['text'].strip()

            srt_content += f"{i+1}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{text}\n\n"

        return srt_content

    def seconds_to_srt_time(self, seconds):
        """تبدیل ثانیه به فرمت SRT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# استفاده
platform = OnlineLearningPlatform()
lecture_data = platform.process_lecture_recording("physics_lecture.wav", "fa")

# ایجاد زیرنویس فارسی
persian_srt = platform.create_subtitles(lecture_data['segments'], "fa")
with open("lecture_fa.srt", "w", encoding="utf-8") as f:
    f.write(persian_srt)
```

### 2. سیستم مدیریت تماس مشتریان

```python
class CallCenterManager:
    def __init__(self):
        self.stt_url = "https://localhost:8003"

    def process_customer_call(self, call_recording_url, customer_language="fa"):
        """پردازش تماس مشتری"""

        # دانلود و تبدیل از URL
        payload = {
            "url": call_recording_url,
            "language": customer_language,
            "max_file_size": 100  # 100MB برای تماس‌های طولانی
        }

        params = {"translate_to": "en"}  # ترجمه برای آنالیز

        response = requests.post(
            f"{self.stt_url}/transcribe-url",
            json=payload,
            params=params
        )

        if response.status_code == 200:
            result = response.json()

            return {
                'call_transcript': result['text'],
                'call_translation': result.get('translation', {}).get('translated_text'),
                'call_duration': result['duration'],
                'call_language': result['language'],
                'processing_time': result['processing_time']
            }

    def analyze_customer_sentiment(self, transcript):
        """آنالیز احساسات مشتری (نمونه)"""
        # اینجا می‌تونید از سرویس‌های آنالیز احساسات استفاده کنید
        positive_words = ['خوب', 'عالی', 'راضی', 'ممنون']
        negative_words = ['بد', 'مشکل', 'ناراضی', 'شکایت']

        positive_count = sum(1 for word in positive_words if word in transcript)
        negative_count = sum(1 for word in negative_words if word in transcript)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

# استفاده
call_center = CallCenterManager()
call_data = call_center.process_customer_call(
    "https://storage.example.com/calls/call_123.mp3",
    "fa"
)

sentiment = call_center.analyze_customer_sentiment(call_data['call_transcript'])
print(f"Customer sentiment: {sentiment}")
```

### 3. سیستم یادداشت‌برداری هوشمند

```javascript
// Smart Note-Taking System
class SmartNoteTaker {
  constructor() {
    this.baseUrl = "/api/v1/stt";
  }

  async recordAndTranscribe(
    audioBlob,
    meetingLanguage = "fa",
    translateTo = "en"
  ) {
    // تبدیل audio blob به base64
    const base64Audio = await this.blobToBase64(audioBlob);

    const payload = {
      audio_data: base64Audio,
      filename: `meeting_${Date.now()}.wav`,
      language: meetingLanguage,
      format: "wav",
    };

    const params = translateTo ? `?translate_to=${translateTo}` : "";

    try {
      const response = await fetch(
        `${this.baseUrl}/transcribe-base64${params}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );

      const result = await response.json();

      return {
        transcript: result.text,
        translation: result.translation?.translated_text,
        segments: result.segments,
        confidence: result.confidence,
        language: result.language,
      };
    } catch (error) {
      console.error("Recording transcription failed:", error);
      throw error;
    }
  }

  async blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = reader.result.split(",")[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  generateSummary(segments) {
    // خلاصه‌سازی ساده بر اساس طولانی‌ترین جملات
    const longSegments = segments
      .filter((seg) => seg.text.length > 50)
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5);

    return longSegments.map((seg) => seg.text).join(" ");
  }
}

// استفاده در اپلیکیشن
const noteTaker = new SmartNoteTaker();

// ضبط و تبدیل در جلسه
navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
  const mediaRecorder = new MediaRecorder(stream);
  const audioChunks = [];

  mediaRecorder.ondataavailable = (event) => {
    audioChunks.push(event.data);
  };

  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });

    try {
      const result = await noteTaker.recordAndTranscribe(audioBlob, "fa", "en");

      // نمایش نتایج
      document.getElementById("transcript").innerText = result.transcript;
      document.getElementById("translation").innerText = result.translation;

      // ایجاد خلاصه
      const summary = noteTaker.generateSummary(result.segments);
      document.getElementById("summary").innerText = summary;
    } catch (error) {
      console.error("Error:", error);
    }
  };

  // شروع ضبط
  mediaRecorder.start();

  // توقف بعد از 30 ثانیه (مثال)
  setTimeout(() => {
    mediaRecorder.stop();
    stream.getTracks().forEach((track) => track.stop());
  }, 30000);
});
```

## 🔧 نکات بهینه‌سازی

### 1. انتخاب مدل مناسب

```bash
# برای سرعت بیشتر (مناسب real-time)
WHISPER_MODEL_SIZE=tiny

# برای تعادل سرعت/دقت (پیشنهادی)
WHISPER_MODEL_SIZE=base

# برای دقت بالا (مناسب محتوای مهم)
WHISPER_MODEL_SIZE=large
```

### 2. مدیریت حافظه

```python
# پردازش فایل‌های بزرگ به صورت تکه‌ای
def process_large_audio(file_path, chunk_duration=30):
    """پردازش فایل‌های صوتی بزرگ"""

    # تقسیم فایل به بخش‌های کوچکتر
    # و پردازش جداگانه هر بخش

    results = []
    # پیاده‌سازی تقسیم‌بندی...

    return results
```

### 3. کش کردن نتایج

```python
import hashlib

def get_audio_hash(audio_content):
    """ایجاد hash برای فایل صوتی"""
    return hashlib.md5(audio_content).hexdigest()

def cached_transcribe(audio_file, language="fa"):
    """تبدیل با کش"""

    with open(audio_file, 'rb') as f:
        content = f.read()

    audio_hash = get_audio_hash(content)
    cache_key = f"stt_{audio_hash}_{language}"

    # بررسی cache
    # اگر وجود داشت، return کن
    # در غیر این صورت، پردازش کن و cache کن
```

این سیستم حالا قابلیت پردازش صوت از منابع مختلف و ترجمه آنلاین را دارد! 🚀

🎯 خلاصه کامل سیستم STT پیشرفته
✅ قابلیت‌های جدید اضافه شده:
🎵 منابع ورودی متنوع:

File Upload - آپلود مستقیم فایل (وب اپلیکیشن)
Base64 Audio - پردازش صوت کدگذاری شده (موبایل اپ)
URL Download - دانلود و پردازش از لینک
Local File Path - پردازش فایل‌های لوکال سرور

🌐 ترجمه آنلاین:

ترجمه خودکار همراه با تبدیل گفتار
پشتیبانی از Google Translate
ترجمه مستقل متن
ترجمه دسته‌ای

🔧 ویژگی‌های پیشرفته:

Enhanced Error Handling - مدیریت خطای بهبود یافته
Multiple Format Support - WAV, MP3, M4A, FLAC, OGG, WebM, AAC
Confidence Scoring - امتیاز اطمینان برای هر بخش
Segment Timestamps - زمان‌بندی دقیق برای زیرنویس
Comprehensive Logging - لاگ‌گیری کامل
Performance Monitoring - نظارت بر کارایی

🚀 API Endpoints جدید:

---

---

---

# راهنمای کامل سرویس STT - تبدیل گفتار به متن

## 🎯 معرفی سرویس

سرویس STT شما یک سرویس پیشرفته برای تبدیل گفتار به متن است که از مدل Whisper استفاده می‌کند و امکانات زیر را ارائه می‌دهد:

- **4 روش مختلف ورودی** برای فایل‌های صوتی
- **ترجمه آنلاین** متن تبدیل شده
- **پردازش دسته‌ای** چندین فایل همزمان
- **پشتیبانی از فرمت‌های مختلف** صوتی
- **بهینه‌سازی GPU** برای سرعت بالا

---

## 🚀 راه‌اندازی سرویس

### پیش‌نیازها

```bash
# نصب dependencies
pip install -r requirements.txt

# متغیرهای محیطی مهم
export WHISPER_MODEL_SIZE="base"  # tiny, base, small, medium, large
export WHISPER_MODEL_PATH="/app/models"
export MAX_FILE_SIZE_MB="25"
export GPU_ENABLED="true"
```

### اجرای سرویس

```bash
# مستقیماً
python services/audio-service/stt/main.py

# یا با uvicorn
uvicorn main:app --host 0.0.0.0 --port 8003 --reload

# یا از طریق API Gateway
# سرویس در http://localhost:8000/api/v1/stt در دسترس است
```

---

## 📁 فرمت‌های پشتیبانی شده

### فرمت‌های صوتی

- **WAV** (.wav) - بهترین کیفیت
- **MP3** (.mp3) - رایج‌ترین فرمت
- **M4A** (.m4a) - فرمت Apple
- **FLAC** (.flac) - فشرده‌سازی بدون اتلاف
- **OGG** (.ogg) - فرمت متن‌باز
- **WebM** (.webm) - فرمت وب
- **AAC** (.aac) - فرمت پیشرفته

### محدودیت‌های فایل

- **حداکثر اندازه**: 25MB (قابل تنظیم)
- **حداکثر مدت**: 10 دقیقه (600 ثانیه)
- **نرخ نمونه‌برداری بهینه**: 16kHz
- **کانال**: مونو (تک‌کانال) ترجیح داده می‌شود

---

## 🔗 API Endpoints و نحوه استفاده

### 1. آپلود مستقیم فایل (File Upload)

**مناسب برای**: رابط‌های وب، برنامه‌های دسکتاپ

```bash
# استفاده با curl
curl -X POST "http://localhost:8003/transcribe" \
  -F "audio_file=@/path/to/your/audio.wav" \
  -F "language=fa" \
  -F "translate_to=en"

# یا از طریق API Gateway
curl -X POST "http://localhost:8000/api/v1/stt/transcribe" \
  -F "audio_file=@/path/to/audio.mp3" \
  -F "language=auto"
```

**پارامترها:**

- `audio_file`: فایل صوتی (اجباری)
- `language`: زبان منبع (`fa`, `en`, `auto`) - اختیاری
- `task`: `transcribe` یا `translate` - پیش‌فرض `transcribe`
- `translate_to`: زبان مقصد برای ترجمه (`fa`, `en`) - اختیاری

**پاسخ نمونه:**

```json
{
  "text": "سلام، این یک تست صوتی است",
  "language": "fa",
  "confidence": 0.95,
  "duration": 3.2,
  "processing_time": 1.8,
  "source_type": "file_upload",
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "سلام، این یک تست صوتی است",
      "confidence": 0.95
    }
  ],
  "translation": {
    "service": "google",
    "source_language": "fa",
    "target_language": "en",
    "translated_text": "Hello, this is an audio test",
    "success": true
  }
}
```

### 2. پردازش Base64 (موبایل اپ‌ها)

**مناسب برای**: اپلیکیشن‌های موبایل، PWA

```python
import base64
import requests

# تبدیل فایل صوتی به base64
with open("audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# ارسال درخواست
payload = {
    "audio_data": audio_base64,
    "filename": "audio.wav",
    "language": "fa",
    "format": "wav"
}

response = requests.post(
    "http://localhost:8003/transcribe-base64",
    json=payload,
    params={"translate_to": "en"}
)
```

### 3. دانلود از URL

**مناسب برای**: فایل‌های آنلاین، استریم‌ها

```python
import requests

payload = {
    "url": "https://example.com/audio.mp3",
    "language": "auto",
    "max_file_size": 50  # MB
}

response = requests.post(
    "http://localhost:8003/transcribe-url",
    json=payload
)
```

### 4. فایل محلی (سرور)

**مناسب برای**: فایل‌های داخل سرور

```python
import requests

payload = {
    "file_path": "/app/uploads/audio.wav",
    "language": "fa"
}

response = requests.post(
    "http://localhost:8003/transcribe-local",
    json=payload
)
```

### 5. پردازش دسته‌ای (Batch)

**مناسب برای**: پردازش چندین فایل همزمان

```bash
curl -X POST "http://localhost:8003/transcribe-batch" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.mp3" \
  -F "files=@audio3.m4a" \
  -F "translate_to=en"
```

---

## 🌐 ترجمه آنلاین

سرویس امکان ترجمه خودکار متن تبدیل شده را دارد:

```python
# ترجمه مستقل
payload = {
    "text": "سلام دنیا",
    "source_language": "fa",
    "target_language": "en",
    "service": "google"
}

response = requests.post(
    "http://localhost:8003/translate",
    json=payload
)
```

---

## 📊 مثال‌های عملی

### مثال 1: اپلیکیشن وب ساده

```html
<!DOCTYPE html>
<html>
  <body>
    <input type="file" id="audioFile" accept="audio/*" />
    <button onclick="transcribe()">تبدیل به متن</button>
    <div id="result"></div>

    <script>
      async function transcribe() {
        const fileInput = document.getElementById("audioFile");
        const file = fileInput.files[0];

        const formData = new FormData();
        formData.append("audio_file", file);
        formData.append("language", "fa");
        formData.append("translate_to", "en");

        const response = await fetch(
          "http://localhost:8000/api/v1/stt/transcribe",
          {
            method: "POST",
            body: formData,
          }
        );

        const result = await response.json();
        document.getElementById("result").innerHTML = `<p>فارسی: ${
          result.text
        }</p>
             <p>English: ${result.translation?.translated_text || "N/A"}</p>`;
      }
    </script>
  </body>
</html>
```

### مثال 2: اپلیکیشن موبایل (React Native)

```javascript
import { Audio } from "expo-av";
import * as FileSystem from "expo-file-system";

const recordAndTranscribe = async () => {
  // ضبط صدا
  const recording = new Audio.Recording();
  await recording.prepareToRecordAsync(
    Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY
  );
  await recording.startAsync();

  // توقف ضبط
  setTimeout(async () => {
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();

    // تبدیل به base64
    const base64 = await FileSystem.readAsStringAsync(uri, {
      encoding: FileSystem.EncodingType.Base64,
    });

    // ارسال به سرویس
    const response = await fetch(
      "http://your-server:8000/api/v1/stt/transcribe-base64",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_data: base64,
          filename: "recording.m4a",
          language: "fa",
          format: "m4a",
        }),
      }
    );

    const result = await response.json();
    console.log("نتیجه:", result.text);
  }, 5000);
};
```

### مثال 3: اسکریپت پایتون

```python
#!/usr/bin/env python3
import requests
import os
from pathlib import Path

def process_audio_folder(folder_path, output_file="results.txt"):
    """پردازش تمام فایل‌های صوتی در یک پوشه"""

    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    results = []

    for file_path in Path(folder_path).iterdir():
        if file_path.suffix.lower() in audio_extensions:
            print(f"پردازش {file_path.name}...")

            with open(file_path, 'rb') as f:
                files = {'audio_file': f}
                data = {'language': 'auto', 'translate_to': 'en'}

                response = requests.post(
                    'http://localhost:8000/api/v1/stt/transcribe',
                    files=files,
                    data=data
                )

                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        'file': file_path.name,
                        'text': result['text'],
                        'translation': result.get('translation', {}).get('translated_text', 'N/A'),
                        'confidence': result['confidence'],
                        'duration': result['duration']
                    })
                    print(f"✅ {file_path.name} پردازش شد")
                else:
                    print(f"❌ خطا در {file_path.name}: {response.text}")

    # ذخیره نتایج
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"=== {result['file']} ===\n")
            f.write(f"متن فارسی: {result['text']}\n")
            f.write(f"English: {result['translation']}\n")
            f.write(f"اعتماد: {result['confidence']:.2f}\n")
            f.write(f"مدت: {result['duration']:.1f}s\n\n")

    print(f"نتایج در {output_file} ذخیره شد")

if __name__ == "__main__":
    process_audio_folder("./audio_files")
```

---

## ⚙️ تنظیمات و بهینه‌سازی

### متغیرهای محیطی مهم

```bash
# اندازه مدل Whisper
WHISPER_MODEL_SIZE=base  # tiny=سریع, large=دقیق

# GPU
CUDA_VISIBLE_DEVICES=0

# محدودیت‌ها
MAX_FILE_SIZE_MB=25
MAX_AUDIO_DURATION=600
MAX_CONCURRENT_REQUESTS=5

# کش
CACHE_ENABLED=true
CACHE_TTL=3600

# ترجمه
TRANSLATION_SERVICE=google
```

### فایل پیکربندی (stt_config.yaml)

```yaml
stt_service:
  model:
    size: "base" # tiny, base, small, medium, large
    device: "auto" # auto, cpu, cuda

  audio:
    max_file_size: 25 # MB
    max_duration: 600 # seconds

  processing:
    normalize_audio: true
    remove_silence: true
    enhance_audio: true
```

---

## 🔍 مانیتورینگ و عیب‌یابی

### بررسی سلامت سرویس

```bash
# بررسی سلامت کلی
curl http://localhost:8003/health

# بررسی زبان‌های پشتیبانی شده
curl http://localhost:8003/languages

# از طریق API Gateway
curl http://localhost:8000/api/v1/stt/health
```

### لاگ‌های مفید

```bash
# مشاهده لاگ‌ها
tail -f /app/logs/stt.log

# یا در صورت اجرا مستقیم
python main.py  # لاگ‌ها در کنسول نمایش داده می‌شوند
```

### رفع مشکلات متداول

1. **خطای "Model not loaded"**

   ```bash
   # بررسی فضای دیسک
   df -h
   # دانلود مجدد مدل
   rm -rf /app/models && python main.py
   ```

2. **خطای "CUDA out of memory"**

   ```bash
   # استفاده از مدل کوچک‌تر
   export WHISPER_MODEL_SIZE=tiny
   # یا استفاده از CPU
   export WHISPER_MODEL_DEVICE=cpu
   ```

3. **خطای "File too large"**
   ```bash
   # افزایش محدودیت
   export MAX_FILE_SIZE_MB=50
   ```

---

## 🎯 نکات بهینه‌سازی

### برای بهترین کیفیت:

- از فرمت WAV با 16kHz استفاده کنید
- صدای واضح و بدون نویز ضبط کنید
- از `language=auto` برای تشخیص خودکار زبان
- مدل `large` برای دقت بالا

### برای بهترین سرعت:

- از مدل `tiny` یا `base` استفاده کنید
- فایل‌های کوتاه‌تر (زیر 30 ثانیه) ارسال کنید
- از GPU استفاده کنید
- کش را فعال نگه دارید

### برای برنامه‌های تولیدی:

- Rate limiting پیاده‌سازی کنید
- Authentication اضافه کنید
- لاگ‌گیری و مانیتورینگ تنظیم کنید
- HTTPS استفاده کنید

---

## 📈 مثال کاربردهای عملی

1. **ربات ضبط جلسات**: ضبط و متن‌سازی جلسات
2. **اپ یادگیری زبان**: تشخیص تلفظ و تصحیح
3. **سیستم پاسخگویی**: تبدیل سوالات صوتی به متن
4. **زیرنویس خودکار**: تولید زیرنویس برای ویدئوها
5. **دستیار صوتی**: پردازش فرمان‌های صوتی

این سرویس STT بسیار قدرتمند و انعطاف‌پذیر است و می‌تواند نیازهای مختلف پردازش گفتار به متن را پوشش دهد!

--------------------run
docker build -t stt-service .
