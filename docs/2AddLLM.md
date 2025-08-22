-------------Docker Compose Container:------------

سرویس جدید llm-service:

کانتینر در پورت 8002 اجرا می‌شود
مدل gpt2-fa از مسیر /data/models/llm/gpt2-fa mount می‌شود
متغیرهای محیطی مخصوص مدل زبانی تنظیم شده
محدودیت حافظه (4GB حداکثر، 2GB رزرو) برای مدیریت بهتر منابع
Health check با timeout بیشتر (30 ثانیه) برای مدل‌های سنگین

تغییرات در سرویس‌های موجود:

در api-gateway و test-service متغیر LLM_SERVICE_URL اضافه شد
وابستگی به llm-service در سرویس‌های مربوطه تعریف شد

نکات مهم:

Volume مدل به صورت read-only mount شده (:ro)
Health check برای LLM service زمان بیشتری دارد چون لود مدل زمان‌بر است
محدودیت حافظه برای جلوگیری از مصرف بیش از حد منابع تنظیم شده

حالا می‌توانید Dockerfile و کد سرویس LLM را بسازید و کانتینر را اجرا کنید.

-------------nginx.conf--------------------
nginx.conf به‌روزرسانی شد. تغییرات اصلی:
🔧 تغییرات کلیدی:

Upstream جدید: llm_service برای سرویس LLM اضافه شد
Rate Limiting:

llm_limit با محدودیت 2 درخواست در ثانیه (کمتر از API عادی)
Burst تا 5 درخواست برای LLM

مسیرهای جدید:

/api/llm/ → هدایت به llm-service
/llm/health → health check مخصوص LLM

تنظیمات خاص LLM:

Timeout بیشتر: 120 ثانیه برای پردازش مدل‌های سنگین
Buffer بزرگ‌تر: برای پاسخ‌های طولانی (128k/256k)
WebSocket Support: برای streaming responses

🎯 مسیرهای در دسترس:

GET /llm/health → بررسی سلامت سرویس LLM
POST /api/llm/generate → تولید متن
POST /api/llm/chat → گفتگو با مدل
سایر endpoint های LLM

⚡ بهینه‌سازی عملکرد:

Timeout های مناسب برای درخواست‌های سنگین
Buffer های بزرگ برای پاسخ‌های طولانی
Rate limiting مناسب برای محافظت از منابع

حالا nginx آماده کار با سرویس LLM است!

----------------------------prometheus.yml---------------
prometheus.yml به‌روزرسانی شد. تغییرات اصلی:
🔧 سرویس جدید اضافه شده:

llm-service job:

Target: llm-service:8002
Scrape interval: 60 ثانیه (بیشتر از سرویس‌های عادی)
Timeout: 30 ثانیه (برای metrics سنگین)
Params خاص برای metrics مدل زبانی

llm-health job:

مانیتورینگ جداگانه برای سلامت سرویس
Interval: 120 ثانیه (هر 2 دقیقه)
Timeout: 45 ثانیه

📊 Metrics مورد انتظار از LLM Service:

model_inference_duration - زمان استنتاج
model_memory_usage - مصرف حافظه
model_requests_total - تعداد درخواست‌ها
model_errors_total - تعداد خطاها
gpu_utilization - استفاده از GPU (اگر وجود داشته باشد)
model_queue_size - اندازه صف درخواست‌ها

⚡ بهینه‌سازی:

فواصل زمانی بیشتر برای جلوگیری از اختلال در کارکرد مدل
Timeout های مناسب برای عملیات سنگین
جداسازی health check از metrics اصلی

حالا Prometheus آماده مانیتور کردن سرویس LLM است!

---------------kole file haie llm-service ra ba version ghabl avaz kardam o saboktar kardam-------------------

----------------services/api-gateway/routes/llm.py--------------------
services/api-gateway/routes/llm.py داریم و جدول پورت‌ها را نیز اصلاح :
ه:

services/api-gateway/routes/llm.py - Route handler برای LLM endpoints
جدول پورت‌ها - به‌روزرسانی شده با LLM Service

🔧 ویژگی‌های Route Handler:

Health Check - بررسی وضعیت سرویس
Text Generation - تولید متن با پارامترهای قابل تنظیم
Chat Interface - گفتگوی تعاملی
Model Info - اطلاعات مدل
Batch Processing - پردازش دسته‌ای (آماده برای آینده)
Error Handling - مدیریت خطاهای مختلف
Logging - ثبت درخواست‌ها و عملکرد
Timeout Management - مدیریت زمان انتظار

📡 نکته مهم:
برای فعال‌سازی این route ها، باید در فایل اصلی API Gateway (معمولاً main.py) این router را import و اضافه کنید:

---------------------فایل main.py اصلاح دوباره---------------
بخاطر api gateway routes:
ه:
🔧 سازگاری با API Gateway:

Chat Endpoint جدید:

POST /chat - گفتگوی تعاملی با مدیریت session
Chat session management با UUID
Context awareness برای مکالمات

Response Models بهبود یافته:

GenerationResponse حالا single string برمی‌گرداند (نه list)
ChatResponse مطابق با API Gateway
BatchGenerationResponse برای پردازش دسته‌ای

Batch Processing:

POST /batch/generate - پردازش همزمان چندین درخواست
محدودیت 10 درخواست در هر batch

📊 ویژگی‌های جدید:

Chat Session Management:

ذخیره مکالمات در حافظه
Auto-cleanup sessions قدیمی (بعد از 1 ساعت)
Context window برای مکالمات (آخرین 5 پیام)

Enhanced Monitoring:

/health/metrics - metrics خاص برای Prometheus
CHAT_SESSIONS gauge برای تعداد session های فعال
بهتر شدن memory و GPU monitoring

API Endpoints اضافی:

DELETE /chat/{conversation_id} - حذف session
بهبود /model/info با اطلاعات کامل‌تر

🎯 مطابقت کامل:

همه Response/Request models مطابق با API Gateway
Error handling یکسان
Timeout ها و محدودیت‌های مشابه
Metrics integration کامل

**\*\*\*\***\***\*\*\*\*** فایل های زیر اختیاری **\*\***\*\***\*\***
این فایل ها اختیاری است و برای فاز بعدی در نظر میگیریم:
shared/models/request.py - اضافه کردن مدل‌های درخواست LLM
shared/models/response.py - اضافه کردن مدل‌های پاسخ LLM

configs/models/llm_configs.yaml - تنظیمات مدل‌ها
llm-service/config/model_configs.yaml - کپی از تنظیمات
llm-service/utils/model_cache.py - کش مدل‌ها

این ها در فولدر جدای faz2/extra میذاریم:

---------هدف کلی فایل main.py---------
این فایل، سرویس LLM را با استفاده از FastAPI پیاده‌سازی کرده که می‌تونه:

مدل زبان فارسی (مثل GPT2-fa) رو بارگذاری کنه

متن تولید کنه (/generate)

حالت گفت‌وگویی داشته باشه (/chat)

وضعیت سلامت رو ارائه بده (/health, /metrics)

-----------------تست--------------
در cmd:
curl -X POST http://localhost:8002/generate -H "Content-Type: application/json" -d "{\"prompt\": \"چیک شعر عاشقانه چهار بیتی در سبک حافظ بنویس\", \"max_length\": 400, \"temperature\":0.7}"

----todo:
model gpt2-fa aslan khob javab nemide , bejash ieki dige bezar
