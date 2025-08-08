# Agentic AI Platform

یک پلتفرم جامع و مقیاس‌پذیر برای ایجاد و مدیریت سیستم‌های هوش مصنوعی با قابلیت‌های پیشرفته.

## 🚀 ویژگی‌های کلیدی

- **API Gateway مرکزی** - مدیریت و روتینگ تمام درخواست‌ها
- **معماری Microservices** - قابلیت توسعه و مقیاس‌پذیری بالا
- **پایگاه داده چندگانه** - PostgreSQL، Redis، Vector DB
- **مانیتورینگ پیشرفته** - Prometheus & Grafana
- **امنیت کامل** - JWT، OAuth2، Rate Limiting
- **Docker و Kubernetes** - آماده برای production

## 🏗️ معماری سیستم

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Nginx       │    │   API Gateway    │    │   Test Service  │
│  Load Balancer  │────│   FastAPI        │────│    FastAPI      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼────┐  ┌──────▼──────┐ ┌─────▼─────┐
        │PostgreSQL  │  │    Redis    │ │Prometheus │
        │  Database  │  │    Cache    │ │ Metrics   │
        └────────────┘  └─────────────┘ └───────────┘
```

## 📋 پیش‌نیازها

- **Docker** v20.10+
- **Docker Compose** v2.0+
- **Python** 3.11+ (برای development)
- **حداقل 4GB RAM**
- **حداقل 10GB فضای خالی**

## 🔧 نصب و راه‌اندازی

### 1. دانلود پروژه

```bash
git clone <repository-url>
cd agentic-ai-platform
```

### 2. ایجاد فایل‌های مورد نیاز

فایل‌های زیر را در root پوشه قرار دهید:

- `docker-compose.yml`
- `nginx.conf`
- `.env`
- `monitoring/prometheus.yml`

### 3. راه‌اندازی خودکار

```bash
python scripts/setup.py
```

### 4. راه‌اندازی دستی

```bash
# ایجاد دایرکتوری‌ها
mkdir -p data/{models,vectors,cache,logs}
mkdir -p services/{api-gateway,test-service}
mkdir -p shared/database

# ساخت و راه‌اندازی
docker compose build
docker compose up -d
```

## 🌐 دسترسی به سرویس‌ها

| سرویس            | URL                   | توضیحات            |
| ---------------- | --------------------- | ------------------ |
| **پلتفرم اصلی**  | http://localhost      | ورودی اصلی         |
| **API Gateway**  | http://localhost:8000 | API مرکزی          |
| **Test Service** | http://localhost:8001 | سرویس تست          |
| **Prometheus**   | http://localhost:9090 | مانیتورینگ metrics |
| **Grafana**      | http://localhost:3000 | داشبورد مانیتورینگ |

### اطلاعات ورود Grafana

- **نام کاربری:** admin
- **رمز عبور:** admin

## 🧪 تست سیستم

### بررسی وضعیت سرویس‌ها

```bash
curl http://localhost/health
```

### تست اتصالات

```bash
curl http://localhost/api/test-connection
```

### تست سرویس آزمایشی

```bash
curl http://localhost/test/ping
```

### مشاهده لاگ‌ها

```bash
docker compose logs -f
```

## 📁 ساختار پروژه

```
agentic-ai-platform/
├── docker-compose.yml          # تنظیمات Docker
├── nginx.conf                  # تنظیمات Nginx
├── .env                       # متغیرهای محیطی
├── services/
│   ├── api-gateway/           # API Gateway اصلی
│   └── test-service/          # سرویس تست
├── monitoring/
│   └── prometheus.yml         # تنظیمات Prometheus
├── shared/
│   └── database/             # اسکریپت‌های پایگاه داده
├── scripts/
│   └── setup.py              # اسکریپت راه‌اندازی
└── data/                     # داده‌ها و فایل‌ها
```

## 🔍 API Documentation

### Endpoints اصلی

| Method | Path                   | توضیحات            |
| ------ | ---------------------- | ------------------ |
| `GET`  | `/`                    | اطلاعات کلی پلتفرم |
| `GET`  | `/health`              | بررسی وضعیت سلامت  |
| `GET`  | `/api/info`            | اطلاعات API        |
| `GET`  | `/api/test-connection` | تست اتصالات        |
| `GET`  | `/test/ping`           | تست سرویس          |

### نمونه Response

```json
{
  "message": "Agentic AI Platform",
  "version": "1.0.0",
  "status": "running",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🛠️ دستورات مفید

### مدیریت سرویس‌ها

```bash
# راه‌اندازی
docker compose up -d

# توقف
docker compose down

# بازسازی
docker compose build --no-cache

# مشاهده وضعیت
docker compose ps

# مشاهده لاگ‌ها
docker compose logs -f [service-name]
```

### دیباگ و عیب‌یابی

```bash
# ورود به کانتینر
docker compose exec api-gateway bash

# بررسی منابع
docker stats

# پاک‌سازی
docker system prune -a
```

## 📊 مانیتورینگ

### Prometheus Metrics

- **API Requests:** کل درخواست‌های API
- **Response Times:** زمان پاسخ‌دهی
- **Error Rates:** نرخ خطاها
- **System Resources:** منابع سیستم

### Grafana Dashboards

- **System Overview:** نمای کلی سیستم
- **API Performance:** عملکرد API
- **Database Metrics:** metrics پایگاه داده

## 🔐 امنیت

### فعلی

- **Rate Limiting:** محدودیت تعداد درخواست
- **CORS Protection:** محافظت از درخواست‌های متقابل
- **Security Headers:** هدرهای امنیتی
- **Health Checks:** بررسی سلامت سرویس‌ها

### آینده (در فازهای بعدی)

- **JWT Authentication**
- **OAuth2 Integration**
- **API Key Management**
- **Role-Based Access Control**

## 🚧 مراحل توسعه

### ✅ فاز 1 - پایه (فعلی)

- [x] API Gateway بیسیک
- [x] Test Service
- [x] Docker Compose
- [x] Nginx Load Balancer
- [x] PostgreSQL & Redis
- [x] Prometheus & Grafana
- [x] Health Checks
- [x] LLM Service (gpt2-fa)

### 🔄 فاز 2 - هوش مصنوعی

- [ ] Text-to-Speech Service
- [ ] Speech-to-Text Service
- [ ] Vector Database Integration

### 🔄 فاز 3 - احراز هویت

- [ ] JWT Authentication Service
- [ ] User Management
- [ ] API Key System
- [ ] Permission System

### 🔄 فاز 4 - ابزارها

- [ ] Tools Service
- [ ] File Handler
- [ ] Code Executor
- [ ] External API Integrations

## 🆘 عیب‌یابی

### مشکلات رایج

**سرویس‌ها راه‌اندازی نمی‌شوند:**

```bash
# بررسی لاگ‌ها
docker compose logs

# بررسی پورت‌ها
netstat -tulpn | grep :80
```

**خطای اتصال به پایگاه داده:**

```bash
# بررسی وضعیت PostgreSQL
docker compose exec postgres pg_isready
```

**خطای Redis:**

```bash
# تست Redis
docker compose exec redis redis-cli ping
```

## 🤝 مشارکت

1. Fork کنید
2. Branch جدید بسازید (`git checkout -b feature/amazing-feature`)
3. تغییرات را commit کنید (`git commit -m 'Add amazing feature'`)
4. Push کنید (`git push origin feature/amazing-feature`)
5. Pull Request ایجاد کنید

## 📝 لایسنس

این پروژه تحت لایسنس MIT منتشر شده است.

## 📞 پشتیبانی

برای سوالات و مشکلات:

- **Issues:** GitHub Issues
- **Documentation:** `/docs` folder
- **API Docs:** http://localhost:8000/docs (بعد از راه‌اندازی)

---

**🎉 موفق باشید در ساخت سیستم هوش مصنوعی خود!**

python scripts/setup.py
تست کن:

docker compose up -d
http://localhost - صفحه اصلی
http://localhost/health - بررسی سلامت
http://localhost:3000 - Grafana (admin/admin)

اطلاعات ورود پیش‌فرض Grafana:
URL: http://localhost:3000

Username: admin

Password: admin

---

docker compose stop

--------------run script:
python scripts/setup.py
