ساخت ایمیج
docker-compose build --no-cache llm-service
اجرا کانتینر
docker-compose up -d llm-service

کانتینرهای در حال اجزا:
docker ps

ریستارت:
docker-compose restart llm-service

لاگ:
docker logs -f agentic-llm-service
