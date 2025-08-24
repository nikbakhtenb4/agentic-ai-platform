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

Stop همه کانتینرهای در حال اجرا
docker stop $(docker ps -q)

لاگ گیری:
docker compose logs -f stt-service

docker compose logs api-gateway --tail=20
