.PHONY: start stop db-reset install seed deploy destroy

start:
	docker compose up -d --build

stop:
	docker compose down

db-reset:
	docker compose down -v
	docker compose up -d --build

install:
	uv sync

seed:
	docker compose exec app uv run --no-dev python seed.py

deploy:
	./deploy.sh

destroy:
	./destroy.sh
