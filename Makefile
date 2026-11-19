.PHONY: start run db-up db-down db-reset install seed

start: db-up
	uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

run:
	uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

db-up:
	docker compose up -d

db-down:
	docker compose down

db-reset:
	docker compose down -v
	docker compose up -d

install:
	uv sync

seed:
	uv run python seed.py
