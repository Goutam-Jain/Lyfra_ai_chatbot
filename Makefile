.PHONY: up down logs build test dev-backend dev-frontend

## Start the whole stack: backend http://localhost:8000, frontend http://localhost:3000
up:
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example - add your API keys to it.")
	docker compose up -d --build
	@echo "Frontend: http://localhost:3000   Backend: http://localhost:8000/health"

## Stop the stack (the SQLite volume is kept)
down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build

## Backend tests (local Python 3.12 venv)
test:
	cd backend && python3 -m venv .venv && .venv/bin/pip install -q -r requirements-dev.txt && .venv/bin/python -m pytest

## Run the backend without Docker (reads ../.env, stores SQLite in backend/data/)
dev-backend:
	cd backend && python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt && \
	set -a && if [ -f ../.env ]; then . ../.env; fi && set +a && \
	DATABASE_URL=sqlite:///./data/app.db .venv/bin/uvicorn app.main:app --reload --port 8000

## Run the frontend without Docker (Vite dev server on :3000)
dev-frontend:
	cd frontend && npm install && npm run dev
