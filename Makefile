.PHONY: help install dev test lint format docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install      - Install runtime and development dependencies"
	@echo "  make dev          - Run development server with auto-reload"
	@echo "  make test         - Run test suite with coverage report"
	@echo "  make lint         - Run ruff linter and mypy type checks"
	@echo "  make format       - Format code using ruff"
	@echo "  make docker-build - Build production multi-stage Docker container"
	@echo "  make docker-up    - Run vision service and Prometheus via docker-compose"
	@echo "  make docker-down  - Stop all running containers"

install:
	pip install -r requirements.txt -r requirements-dev.txt

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	ruff check app tests
	mypy app

format:
	ruff format app tests

docker-build:
	docker build -t claimsight-vision:latest .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
