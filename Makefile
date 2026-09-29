.PHONY: setup dev test train-quick evaluate docker-up docker-down migrate
setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	cd apps/web && npm install
dev:
	python -m uvicorn apps.api.main:app --reload
test:
	python -m pytest -q
	cd apps/web && npm run lint && npm run build
train-quick:
	python -m packages.marl.training --preset quick --algorithm qmix --seed 42
evaluate:
	python scripts/train_and_compare.py
migrate:
	alembic -c infra/alembic.ini upgrade head
docker-up:
	docker compose up --build -d
docker-down:
	docker compose down

