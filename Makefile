.PHONY: dev test lint train promote deploy

dev:
	docker-compose up --build

test:
	pytest tests/ -x --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

train:
	python scripts/dispatch_training.py --budget 50

promote:
	python scripts/evaluate_and_promote.py --threshold 0.02

deploy:
	./scripts/deploy.sh
