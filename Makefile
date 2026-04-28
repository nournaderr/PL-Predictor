.PHONY: install preprocess train test lint clean

install:
	poetry install

preprocess:
	poetry run python src/data/preprocess.py

train:
	poetry run python src/models/train.py

test:
	poetry run pytest tests/ --cov=src --cov-report=term-missing

lint:
	poetry run ruff check src/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete