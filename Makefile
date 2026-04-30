.PHONY: install construct validate clean-data enrich eda pipeline train test lint clean

install:
	poetry install

# --- Data Pipeline ---

construct:
	poetry run python src/data/construct.py \
		--input data/raw \
		--output data/processed/dataset.csv

validate:
	poetry run python src/data/validate.py \
		--input data/processed/dataset.csv \
		--output data/processed/validation_report.txt

clean-data:
	poetry run python src/data/clean.py \
		--input data/processed/dataset.csv \
		--output data/processed/cleaned_dataset.csv

enrich:
	poetry run python src/features/enrich.py \
		--input data/processed/cleaned_dataset.csv \
		--output data/processed/enriched_dataset.csv

eda:
	poetry run python src/visualization/eda.py \
		--input data/processed/enriched_dataset.csv \
		--output data/processed/vis.pdf

feature-selection:
	poetry run python src/features/preprocess.py \
		--input data/processed/enriched_dataset.csv \
		--output data/processed/
pipeline: construct validate clean-data enrich eda feature-selection

# --- ML ---

# train:
# 	poetry run python src/models/train.py

# # --- Dev ---

# test:
# 	poetry run pytest tests/ --cov=src --cov-report=term-missing

# lint:
# 	poetry run ruff check src/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete