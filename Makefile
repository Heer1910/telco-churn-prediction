.PHONY: help setup test run clean download

help:
	@echo "Telco Customer Churn Pipeline"
	@echo ""
	@echo "  make setup      - Create venv and install dependencies"
	@echo "  make download   - Download Telco dataset"
	@echo "  make test       - Run unit tests"
	@echo "  make run        - Execute full pipeline"
	@echo "  make clean      - Remove generated outputs"

setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip setuptools wheel
	./venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "✅ Setup complete. Activate: source venv/bin/activate"

download:
	python download_data.py

test:
	pytest tests/ -v --tb=short

run:
	python run_pipeline.py

clean:
	rm -rf outputs/*
	rm -rf reports/*
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
