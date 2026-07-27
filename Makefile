.PHONY: help setup lint smoke clean

help:
	@echo "make setup   - create the virtualenv and install pinned dependencies (uv)"
	@echo "make lint    - run ruff over the repository"
	@echo "make smoke   - import every entrypoint and dry-run every shipped config"
	@echo "make clean   - remove __pycache__, .DS_Store and experiment outputs"

setup:
	uv sync --extra dev
	@echo "Now: cp .env.example .env and fill in your API keys."

lint:
	uv run ruff check .

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	find . -name '.DS_Store' -delete
	rm -rf white_box_model_experiments/logs black_box_model_experiments/experiments
