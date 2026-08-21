.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python3 -m src

debug:
	uv run python3 -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf data/output

lint:
	flake8 src --exclude=.venv,venv,__pycache__,.git
	mypy src --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run python3 -m flake8 .
	uv run python3 -m mypy . --strict