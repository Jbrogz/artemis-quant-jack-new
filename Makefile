.PHONY: probe universe test

probe:
	uv run python scripts/probe_artemis.py

universe:
	uv run python scripts/build_universe.py

test:
	uv run pytest -q
