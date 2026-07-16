.PHONY: probe universe figures report writeup test

probe:
	uv run python scripts/probe_artemis.py

universe:
	uv run python scripts/build_universe.py

figures:
	uv run python scripts/build_report_figures.py

report:
	uv run python scripts/build_report.py

writeup:
	uv run python scripts/build_writeup_docx.py

test:
	uv run --extra dev pytest -q
