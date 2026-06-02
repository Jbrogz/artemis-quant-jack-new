.PHONY: probe universe returns factor stage2 backtest figures report writeup test lint reproduce

probe:
	uv run python scripts/probe_artemis.py

universe:
	uv run python scripts/build_universe.py

returns: universe
	uv run python scripts/build_returns.py

factor: returns
	uv run python scripts/build_factor_returns.py

stage2: factor
	uv run python scripts/run_stage2.py

backtest: stage2
	uv run python scripts/run_backtest.py

figures: backtest
	uv run python scripts/build_report_figures.py

report: figures
	uv run python scripts/build_report.py

writeup: report
	uv run python scripts/build_writeup_docx.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts

reproduce: writeup
