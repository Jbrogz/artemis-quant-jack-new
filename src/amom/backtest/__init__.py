"""Cost-aware backtest for the Artemis momentum factor (Stage 4, spec §4).

Spot-adapted: fees + size-scaled slippage, NO funding term (spec §3.1 / §4.2;
Artemis has no funding and this is a spot long/short book — disclosed). The
in-sample candidate is characterized net of these costs; the sealed OOS window
is spent exactly once in the Stage-4 runner. All transforms are pure /
walk-forward; no module here reads OOS-dated rows (the OOS-once discipline lives
in ``scripts/run_backtest.py``).
"""
