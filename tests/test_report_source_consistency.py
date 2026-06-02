"""Consistency tests for report/writeup source literals.

The PDF/docx builders intentionally embed some verified literals. These tests
make the highest-risk tables fail loudly when the committed source docs move.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_writeup_docx as writeup  # noqa: E402


def _clean_cell(cell: str) -> str:
    cell = cell.strip()
    cell = cell.replace("`", "").replace("**", "")
    cell = cell.replace("−", "-").replace("→", "->")
    return re.sub(r"\s+", " ", cell)


def _table_after_heading(markdown: str, heading: str) -> tuple[list[str], list[list[str]]]:
    lines = markdown.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == heading:
            break
    else:
        raise AssertionError(f"Heading not found: {heading}")

    table_lines: list[str] = []
    for line in lines[idx + 1 :]:
        if not line.startswith("|"):
            if table_lines:
                break
            continue
        table_lines.append(line)

    if len(table_lines) < 2:
        raise AssertionError(f"No markdown table found after: {heading}")

    def parse(line: str) -> list[str]:
        return [_clean_cell(part) for part in line.strip().strip("|").split("|")]

    header = parse(table_lines[0])
    rows = [parse(line) for line in table_lines[2:]]
    return header, rows


def test_writeup_stage2_tables_match_source_doc():
    source = (ROOT / "docs" / "STAGE2_RESULTS.md").read_text()

    family_header, family_rows = _table_after_heading(
        source, "## Selection family (skip = 1) — the deployment candidates"
    )
    diagnostics_header, diagnostics_rows = _table_after_heading(
        source, "## Diagnostics (skip {2,3}) — reported, not selected"
    )

    assert writeup.STAGE2_COLUMNS == family_header == diagnostics_header
    assert writeup.STAGE2_FAMILY == family_rows
    assert writeup.STAGE2_DIAGNOSTICS == diagnostics_rows


def test_writeup_stage4_tables_match_source_doc():
    source = (ROOT / "docs" / "STAGE4_RESULTS.md").read_text()

    primary_header, primary_rows = _table_after_heading(source, "### Primary L5d_S1d")
    comparator_header, comparator_rows = _table_after_heading(source, "### Comparator L28d_S1d")
    extra_header, extra_rows = _table_after_heading(
        source, "## Additional §4.5 net metrics (total return, avg win / loss)"
    )
    costs_header, costs_rows = _table_after_heading(source, "### 2x costs (in-sample)")
    lookback_header, lookback_rows = _table_after_heading(
        source, "### +/-50% lookback (in-sample, net)"
    )
    regime_header, regime_rows = _table_after_heading(
        source, "### Regime breakdown (in-sample, net mean return per regime)"
    )

    assert writeup.S4_COLUMNS == primary_header == comparator_header == costs_header
    assert writeup.S4_PRIMARY == primary_rows
    assert writeup.S4_COMPARATOR == comparator_rows
    assert writeup.S4_EXTRA_COLUMNS == extra_header
    assert writeup.S4_EXTRA == extra_rows
    assert writeup.S4_2X_COSTS == costs_rows
    assert writeup.S4_LOOKBACK_COLUMNS == lookback_header
    assert writeup.S4_LOOKBACK == lookback_rows
    assert writeup.S4_REGIME_COLUMNS == regime_header
    assert writeup.S4_REGIME == regime_rows
