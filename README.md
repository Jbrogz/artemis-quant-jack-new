# Artemis Momentum Factor Book

Single-factor crypto momentum study for the Artemis Analytics competition. The
project builds an Artemis-sourced spot universe, forms momentum factor returns,
runs the statistical significance battery, runs the cost-aware backtest, and
assembles the final report artifacts.

Authoritative methodology:

- `docs/reference/factor-book-guide.md`
- `docs/specs/2026-05-30-artemis-momentum-design.md`
- Current handoff/status: `docs/STATUS.md`

## Verdict

The pre-registered skip=1 momentum family remains a statistical null for
deployment. The widened skip>=2 analysis is exploratory/post-hoc and found no
candidate that clears costs, out-of-sample performance, and the widened m=21
multiple-testing gate at the same time. Net recommendation: **NO-DEPLOY**.

## Reproduce

Full live reproduction requires the Artemis API key expected by the project
configuration. Add or update `ARTEMIS_API_KEY=<your key>` in your local secret
store or shell before running live data targets; never paste the key into chat
or commit it.

```bash
uv sync
make reproduce
```

`make reproduce` runs the full chain:

```text
universe -> returns -> factor -> stage2 -> backtest -> figures -> report -> writeup
```

To inspect the chain without running it:

```bash
make -n reproduce
```

Useful individual targets:

```bash
make universe
make returns
make factor
make stage2
make backtest
make figures
make report
make writeup
make test
make lint
```

## Reporting Provenance

The report builders are deterministic presentation assemblers. The PDF reads
the full Stage-2 variant table from `data/stats/significance.parquet`; the Word
writeup embeds verified literal tables for the key Stage-2 and Stage-4 results.
`tests/test_report_source_consistency.py` checks those embedded writeup tables
against `docs/STAGE2_RESULTS.md` and `docs/STAGE4_RESULTS.md` to reduce stale
report risk.

This short remediation does not fully refactor every narrative value to be
artifact-driven. Treat narrative and audit figures as verified literals tied to
the committed docs unless a future change wires them directly to regenerated
artifacts.
