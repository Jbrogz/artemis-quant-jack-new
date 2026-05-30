# Data Foundation + Universe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Apply karpathy-guidelines (simplicity, surgical changes, goal-driven). Steps use checkbox (`- [ ]`) syntax. This is **agent-executed TDD**: where a step says "write the test," write the real test; where it says "implement," write the minimal real code to pass — no stubs left behind.

**Goal:** Build a sound, point-in-time, survivorship-bias-free crypto universe sourced entirely from Artemis, plus the data plumbing it sits on — the gated foundation everything else depends on.

**Architecture:** A uv-managed `amom` package. Port the battle-tested Artemis REST client + parquet cache from `src/cmom`. Probe the live Artemis API for coverage/history/dead-coin presence. Reconstruct each coin's listing date from its first Artemis price observation, then build a date-indexed eligibility panel that keeps delisted/collapsed coins and applies point-in-time history/liquidity/exclusion filters with a minimum-universe gate. All eligibility logic is unit-tested against synthetic fixtures (offline), then verified against live Artemis data.

**Tech Stack:** Python 3.11+, uv, pandas, requests, pyarrow (parquet), pytest, python-dotenv, ruff.

**Authoritative references (agents MUST read before coding):**
- Methodology guide: `docs/reference/factor-book-guide.md` (§1.1 universe, §1.2 returns are most relevant).
- Design spec: `docs/specs/2026-05-30-artemis-momentum-design.md` (§3, §4 Stage 1, §6 reuse manifest, §7 anti-bias matrix).
- Reuse sources (copy FROM here, in the parent repo): `../src/cmom/providers/{artemis,base}.py`, `../src/cmom/cache.py`, `../src/cmom/config.py`.

---

## File Structure

```
new-artemis-work/
  pyproject.toml                 # uv project, deps, ruff + pytest config
  .env.example                   # ARTEMIS_API_KEY=
  Makefile                       # probe, universe targets
  src/amom/
    __init__.py
    config.py                    # frozen constants (universe, grid, OOS rule, caps) — spec §4
    cache.py                     # ported parquet cache
    providers/__init__.py
    providers/base.py            # ported provider protocol + ProviderError + METRIC_COLUMNS
    providers/artemis.py         # ported Artemis REST client
    universe/__init__.py
    universe/coverage.py         # first-seen (listing) date reconstruction per symbol
    universe/eligibility.py      # point-in-time eligibility test for one (symbol, date)
    universe/builder.py          # assemble the date x symbol eligibility panel + min-universe gate
  scripts/
    probe_artemis.py             # live connectivity + coverage + dead-coin probe
    build_universe.py            # driver: pull market data -> coverage -> eligibility panel -> parquet
  tests/
    conftest.py                  # synthetic fixtures (incl. a known crash coin)
    test_providers_artemis.py
    test_cache.py
    test_universe_coverage.py
    test_universe_eligibility.py
    test_universe_builder.py
    integration/test_artemis_live.py
  data/                          # gitignored; cache/ + universe/ parquet land here
```

---

## Task 0: Scaffold the uv project

**Files:**
- Create: `pyproject.toml`, `.env.example`, `Makefile`, `src/amom/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`** with project name `amom`, Python ≥3.11, deps: `pandas`, `requests`, `pyarrow`, `python-dotenv`, `numpy`; dev deps: `pytest`, `pytest-cov`, `ruff`. Configure `[tool.pytest.ini_options]` with `testpaths=["tests"]` and `pythonpath=["src"]`; `[tool.ruff]` line-length 100.
- [ ] **Step 2: Create `.env.example`** containing exactly `ARTEMIS_API_KEY=`.
- [ ] **Step 3: Create `Makefile`** with `probe:` (`uv run python scripts/probe_artemis.py`) and `universe:` (`uv run python scripts/build_universe.py`) and `test:` (`uv run pytest -q`).
- [ ] **Step 4: `uv sync`** and verify the env builds. Run: `uv run python -c "import amom; print('ok')"` → expect `ok`.
- [ ] **Step 5: Commit** — `chore: scaffold amom uv project`.

---

## Task 1: Port the Artemis provider + cache

**Files:**
- Create: `src/amom/providers/base.py`, `src/amom/providers/artemis.py`, `src/amom/cache.py`, `src/amom/config.py`
- Test: `tests/test_providers_artemis.py`, `tests/test_cache.py`

Copy `../src/cmom/providers/base.py`, `../src/cmom/providers/artemis.py`, `../src/cmom/cache.py` verbatim, fixing the relative imports to the `amom` package. Into `config.py`, copy from `../src/cmom/config.py` ONLY: `ARTEMIS_BASE_URL`, `MARKET_METRICS`, `STABLECOINS`, `WRAPPED`, `EXCLUDED = STABLECOINS | WRAPPED`. (On-chain/dev metrics and sleeve params are NOT needed for momentum — do not copy them.)

- [ ] **Step 1: Write `test_cache.py`** — round-trip a small DataFrame through `cache_key`/`write_frame`/`read_frame`/`is_cached`; assert equality and that `is_cached` is True after write, False for an unknown key.
- [ ] **Step 2: Run → fail** (modules absent). `uv run pytest tests/test_cache.py -v`.
- [ ] **Step 3: Port `cache.py`** (fix imports). Run → pass.
- [ ] **Step 4: Write `test_providers_artemis.py`** — using `requests_mock` OR a monkeypatched `requests.get`, feed a canned Artemis JSON payload (`{"data":{"symbols":{"btc":{"PRICE":[{"date":"2024-01-01","val":42000.0}]}}}}`) and assert `_parse_response` yields a long frame with columns `[date,symbol,metric,value]`, `date` dtype `datetime64[ns]`, `value` numeric. Assert the string-sentinel case (`"Metric not available"`) is skipped, and assert `_redact` removes the API key from an error string.
- [ ] **Step 5: Run → fail.**
- [ ] **Step 6: Port `base.py` + `artemis.py`** (fix imports). Run → pass.
- [ ] **Step 7: Commit** — `feat: port Artemis provider and parquet cache into amom`.

**Success criteria:** provider + cache import cleanly under `amom`; parsing/redaction/caching unit-tested; no on-chain/sleeve code pulled in.

---

## Task 2: Live Artemis connectivity + coverage probe (CRITICAL GATE)

**Files:**
- Create: `scripts/probe_artemis.py`, `tests/integration/test_artemis_live.py`

This resolves the earlier 403 and proves the data needed for a sound universe exists. The script loads the key via `dotenv.load_dotenv()` then `os.environ["ARTEMIS_API_KEY"]` — it MUST NOT print the key (print only `len`).

- [ ] **Step 1: Write `scripts/probe_artemis.py`** that, via `ArtemisProvider`:
  1. Fetches `PRICE` for `btc` over `2013-01-01`..today → report first/last date + point count (history depth).
  2. Fetches `PRICE,MC,30D_VOLUME` for a sample of ~15 large caps → report which metrics return data.
  3. Probes **dead/collapsed coins** (`luna`, `lunc`, `ftt`, `ust`) → report whether each returns a price series and its last value (tests survivorship coverage).
  4. Attempts to enumerate the **broadest asset list** Artemis serves (try a supported-assets endpoint; if none, report the coverage of a large candidate probe set).
  Print a structured summary: `API_OK=<bool> HISTORY_START=<date> N_ASSETS=<int> DEAD_COIN_COVERAGE=<list>`.
- [ ] **Step 2: Run `uv run python scripts/probe_artemis.py`.** Expected: `API_OK=True`, a history start, dead coins returning series. **If 403/empty → STOP and report; the universe cannot be sound without data.**
- [ ] **Step 3: Write `tests/integration/test_artemis_live.py`** (marked `@pytest.mark.integration`, skipped if no key) asserting BTC price returns > 1000 daily points and at least one probed dead coin returns a non-empty series.
- [ ] **Step 4: Commit** — `feat: Artemis live connectivity and coverage probe`.

**Success criteria:** documented proof that Artemis serves daily prices with multi-year history AND retains at least some collapsed coins. The findings (history start, asset count, dead-coin coverage) are recorded for the spec's limitations section.

---

## Task 3: Listing-date (first-seen) reconstruction

**Files:**
- Create: `src/amom/universe/coverage.py`
- Test: `tests/test_universe_coverage.py`

`coverage.py` exposes `first_seen_dates(price_panel: pd.DataFrame) -> pd.DataFrame` returning columns `[symbol, price_first_date, price_last_date, n_obs]`, computed as the min/max non-NaN price date per symbol. This is the point-in-time listing proxy (guide §1.1: "reconstruct listing and delisting dates").

- [ ] **Step 1: Write `test_universe_coverage.py`** — fixture long price frame with `btc` (2020-01-01..2026-01-01) and `newcoin` (2025-01-01..2026-01-01); assert `first_seen_dates` returns `price_first_date[btc]=2020-01-01`, `[newcoin]=2025-01-01`, and correct `price_last_date`/`n_obs`. Add a coin with an internal NaN gap; assert first/last ignore NaNs.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `first_seen_dates`** (groupby symbol, min/max of dates where price notna, count).
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat: listing-date reconstruction from first Artemis price`.

---

## Task 4: Point-in-time eligibility (the anti-bias core)

**Files:**
- Create: `src/amom/universe/eligibility.py`
- Test: `tests/test_universe_eligibility.py`

`eligibility.py` exposes `is_eligible(symbol, as_of, *, first_date, adv_30d, excluded) -> bool` (and a vectorized `eligible_mask(as_of, coverage_df, adv_df, excluded) -> set[str]`). Rules (spec §4 Stage 1.1, guide §1.1), all evaluated **as-of** `as_of` only:
- history: `(as_of - first_date).days >= MIN_HISTORY_DAYS` (90),
- liquidity: `adv_30d >= MIN_ADV_USD` ($1M), where `adv_30d` is the trailing-30d mean USD volume computed from data ≤ `as_of`,
- exclusion: `symbol not in EXCLUDED` (stablecoins ∪ wrapped).

- [ ] **Step 1: Write `test_universe_eligibility.py`** with these explicit anti-bias cases:
```python
def test_history_filter_is_point_in_time():
    # coin first seen 2025-01-01; ineligible at 2025-02-01 (31d < 90d),
    # eligible at 2025-04-15 (>=90d). No future data consulted.
    assert is_eligible("c", as_of=ts("2025-02-01"), first_date=ts("2025-01-01"),
                       adv_30d=5e6, excluded=set()) is False
    assert is_eligible("c", as_of=ts("2025-04-15"), first_date=ts("2025-01-01"),
                       adv_30d=5e6, excluded=set()) is True

def test_liquidity_filter():
    assert is_eligible("c", as_of=ts("2025-04-15"), first_date=ts("2024-01-01"),
                       adv_30d=0.5e6, excluded=set()) is False  # below $1M

def test_stablecoins_and_wrapped_excluded():
    for sym in ("usdt", "wbtc"):
        assert is_eligible(sym, as_of=ts("2025-04-15"), first_date=ts("2020-01-01"),
                           adv_30d=1e9, excluded={"usdt","wbtc"}) is False

def test_no_lookahead_future_data_irrelevant():
    # eligibility at as_of must not change if data AFTER as_of is altered;
    # is_eligible takes only as-of inputs, so this is structural — assert the
    # function signature accepts no future-dated series.
    ...
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `eligibility.py`** — pure, no I/O, operates only on as-of scalars/series.
- [ ] **Step 4: Run → pass.** Add a `eligible_mask` test over a small coverage+ADV frame.
- [ ] **Step 5: Commit** — `feat: point-in-time eligibility filters (history, liquidity, exclusions)`.

**Success criteria:** eligibility is a pure function of as-of-date inputs; history filter is point-in-time; stablecoins+wrapped excluded; tests prove no future data is consulted.

---

## Task 5: Universe panel builder + survivorship + min-universe gate

**Files:**
- Create: `src/amom/universe/builder.py`
- Test: `tests/test_universe_builder.py`

`builder.py` exposes `build_universe_history(price_panel, volume_panel, *, dates, excluded) -> pd.DataFrame` returning a long panel `[date, symbol, eligible(bool), adv_30d]` covering **every symbol ever seen** (including collapsed ones) across `dates`, applying Task-4 eligibility per (symbol, date), plus a **point-in-time minimum-universe gate**: a date where `eligible.sum() < MIN_ELIGIBLE_NAMES` (so quintiles would be < `MIN_BUCKET_SIZE`) is marked `gated=True` (rebalance later skipped), using only as-of info.

- [ ] **Step 1: Write `test_universe_builder.py`** with the survivorship test as the centerpiece:
```python
def test_collapsed_coin_stays_in_panel_with_final_return():
    # 'deadcoin' has prices to 2024-06-01 then crashes ~95% and stops.
    # It MUST appear in the panel while eligible, and its final observed
    # return must reflect the crash (handled downstream in returns), NOT be
    # dropped. Assert deadcoin rows exist on dates it was eligible and that
    # it is absent (not silently kept eligible) after its last price + grace.
    panel = build_universe_history(price_panel, vol_panel, dates=dates, excluded=set())
    assert (panel.query("symbol=='deadcoin' and eligible")).shape[0] > 0

def test_eligibility_rebuilt_each_date_and_point_in_time():
    # a coin crossing the 90d threshold becomes eligible exactly once it has
    # 90d history as of that date, not before.
    ...

def test_min_universe_gate_is_point_in_time():
    # early date with < MIN_ELIGIBLE_NAMES eligible -> gated True.
    ...
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `builder.py`** — compute trailing-30d ADV per symbol/date from `volume_panel` (≤ date), call eligibility, assemble the panel, apply the gate. No future data in any per-date computation.
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Write `scripts/build_universe.py`** — pull `PRICE` + `30D_VOLUME` (or `24H_VOLUME` → rolling 30d) for the broadest asset list via the provider, build the panel, write `data/universe/universe_history.parquet`. Run it on live data; print row count, # symbols, # ever-eligible, # delisted-ever, eligible-on-latest-date.
- [ ] **Step 6: Commit** — `feat: point-in-time universe panel with survivorship and min-universe gate`.

**Success criteria (the gate the whole project waits on):** the panel includes collapsed/delisted coins on the dates they were eligible; eligibility is rebuilt daily and is strictly point-in-time; the min-universe gate uses as-of info only; the live build produces a plausible panel (hundreds of symbols, a non-trivial delisted-ever count).

---

## Self-Review

- **Spec coverage:** §3.1/§3.4 exclusions (Task 1 config, Task 4), §4 Stage 1.1 universe incl. dead coins + point-in-time + min-gate (Tasks 3–5), §6 reuse manifest (Task 1), §7 matrix rows for survivorship/point-in-time/min-gate/no-look-ahead (Tasks 4–5 tests). Returns (§Stage 1.2), the momentum signal/formation (§Stage 1.3–1.4), and Stages 2–5 are **deliberately deferred to later plans** — this plan is the gated foundation only.
- **Placeholders:** the `test_no_lookahead_future_data_irrelevant` and a couple of secondary tests are described rather than fully coded because they assert structural/no-I/O properties; the executing agent writes the concrete assertions. All correctness-critical anti-bias tests (history point-in-time, survivorship retention, exclusions, min-gate) have concrete code.
- **Type consistency:** `first_date`/`price_first_date`, `adv_30d`, `eligible`, `gated` names are consistent across Tasks 3–5.
