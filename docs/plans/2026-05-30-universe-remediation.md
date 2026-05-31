# Universe Remediation Implementation Plan (rev 3)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Apply karpathy-guidelines (minimal real code, surgical, goal-driven TDD). This plan SUPERSEDES the universe portions of `2026-05-30-data-foundation-universe.md`. The scaffold, ported Artemis provider/cache (Tasks 0–1), the live probe (Task 2), and `coverage.first_seen_dates` (Task 3) are KEPT. We are fixing the universe so it is genuinely survivorship-sound on real Artemis data.

**Goal:** Replace the hand-curated universe with a programmatic, `artemis_id`-keyed Artemis enumeration; make liquidity computable from the volume Artemis actually serves; carry the death signal so collapses produce realized crash returns; and quantify the residual survivorship — until the 3-lens adversarial gate returns SOUND.

**Why (verified facts — see spec Appendix B):** the rev-2 build failed the survivorship gate. Live Artemis investigation found: `/asset` enumerates 1,013 assets keyed on a stable `artemis_id`; `30D_VOLUME` is real-time-only and `24H_VOLUME` is unreliable; deaths appear as price-decay-to-~0 (series continues) or ticker re-pointing; `lunc`→`terra` holds the real LUNA crash; DAY granularity only.

**Tech Stack:** Python 3.12 (uv-pinned), pandas, requests, pyarrow, pytest. Same repo: `/Users/jackbrogan/Desktop/artemis-quant-joint/new-artemis-work`.

**References (read first):** `docs/specs/2026-05-30-artemis-momentum-design.md` §3.5, §3.6, §4 Stage 1.1, §7, §10, Appendix B.

**Hard rules:** TDD (real tests first, watch fail, minimal impl, watch pass, commit). Never open/print `.env` — scripts load the key via `load_dotenv()`+`os.environ`, print only `len`. One-line commits, no Co-Authored-By trailer. No stubs/placeholders left behind.

---

## Task R0: Config corrections + Artemis asset registry

**Files:** Create `src/amom/universe/registry.py`, `tests/test_universe_registry.py`; Modify `src/amom/config.py`.

- [ ] **Config fixes:** in `config.py` — set `MARKET_METRICS = ("PRICE", "MC", "24H_VOLUME")` (drop the real-time-only `30D_VOLUME` and unused `FDMC`). Add: `ASSET_CATALOG_URL = "https://data-svc.artemisxyz.com/asset"`; `MIN_MC_USD = 10_000_000`; `MIN_MEDIAN_VOL_USD = 1_000_000`; `LIQUIDITY_VOL_WINDOW_DAYS = 30`; `MIN_OBS_DENSITY = 0.5` (≥50% of trailing-90d days have a price); `LISTING_STALENESS_DAYS = 7` (tradeability grace); `UNIVERSE_GRID_FREQ = "D"`. Keep `MIN_HISTORY_DAYS=90`, `MIN_ELIGIBLE_NAMES`, `MIN_BUCKET_SIZE`.
- [ ] **Test first** (`test_universe_registry.py`): monkeypatch the HTTP GET to return a canned `/asset` payload (a list of dicts with `artemis_id`,`symbol`,`coingecko_id`,`title`); assert `enumerate_assets()` returns a DataFrame with those columns, `artemis_id` unique, and that a missing `coingecko_id` becomes NaN (not a crash). Watch it FAIL.
- [ ] **Implement** `registry.enumerate_assets(url=ASSET_CATALOG_URL) -> pd.DataFrame` (GET, no key needed, parse the list; cache to `data/universe/asset_registry.parquet`). Watch it PASS.
- [ ] **Live check:** `uv run python -c "from amom.universe.registry import enumerate_assets; df=enumerate_assets(); print(len(df), df.artemis_id.is_unique)"` → expect ~1013 and True.
- [ ] **Commit:** `feat: Artemis /asset enumeration registry + config liquidity params`.

**Success:** universe membership comes from Artemis `/asset`, keyed on `artemis_id`; no hand-curated list; `MARKET_METRICS` no longer contains the dead `30D_VOLUME`.

## Task R1: Recycled-ticker splitter

**Files:** Create `src/amom/universe/recycle.py`, `tests/test_universe_recycle.py`.

- [ ] **Test first:** synthetic price series for one symbol that (a) decays to ~0 (≥90% drawdown from its peak) and then has a multi-month gap before resuming at a new regime → `split_recycled(price_panel)` yields TWO synthetic assets (`sym__seg0`, `sym__seg1`), `seg0` ending at the crash; (b) a continuous healthy series → ONE segment unchanged; (c) a series that crashes but never revives → ONE segment ending at the crash (not split). Watch FAIL.
- [ ] **Implement** `recycle.split_recycled(price_panel, *, drawdown_thresh=0.9, gap_days=45) -> price_panel` that detects terminal-drawdown-to-near-zero followed by a gap/new-regime and relabels post-revival rows as a new synthetic `artemis_id`. Pure, point-in-time-safe (splitting uses only the realized series). Watch PASS.
- [ ] **Commit:** `feat: recycled-ticker splitter (crash+gap -> distinct synthetic assets)`.

**Success:** a ticker reused by a new project never splices a healthy series onto a dead asset's crash; the dead segment is preserved with its collapse.

## Task R2: Liquidity + observation-density + tradeability (eligibility redesign)

**Files:** Modify `src/amom/universe/eligibility.py`; extend `tests/test_universe_eligibility.py`.

- [ ] **Tests first** (extend): (a) a coin with **2 volume prints of $5M** in the trailing 30d does NOT pass liquidity (its trailing-30d *median* 24H volume, and/or sum/30, is below `MIN_MEDIAN_VOL_USD`) — the rev-2 mean-of-present-rows bug must be gone; (b) broken sub-dollar prints in the window do not by themselves flip a genuinely liquid coin (median is robust); (c) `MC < MIN_MC_USD` fails even if volume passes; (d) a coin whose last price is older than `LISTING_STALENESS_DAYS` before `as_of` is INELIGIBLE (tradeability); (e) a coin with calendar-age ≥90d but obs-density < `MIN_OBS_DENSITY` fails. Watch FAIL.
- [ ] **Implement:** liquidity = `MC >= MIN_MC_USD` AND `median(24H_VOLUME over trailing window) >= MIN_MEDIAN_VOL_USD` (winsorize/ignore non-positive prints); add obs-density (use `n_obs` in trailing 90d) and staleness (latest price within `LISTING_STALENESS_DAYS`) checks — all point-in-time (data ≤ as_of). Watch PASS.
- [ ] **Commit:** `fix: liquidity via MC + median 24H volume; add obs-density and tradeability filters`.

**Success:** the ADV/liquidity bug is fixed; thin/dying/stale names are correctly excluded point-in-time; broken volume prints don't corrupt the gate.

## Task R3: Death signal in the universe panel

**Files:** Modify `src/amom/universe/builder.py`; extend `tests/test_universe_builder.py`.

- [ ] **Tests first:** `build_universe_history(...)` output panel includes `price_last_date` and a point-in-time `delisted_asof` boolean (= `as_of - price_last_date > LISTING_STALENESS_DAYS`, computed from data ≤ as_of). Assert a collapsed coin shows `delisted_asof=True` only on/after the date its reporting stops + grace, and never using future data. Watch FAIL.
- [ ] **Implement:** thread `price_last_date` (from `coverage.first_seen_dates`, recomputed as-of) and `delisted_asof` into the panel. Keep the min-universe gate, derive it from `MIN_BUCKET_SIZE`. Watch PASS.
- [ ] **Commit:** `feat: carry price_last_date + point-in-time delisted_asof into universe panel`.

**Success:** the returns layer has an explicit, point-in-time death signal to act on.

## Task R4: Spot returns + terminal crash-return imputation (closes the survivorship loop)

**Files:** Create `src/amom/returns/__init__.py`, `src/amom/returns/spot.py`, `tests/test_returns_spot.py`.

- [ ] **Tests first (the test that was vacuous, now real):** a fixture coin that crashes ~−95% and then stops reporting → `build_holding_returns(price_panel, universe_panel)` produces a holding-return series whose **realized terminal return ≈ −0.95** booked on the delisting date (from `delisted_asof`), NOT a dropped/NaN value. Also: simple daily returns for healthy coins; no funding term (spot). Watch FAIL.
- [ ] **Implement** `spot.build_holding_returns(...)`: simple spot price returns; on the `delisted_asof` transition, book the final realized return to (near) zero; log-vs-simple conventions per spec §4 Stage 1.2. Watch PASS.
- [ ] **Commit:** `feat: spot holding returns with terminal crash-return on delisting`.

**Success:** the guide §1.1 requirement — "a delisting after a 90% drop is a return of −90%, not a skipped value" — is now asserted on a real code path (the §7 survivorship row is satisfiable).

## Task R5: Daily-grid live build from the registry + survivorship quantification

**Files:** Modify `scripts/build_universe.py`; remove the `BROAD_CANDIDATES` dependency.

- [ ] **Rewrite** `build_universe.py` to: load the asset registry (R0), pull `PRICE`+`24H_VOLUME`+`MC` for all registry `artemis_id`s, run the recycled-ticker splitter (R1), build the panel on a **daily** grid (R3), write `data/universe/universe_history.parquet`. Print: rows, #assets, #ever-eligible, #eligible-on-latest, **#assets showing a terminal collapse (>90% drawdown to a sustained low)** as the quantified survivorship figure.
- [ ] **Run live** (api_ok was true). Report the printed stats. Verify `lunc`/`terra` carries the real LUNA crash in the returns panel.
- [ ] **Commit:** `feat: daily-grid universe build from Artemis registry + survivorship quantification`.

**Success:** the delivered panel is built from the Artemis enumeration on a daily grid, includes real collapses, and reports a quantified survivorship figure — no hand-curated list anywhere.

## Task R6: Test-strength + hygiene fixes

**Files:** `tests/test_universe_builder.py`, `src/amom/universe/eligibility.py`, `src/amom/providers/artemis.py` or builder.

- [ ] Strengthen no-look-ahead tests to be **per-date**: for each grid date `d`, randomize/drop all rows dated `> d` and assert `eligible`/`gated`/`delisted_asof` at `d` are unchanged vs baseline.
- [ ] Normalize panel dates (`.dt.normalize()`) at the ingest boundary so half-open `<=`/`>` window comparisons are robust to any intraday timestamp; add an intraday-bar boundary test.
- [ ] Either wire `eligible_mask` into the builder (so its anti-bias tests guard the real path) or delete it + its tests. No separately-tested-but-unused parallel eligibility.
- [ ] Disclose left-censoring at the pull-start (flag assets whose `price_first_date == pull_start` as censored/unknown true listing).
- [ ] **Commit:** `test: per-date no-look-ahead, date normalization, eligibility-path hygiene`.

**Success:** the no-look-ahead tests are discriminating; boundary assumptions are guarded; no dead code inflates anti-bias coverage.

---

## Verification gate (re-run, must reach SOUND)

Re-run the 3-lens adversarial review (survivorship · look-ahead/point-in-time · data-correctness) + synthesis against the corrected universe + the new `returns/` crash path, on synthetic fixtures AND the live parquet. Proceed to the rest of the project (factor grid → Stage-2 stats → backtest → report) ONLY when the synthesis returns **SOUND**. Any surviving critical/high finding loops back to a fix task.

## Self-Review

- **Must-fix coverage:** hardcoded list → R0/R5; recycled tickers → R1; ADV denominator → R2; death signal → R3; vacuous crash test → R4. Nice-to-haves: staleness/density → R2; daily grid → R3/R5; per-date look-ahead + normalization + eligible_mask + censoring → R6; MIN_BUCKET_SIZE gate → R3.
- **Scope:** R4 begins Stage 1.2 (returns) because the crash-return is the survivorship payoff and must be proven to close the gate — not scope creep.
- **Type consistency:** `artemis_id`, `price_last_date`, `delisted_asof`, `adv_*`/median-vol names used consistently across R0–R5.
