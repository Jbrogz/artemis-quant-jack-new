# amom — Artemis Momentum Factor Book

**New window? READ `docs/STATUS.md` FIRST** — it has full project state, the verified findings, and the open work item.

Single-factor (**momentum only**) crypto long/short factor book, **all data from Artemis**, following `docs/reference/factor-book-guide.md` strictly. Package `amom`. Repo: `github.com/Jbrogz/artemis-quant-jack-new` (private).

## Status (2026-05-31)
Complete through Stage 5, every stage adversarially verified SOUND, 250 tests passing. Deliverables: `docs/report/Artemis_Momentum_Report.pdf` + `Artemis_Momentum_Findings.docx`. **Verified finding: momentum is a rigorous statistical NULL in the pre-registered family → NO-DEPLOY.**

**Resolved (no open work item):** the strong `skip≥2` variants were validated as a widened/post-hoc m=21 family with each OOS spent once — **no deployable candidate** (the m=21-robust survivors L3d/S3d & L14d/S3d fail OOS; the OOS-positive L5d/S2d, L5d/S3d, L1d/S3d are not multiple-testing-robust). The pre-registered skip=1 NULL stands as the headline. See `docs/STATUS.md` and the widened sections of `docs/STAGE2_RESULTS.md` / `docs/STAGE4_RESULTS.md`.

## Hard rules (do not violate)
- **`.env` is off-limits** — never open/cat/grep/print it; scripts load `ARTEMIS_API_KEY` via python-dotenv and print only `len`.
- **No look-ahead** is the cardinal rule (decision at t uses only data ≤ close t; execute t+1 close; tests must be discriminating).
- **TDD** (real test → fail → minimal impl → pass → commit), karpathy-guidelines, honest reporting (HAC not naive; failures included; gross-vs-net & IS-vs-OOS side by side; never massage toward significance).
- Build via **gated multi-agent workflows** + 3-lens adversarial verification to SOUND before proceeding; model-delegate (opus = correctness/verification, sonnet = mechanical).
- **Push gotcha:** run `gh auth switch -u Jbrogz` before every `git push` (the macOS keychain may serve a different credential → plain push 404s).
- Commits: one-line, no Co-Authored-By trailer. Python via `uv`.
