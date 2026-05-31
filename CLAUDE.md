# amom — Artemis Momentum Factor Book

**New window? READ `docs/STATUS.md` FIRST** — it has full project state, the verified findings, and the open work item.

Single-factor (**momentum only**) crypto long/short factor book, **all data from Artemis**, following the Project 1 Factor Book Guide methodology strictly. Package `amom`. Repo: `github.com/Jbrogz/artemis-quant-jack-new` (private).

## Status (2026-05-31)
Complete through Stage 5, every stage adversarially verified SOUND, 226 tests passing. Deliverables: `docs/report/Artemis_Momentum_Report.pdf` + `Artemis_Momentum_Findings.docx`. **Verified finding: momentum is a rigorous statistical NULL in the pre-registered family → NO-DEPLOY.**

**Open work item:** validate the strong `skip≥2` variants (L3d/S3d HAC t=5.0, sign-stable, DSR 0.98, etc.) as a widened/post-hoc family — see `docs/plans/2026-05-31-skip-variant-validation.md`.

## Hard rules (do not violate)
- **`.env` is off-limits** — never open/cat/grep/print it; scripts load `ARTEMIS_API_KEY` via python-dotenv and print only `len`.
- **No look-ahead** is the cardinal rule (decision at t uses only data ≤ close t; execute t+1 close; tests must be discriminating).
- **TDD** (real test → fail → minimal impl → pass → commit), karpathy-guidelines, honest reporting (HAC not naive; failures included; gross-vs-net & IS-vs-OOS side by side; never massage toward significance).
- Build via **gated multi-agent workflows** + 3-lens adversarial verification to SOUND before proceeding; model-delegate (opus = correctness/verification, sonnet = mechanical).
- **Push gotcha:** run `gh auth switch -u Jbrogz` before every `git push` (macOS keychain serves a `JB-acap` credential → plain push 404s).
- Commits: one-line, no Co-Authored-By trailer. Python via `uv`.
