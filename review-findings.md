## REVIEW CLEAN
## Multi-Persona Review: shifaa (all core modules)
### Date: 2026-04-08
### Summary: 1 P0, 6 P1, 4 P2 — ALL P0+P1 FIXED (46/46 tests pass)

---

#### P0 -- Critical

- **P0-1** [Software Engineer]: Silent exception swallowing in fit_poisson_model() — catches all exceptions and returns None with no logging (analysis/regression.py:20-21)
  - Suggested fix: Log the exception to stderr before returning None. Distinguish convergence failure from bad data.

#### P1 -- Important

- **P1-1** [Statistical]: Crosswalk matching bidirectional substring can produce false positives — "HIV" matches "CHIV", "TB" matches "STUB" (crosswalk/loader.py:20)
  - Suggested fix: Require word-boundary matching or minimum term length (3+ chars for substring).

- **P1-2** [Software Engineer]: compute_lorenz() crashes on empty array — cumulative[-1] raises IndexError when values is empty (analysis/equity_trend.py:21)
  - Suggested fix: Add early return for empty input, returning ([0,1], [0,1]).

- **P1-3** [Software Engineer]: read_wb_indicators() returns None if no Parquet found, but build_shifaa_matrix() assumes DataFrame — merge() will crash (lakehouse/reader.py:31, join.py:11)
  - Suggested fix: Return empty DataFrame with correct columns instead of None.

- **P1-4** [Statistical]: REI falls back to raw DALYs when population is missing, without warning — produces incomparable rates (lakehouse/join.py:16-22)
  - Suggested fix: Add stderr warning when population is missing for any rows.

- **P1-5** [Statistical]: extract_coefficients() assumes statsmodels column names "[0.025" and "0.975]" — fragile to API changes (analysis/regression.py:25-29)
  - Suggested fix: Use conf_int() method instead of parsing summary table.

- **P1-6** [Software Engineer]: Session lifecycle inconsistent — search_studies_by_condition() creates session if None but never closes it (ctgov/api.py:41-42)
  - Suggested fix: Document that caller owns session lifecycle; don't auto-create.

#### P2 -- Minor

- **P2-1** [Statistical]: REI clip(-10, 10) bounds undocumented — intentional for zero-trial countries but should have a comment.
- **P2-2** [Engineer]: No test for empty DataFrame in build_trial_matrix(), compute_annual_gini().
- **P2-3** [Engineer]: pivot_wb_wide() uses aggfunc="first" silently discarding duplicates.
- **P2-4** [Engineer]: Country name variants ("South Korea" vs "Korea, Republic of") — allocator has both but may miss others.

#### False Positive Watch
- P0-8 (XSS in dashboard): Data embedded at build time from our own CSVs, not user input. Safe.
- P0-9 (Cache filename path traversal): gbd_cause_id from our crosswalk CSV, always integer. Safe.
- P0-37 (Session closed mid-loop): session.close() is in finally AFTER the loop — correct behavior.
- P0-2 (Gini 0.0 conflation): Returns 0.0 for empty is reasonable default; never called with empty in practice.
- P0-1 (REI clipping): Intentional design for zero-trial countries; clip(-10,10) prevents log(0) artifacts.
