# AGENTS.md

## What this repo is

A frozen benchmark harness plus a **single-file submission**. The only file you may create or
modify is `adapters/myteam.py` containing `class Engine(Adapter)`. Everything else
(`adapter.py`, `recovery_model.py`, `data.py`, `metrics.py`, `harness.py`, `run.py`,
`self_check.py`, existing adapters, `requirements.txt`, docs) is frozen — the organizers hash and
rerun that exact file. **This directory is not a git repo**, so nothing enforces the one-file rule;
keep track of your own diff and don't leave scratch files here (use `%TEMP%`).

Long-form brief with all constraints and the intended research process: `Workflow.txt`.
Task statement: `PROBLEM.md`. Scoring/dynamics spec: `README.md`.

## Commands (PowerShell, from the repo root)

```powershell
pip install -r requirements.txt          # numpy>=1.20 only
$env:PYTHONIOENCODING="utf-8"            # REQUIRED, see below
python self_check.py --adapter adapters.dummy:DummyAgent --quick     # baseline sanity
python self_check.py --adapter adapters.myteam:Engine --quick         # iterate
python run.py --adapter adapters.myteam:Engine --seeds 7 13 31 97 211 503 1009 --out report.json
```

- **The env var is not optional on Windows.** `self_check.py` prints Unicode box-drawing
  characters and dies with `UnicodeEncodeError` at `self_check.py:71` under the default cp1252
  console. Verified: with the var set, exit code 0.
- `--adapter` is `module:Class` split on `:` (`run.py:23`). The class is constructed
  positionally as `cls(X, params)`.
- Must run from the repo root so the adapter's `from adapter import Adapter` resolves.
- `run.py --out` defaults to `-` (stdout). It writes `report.json` *into the repo* — pass a path
  outside the tree if you care about a clean submission directory.
- No linter, formatter, typechecker, or test suite exists here. `self_check.py --quick` is the only
  verification tool.
- Verified wall time: `--quick` (2 seeds) ≈ 77–86 s even with the all-ones dummy adapter, because
  the harness itself runs the 3000-step dynamics for baseline + agent on every query. Full config
  (5–7 seeds x 3 levels x 250 queries) is minutes-to-tens-of-minutes; **your per-query cost is
  added directly on top of that**, so do not run full simulations inside `predict_precision`.

## Reference numbers (verified locally, `--quick`, seeds 42/101)

| adapter | seed 42 delta | mean reduction | automated pts |
| --- | --- | --- | --- |
| `adapters.dummy:DummyAgent` | +0.000 | 1.00x | 0.00 |
| `adapters.variance:VarianceAgent` | **-0.325** | 1.00x | 0.00 |
| `adapters.class_conditional:ClassConditionalAgent` | -0.042 | 1.05x | 0.67 |

The two shipped "reference" adapters score ~0 — treat them as illustrations, not as bars to beat.
Note also that on these synthetic seeds direct cosine classification (0.742) *beats* the uniform
dynamics (0.667). The `dynamics_adds_value` diagnostic is informational only and carries no
penalty (`harness.py:184-189`) — don't burn effort trying to flip it.

## Scoring gates that actually cost points

- Retrieval (70): linear on `mean_delta`, full marks at `+0.08`, hard zero if mean <= 0, **halved
  if any single seed has delta < 0** (`harness.py:176-211`).
- Balance (20): logarithmic, full marks at `5x`, hard zero if mean reduction <= 1.0, **halved if
  any single seed has reduction <= 1.0** (`harness.py:214-231`).
- Therefore optimize worst-seed robustness, not the mean. A flat +0.04 everywhere beats +0.15 on
  one seed and -0.05 on another.

## `predict_precision` is called on two very different input distributions

This is the single most important structural fact, and it is invisible from the filenames:

- **Retrieval path** (`metrics.py:36-43`): query = stored pattern with 60/75/85% of dimensions
  zeroed plus Gaussian noise of `0.4/sqrt(N)`, renormalized (`data.py:76-96`).
- **Balance path** (`metrics.py:109-120`): probe = `x_i + 0.05 * standard_normal(N)` — an almost
  *clean* stored pattern, unscaled per-dimension noise — renormalized.

Consequence: a pure query-reliability rule (anything driven by `|q|` or `|q - m|`) looks nearly
uniform on the balance probe, returns ~1.00x reduction, and scores 0 balance points. Balance needs
a signal derived from the frozen geometry (`R`, Hessian of `sqrt(pi) H sqrt(pi)`), which is
mostly query-independent. Retrieval and balance must be blended, not traded off.

## Balance metric mechanics

- The probe's equilibrium `a*` is found under `pi = 1` with no input (`metrics.py:111`,
  `recovery_model.py:175-182`), i.e. near `eta * R^-1 x_i`, not at `x_i`. The adapter's `pi` cannot
  move `a*`; it only changes the conditioning of `sqrt(pi) H(a*) sqrt(pi)` at that fixed point.
- Spread = `eig_max / eig_min` of the symmetrized matrix with eigenvalues `<= 1e-9` dropped;
  a sample is silently **skipped** (returns `None`) when `H(a*)` is not positive definite
  (`metrics.py:65-83`). Fewer usable samples changes the reported mean.
- Scores are halved if `min reduction <= 1.0`, so verify every seed clears 1.0x.

## Weight post-processing you must design around

`clip_and_normalise` (`recovery_model.py:118-142`) clips to `[pi_min, pi_max] = [0.1, 10.0]`,
then iterates clip + rescale-to-mean-1 (max 20 iterations; the final clip can leave the mean off
1.0). Consequences:

- Absolute scale is discarded — **only the shape of the returned vector matters.**
- Ratios beyond 10:1 saturate and then get flattened; design the final vector to survive clipping.
- A non-finite return is silently replaced by `ones(N)` (`recovery_model.py:127-128`), i.e. a
  silent zero score. Guard every division, `exp`, `log`, and normalization yourself and fall back
  to a finite `ones(N)`-like vector.

## The dynamics your weights steer (`recovery_model.py`)

```
g(a)     = R a - eta * X.T @ softmax(beta * X @ a)
update   = -pi * g(a) + q   for t < T_in (T_in = 100), then -pi * g(a)
a_next   = a + dt * update   (dt = 0.01, T_max = 3000, tol = 1e-6)
```

`pi` multiplies the *entire* gradient, so it re-rates both how fast dimensions respond to the
external query during the input window and the free relaxation afterwards. `R` is a structured
non-diagonal operator (`0.5*I + 0.2*L_laplacian + 0.1*11^T`, `recovery_model.py:29-56`) — do not
assume dimensions are independent.

## Conventions and traps

- One adapter instance per seed (`harness.py:125`). Cache everything derived from `X`, `R`, and
  the model params in `__init__` (`X.T`, `X @ X.T`, per-dimension stats, `R` diagonal / row norms,
  any decomposition). Keep `predict_precision` to query-dependent work only.
- Patterns are clustered (4 clusters, `intra_sim = 0.5`) and assigned round-robin by index, so
  index order carries no meaning and neighbouring indices are often same-cluster rivals. Infer
  structure numerically instead.
- Do not hardcode `N = 64` or `K = 16`; hidden evaluation may use larger pattern sets. Derive `N`
  from `stored_patterns.shape[1]`.
- Patterns, `R`, queries, and truth labels are regenerated per seed; the adapter never sees the
  source index. No seed-specific tables, no public-seed tuning.
- Treat `stored_patterns`, `model_params`, and every `corrupted_query` as read-only.
- Prohibited by the rules: caller-frame/evaluator-local inspection, monkeypatching, `eval`/`exec`,
  dynamic imports for cheating, subprocesses, filesystem, network, environment inspection,
  obfuscated or encoded payloads, or any modification of benchmark files.
- Return exactly `N` finite values; the submission must be one readable, self-explaining file
  (10 of 100 points are a manual code-quality review).