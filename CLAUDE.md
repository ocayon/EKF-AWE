# EKF-AWE — Agent Context

Extended Kalman Filter for airborne-wind-energy flight-data analysis
(`src/awes_ekf/`). Consumed by AWETrim (sibling repo
`C:\Users\ocayon\Repositories\AWETrim`). Since 2026-08-24 AWETrim's venv has
an EDITABLE install of this working tree (`pip install -e`, replacing the old
`git+...@main` pin), so local edits are live in AWETrim immediately — but a
venv rebuild from the pyproject pin would silently go back to origin main, so
keep main pushed. Layout:
`ekf/` (filter + calibration pre-run), `setup/` (SimulationConfig,
TuningParameters, kite/tether/KCU models), `postprocess/`, `load_data/`.
Input CSV spec and diagrams: `.claude/`.

---

## Handoff (2026-08-24, evening — steering-dependent aero stages DONE)

### State of the repo

- `main` is at `4dee7c6` (calibration pre-run robustness). **Unpushed.**
- UNCOMMITTED implementation of the steering-dependent aero stages (this
  session, tested, works): `setup/settings.py`, `setup/kite.py`,
  `ekf/kalman_filter.py`, `ekf/ekf_output.py`,
  `load_data/create_input_from_csv.py`. Both flags default False → base runs
  bit-identical. Also uncommitted user WIP:
  `examples/identify_aero_parameters_turn_law.py`.
- AWETrim-side edit: `tune_ekf.py: apply_override` now CREATES unknown keys
  inside known blocks (needed to `--set` the new flags, absent from base h5s).

### The steering-dependent aero stages (implemented + validated)

Config flags in `simulation_parameters` (SimulationConfig):

- `steering_dependent_cs` (stage 1): side force
  `CS_tot = CL_eff·tan(k_phi_us·u_s) + CS`; the CS state becomes the residual
  (run with `model_stdv.CS = 0.003`). New constant state `k_phi_us` (Q 1e-6,
  the k_cl_up pattern), `u_s` joins the input vector (kite.get_input +
  ekf.update_input_vector — that latent model_yaw us-input gap is fixed too).
- `steering_dependent_clcd` (stage 2): `+ k_cl_us·|u_s|` on CL,
  `+ k_cd_us·u_s²` on CD (even by symmetry — direction can't matter for
  lift/drag); run with `model_stdv.CL/CD = 0.005/0.002` (tightening CONFIRMED
  better than leaving 0.01/0.003: loose control run had wind-speed pattern
  0.457 vs 0.401 pp).
- Outputs: `wing_*_coefficient` = the STATES (residuals when stages on, so
  diagnose_ekf stays apples-to-apples); totals in `wing_*_coefficient_total`;
  constants in `k_phi_us`, `k_cl_us`, `k_cd_us`.
- NOTE `create_input_from_csv` normalizes steering by max|kcu_actual_steering|
  of the LOADED slice (35.0 for minutes 60–100), not /200 — constants scale
  accordingly (k_phi_us −0.096 on the slice ≡ −0.55 on the /200 scale).

Results on the 60–100 min slice (`_tune_s1/_tune_s2` vs `_tune_pinz2`, all
with vw 0.02, vwz 0.005, enforce_vertical_wind_to_0, frozen pitot):

- Constants converge in minutes, split-half agreement 1–5 %. k_phi_us −0.546
  on the /200 scale (regression predicted −0.55). k_cl_us −0.32, k_cd_us +3.3
  (/200 scale; regression said −0.59/+1.33 — expected drift, the regression
  was fit on the old lagging walk).
- CS state: pattern 0.078 → 0.025 pp, hp-std 3.3× down, us-correlation
  −0.88 → +0.03. CL: pattern 0.086 → 0.046 pp (pat% 26→17). CD: 0.051 → 0.021
  (24→11). Vertical wind clearly better: pattern 0.183 → 0.124, mean w_z bias
  +0.20 → +0.10. NIS 0.7 → 0.6.
- Horizontal wind UNCHANGED (speed pattern 0.394→0.401 pp, direction
  2.31→2.48° pp, TI 8.8→8.7 %): the wind-channel pattern leak does NOT come
  from the aero-coefficient walks. Don't re-attack the wind through them.

Follow-up candidates (TRIED, same session, all implemented + kept):

- `steering_input_lag` (SimulationConfig, seconds; first-order pre-filter on
  u_s in create_input_from_csv): identified tau = 0.3 s from the s2 CS
  residual (zeroes r(residual, du_s/dt); the variance minimum at 0.4 is
  shallow). With the lag, k_phi_us rises to −0.60 /200-scale (the unlagged
  fit was attenuation-biased).
- `steering_dependent_cl_asym`: signed `k_cl_us_odd·u_s` on CL ONLY.
  Converges to +0.20 /200-scale (the +0.23 the original regression saw).
  A signed CD term was ALSO tried and made CD worse in every run that had it
  (0.021→0.028–0.034 pp) — removed from the code, do not re-add.

Candidate landscape, pattern pp per channel (60–100 min slice):

  |               | pinz2 |   s2  | s3b (s2+lag+cl_asym) |
  | wind speed    | 0.394 | 0.401 | 0.448  <- the one consistent cost
  | wind dir deg  | 2.310 | 2.481 | 2.188  <- best of all runs
  | w_z           | 0.183 | 0.124 | 0.123  (bias +0.20 -> +0.10)
  | CL            | 0.086 | 0.046 | 0.030
  | CD            | 0.051 | 0.021 | 0.028  (lag slightly hurts CD)
  | CS            | 0.078 | 0.025 | 0.017

  Intermediates (s3lag, s3asym, s3full) confirmed attribution: the lag buys
  CS + direction + w_z and costs wind-speed pattern (+0.04 pp); the CL asym
  buys CL (0.046→0.025 alone) at small wind-speed cost. No variant beats
  pinz2 on wind-speed pattern — every deterministic term nudges it up ~0.5 %
  of the mean; whatever the wind-speed leak is, it is NOT the aero walks.
- RECOMMENDED: s3b config for aero-coefficient work (constants: k_phi_us
  −0.60, k_cl_us −0.28, k_cd_us +3.7, k_cl_us_odd +0.20, all /200 scale,
  split-half 1–12 %); s2 or pinz2 when horizontal wind speed is the only
  thing that matters (differences are ≤0.06 m/s pp anyway).

### Test harness (AWETrim, unchanged)

- `AWETrim\scripts\personal\wes-quasi-steady\tune_ekf.py --name X --pitot
  0.8224,33.0322 --set ...` (60–100 min slice; stage flags via --set as
  above), then `diagnose_ekf.py --ekf <h5> [--ekf <h5> ...]` side by side.
- Reference h5s in `AWETrim\results\LEI-V3-KITE\ekf\`: `_tune_pinz2` (best
  no-stage tuning), `_tune_s2`, `_tune_s3lag/s3asym/s3full` (attribution
  runs), `_tune_s3b` (recommended). NOTE: the pre-15:00 tune files (incl. the
  original pinz2/s1/s2) were deleted outside the session on 2026-08-24;
  pinz2 and s2 were regenerated and reproduce the originals exactly, s1 and
  s2loose were not (regenerate with the flags above if needed).

### Fixed background decisions (do not relitigate)

- vw 0.02 is the sweet spot (0.05 wanders with the loop, 0.01 suppresses real
  turbulence and biases the wind level +0.4 m/s).
- Mean w_z is unobservable in this filter (sign flips with tuning on the same
  data; 2019-10-08 was overcast + rain, so the +1 m/s "updraft" is spurious)
  → `enforce_vertical_wind_to_0: true` in AWETrim's ekf_config.yaml.
- NIS consistency is muddied by the 1e-5 least-squares pseudo-measurements —
  don't tune toward it.
- CL/CD steering terms are EVEN (|u_s|, u_s²); only CS is odd. A signed
  sign(u_s)·u_s² would claim left/right turns change drag oppositely.

### Test harness (lives in AWETrim, works on any results h5)

- Solver driver:
  `C:\Users\ocayon\Repositories\AWETrim\scripts\personal\wes-quasi-steady\tune_ekf.py`
  — non-interactive re-run of the 2019 flight on a minutes slice (default
  60–100), config taken from an existing results h5, `--set key=value`
  overrides, `--name X` → `..._tune_X.h5`. `--pitot 0.8224,33.0322` freezes
  the pitot calibration (recovered from the production solve) — WITHOUT it the
  pre-run refits k with the candidate's own tuning, which moved k by 20 %
  between candidates and shifted the whole wind level (a confound).
- Acceptance metric:
  `...\wes-quasi-steady\diagnose_ekf.py --ekf <h5> [--plot]` — decomposes each
  wind/coefficient channel into loop-locked (pattern), height (shear) and
  residual parts over the figure-eight phase. Success = the CS/CL/CD "pat%"
  columns drop with the wind channels unchanged or better.
- Reference numbers to beat (recommended tuning `pinz2`, minutes 60–100:
  vw 0.02, vwz 0.005, CL/CD/CS 0.01/0.003/0.01, enforce_vertical_wind_to_0
  true): wind speed 3.1 % pattern-locked / TI 8.8 %, direction 2.3° pp,
  CS 72 % pattern-locked (the number stage 1 attacks).

### Fixed background decisions (do not relitigate)

- vw 0.02 is the sweet spot (0.05 wanders with the loop, 0.01 suppresses real
  turbulence and biases the wind level +0.4 m/s).
- Mean w_z is unobservable in this filter (sign flips with tuning on the same
  data; 2019-10-08 was overcast + rain, so the +1 m/s "updraft" is spurious)
  → `enforce_vertical_wind_to_0: true` in AWETrim's ekf_config.yaml.
- NIS consistency is muddied by the 1e-5 least-squares pseudo-measurements —
  don't tune toward it.
