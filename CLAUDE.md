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

- NOTE: the PRODUCTION 2019 h5 (`LEI-V3 Kite_2019-10-08.h5`) was re-solved
  by the user with the s3b config on 2026-08-25 — base h5 configs read from
  it now carry the stage flags and tightened walks by default.
- `main` = `f7ba973`, PUSHED to origin: the steering-dependent stages
  (`c12e96a`), the folder-based config convention (`f7ba973` — load_config
  asks for a data/<KITE-NAME>/ folder holding ekf_config.yaml + awesIO
  system*.yaml variants, same extraction as AWETrim; data/LEI-V3-KITE/
  carries both flown variants, V9 stays local-only per the ignore rules),
  and the earlier calibration robustness (`4dee7c6`). Uncommitted user WIP:
  `examples/identify_aero_parameters_turn_law.py`.
- AWETrim-side edits (UNCOMMITTED in AWETrim): `tune_ekf.py: apply_override`
  now CREATES unknown keys inside known blocks (needed to `--set` new flags
  absent from base h5s), and `data/LEI-V3-KITE/ekf_config.yaml` now carries
  the s3b recommendation (all three stage flags + lag 0.3 + walks
  0.005/0.002/0.003) — the next production solve uses the stages.

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
- Outputs (convention FLIPPED 2026-08-25, user request): `wing_*_coefficient`
  = the FULL coefficients the force model used (paper-comparable, what the
  plots show); the walk states alone are in `wing_parasitic_drag_coefficient`
  and `wing_lift/sideforce_coefficient_residual` (the old `_total` columns
  are gone — h5s from before the flip have plain=STATE and `_total`=full).
  diagnose_ekf reads the residual columns (falls back to plain on old h5s)
  so its CL/CD/CS leakage channels still score the walks. Constants in
  `k_phi_us`, `k_cl_us`, `k_cd_us`.
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

### 2025-10-09 lidar validation (28-min flight, profiling lidar 40-250 m)

The one flight with an independent wind reference. base25 (pinz2 tuning,
pre-run pitot k=0.8308 b=-5.12 — k nearly identical to 2019's 0.8224) vs
s3b25 (same frozen pitot + the s3b flags), full flight minus the last ~40 s
(landing garbage that reset-storms the filter; cut at minute 28):

- EKF vs lidar at kite height, 60-s blocks: speed bias +0.16 -> +0.10 m/s
  (RMS 0.48/0.47, corr 0.81/0.78), direction bias -0.4 -> -0.2 deg (RMS
  10.2 -> 9.9), w_z bias +0.13 -> +0.07 (lidar mean -0.08, EKF -0.01 with
  the stages). The stages are equal-or-better on every lidar metric; the
  small wind-speed pattern increase the phase metric shows (1.6 -> 2.6 %)
  does NOT show up as a real accuracy loss against the lidar.
- Stages transfer: CS pattern 0.072 -> 0.042 pp, CL 0.118 -> 0.056 (pat%
  33 -> 13), CD 0.036 -> 0.023. Mean profile matches lidar within ~0.2 m/s
  and 1-2 deg at every height.
- Constants (/200 scale): k_phi_us -0.75 (2019: -0.60, split-half 4 %),
  k_cl_us -1.1 and k_cd_us +7.5 (larger than 2019, split-half 10-22 % — a
  28-min flight converges them only roughly), k_cl_us_odd NOT identified
  (split-half 206 %) — don't trust the asym constant from short flights.
- Results: `_tune_base25`, `_tune_s3b25` h5s; lidar comparison script:
  the session scratchpad's `lidar_compare.py` (interpolates the profile to
  kite height; met dir -> ENU downwind is rad = deg2rad(270 - dir_met)).

### The wing-CD dips and the drag-polar stage (2026-08-25, negative result)

The 2019 wing CD (median 0.09) dips below 0.05 for ~12 % of samples and
below 0 for ~0.3 % (2.6 % in the old loose production tuning). Diagnosed:

- Dips are brief (2-4 s) episodes, 88-93 % of them in REEL-OUT, at normal
  va, with the parasitic KCU+tether+bridle CD flat (0.068) — NOT a low-va,
  depower, or component-drag problem. At the dips CL is HIGH (0.66 vs
  0.59): the signature of lift/drag AXIS MISATTRIBUTION — a 2-3 deg
  apparent-wind direction error rotates ~eps*CL between the axes, which at
  CL 0.7 is exactly the 0.03 dip scale. CD suffers most because it sits
  orthogonal to a force 7x larger.
- `drag_polar` (SimulationConfig, default OFF): CD = CD0 + k_cd_cl2*CL_eff^2
  is IMPLEMENTED (constant state k_cd_cl2; with the flag on, the depower
  path k_cd_up*delta_up on CD is disabled so the polar owns the slow
  signal). DO NOT ENABLE for this data: k_cd_cl2 converges confidently
  NEGATIVE (-0.013, split-half 3.5 %) because the fast CD-CL covariance the
  filter sees is the anticorrelated misattribution noise, and the slow
  phase-level relation is flat too — reel-in flies CL 0.42 at CD 0.089 vs
  reel-out CL 0.66 at CD 0.093, where a physical polar (k ~ 0.11 from
  AR 3.47) would give ~0.064. Whether that is genuine depowered-LEI
  aerodynamics or reel-in parasitic-drag bias, the polar is not
  identifiable from this flight; it also leaves the dips unchanged
  (12.3 -> 12.1 % below 0.05). Runs: `_tune_s4` (first form), `_tune_s4b`
  (depower path removed), `_tune_s4c` (CD walk 0.001 — dips 11.2 %, best
  of the three, still no fix).
- RESOLUTION (user's insight): with the stages carrying the deterministic
  physics, the CD state is a PARASITIC near-constant and should barely
  move — tightening model_stdv.CD is the fix, no polar needed (at tight CD
  the polar-vs-no-polar runs are identical; keep drag_polar OFF).
  2019 slice, s3b flags, by CD walk (total wing CD):
    0.002: 12.3 % below 0.05, p1 +0.014 (the complaint)
    0.0005 (`_tune_s5d`): 9.5 %, p1 +0.025, wind-dir pattern 2.19->1.70
    0.0001 (`_tune_s5e`): 1.6 %, p1 +0.043, dir 1.63 — but w_z pattern
      0.123->0.152 and TI 8.8->6.6 % (stiffness warnings on 2019)
  The remaining <0 rate (0.2 %) is only the first-seconds transient.
  LIDAR REFEREE (2025, `_tune_s4d25/_tune_s4e25` vs `_tune_s3b25`):
  agreement unchanged at every tightness (speed bias +0.10/+0.09/+0.10,
  RMS 0.47/0.47/0.48, dir RMS 9.9/10.4/10.5, w_z bias +0.07/+0.07/+0.05) —
  the 2019 wind-speed-pattern rise is not a real accuracy loss.
  RECOMMENDED: model_stdv.CD = 0.0005 for production (balanced); 0.0001 if
  a hard floor matters more than the 2019 w_z/TI stiffness signs. A
  positivity reparameterization (softplus floor) is now likely unnecessary.
  ADOPTED 2026-08-25: CD = 0.0005 is in data/LEI-V3-KITE/ekf_config.yaml in
  BOTH repos (EKF-AWE committed, AWETrim committed separately).

### The distance_kcu_kite=0 regression and the wing-CD level (2026-08-25)

User asked why the 2019 wing CD is ~0.07 when the paper (`v1.2.0-paper` tag,
Apr 2025, old h5s in `EKF-AWE\results\v3\`) said 0.123. Decomposition:

- ~0.015 was display: `wing_drag_coefficient` held the residual STATE from
  the stages until 2026-08-25; the user disliked that and the convention is
  now flipped (plain = full coefficient, see the stages section).
- ~0.011 is the pitot dynamic-pressure calibration (paper used raw va;
  calibrated va is ~+5 % speed → all coefficients ~−10 %, lidar-backed).
- ~0.014 is tether discretization 5→30 elements (tether CD 0.031→0.045).
- +0.006 was a BUG: every awesIO-convention run (both productions, ALL tune
  h5s, the lidar validation) had `distance_kcu_kite = 0`, because the
  extraction (in BOTH repos) read `bridle_point_node[2]`, which is the
  body-frame origin in awesIO — always 0. Zero bridle length degenerates the
  KCU→kite direction ej in tether.py to solver-residual noise: bridle drag
  collapsed 0.012→0.001, KCU CD doubled 0.007→0.014, and the tether-length
  offset state silently absorbed ~9 m.
- FIX (user decision): derive the distance from the wing CG height above the
  bridle point — `wing_struct["center_of_mass"][2]` = 10.3 m for the V3 —
  in both `src/awes_ekf/setup/settings.py` and AWETrim's
  `src/awetrim/experimental/settings.py` (the production pipeline
  `run_analysis_ekf.py` uses the AWETrim copy).
- A/B on the slice (`_tune_s5d` dist 0 vs `_tune_dist115` dist 11.5, all
  else identical): wing CD total 0.088→0.094 flight median, bridle/KCU CD
  restored to paper-era values, dips <0.05 9.5→6.6 %, wind + pattern metrics
  + NIS UNCHANGED, k_phi_us unchanged (only k_cd_us −16 %). The stages and
  lidar conclusions survive; only drag attribution was corrupted.
- Reconciliation: paper 0.123 − calibration − tether-resolution ≈ 0.095;
  fixed run gives total 0.094. The books balance; ~0.094 total (CD0 state
  ~0.079) is the defensible current number.
- PENDING: both production h5s must be RE-SOLVED by the user (interactive
  run_analysis_ekf) to pick up the fix; until then base-h5 configs feed
  tune runs distance 0 (override with --set kcu.distance_kcu_kite=10.3).
  All reference tune h5s except `_tune_dist115`/`_tune_cg103` carry the bug.
- `_tune_cg103` = the slice reference with the CG-derived 10.3 m AND the
  renamed output columns: wing CD (full) 0.093 flight med / 0.091 reel-out,
  parasitic CD0 0.084/0.079, bridle CD 0.012, KCU 0.007, offset 26.2 m,
  CD<0.05 at 6.9 %, constants within noise of `_tune_dist115`.

### Fixed background decisions (do not relitigate)

- vw 0.02 is the sweet spot (0.05 wanders with the loop, 0.01 suppresses real
  turbulence and biases the wind level +0.4 m/s).
- Mean w_z is unobservable in this filter (sign flips with tuning on the same
  data; 2019-10-08 was overcast + rain, so the +1 m/s "updraft" is spurious)
  → `enforce_vertical_wind_to_0: true` in AWETrim's ekf_config.yaml.
- NIS consistency is muddied by the 1e-5 least-squares pseudo-measurements —
  don't tune toward it.
- CL/CD steering terms are EVEN (|u_s|, u_s²); only CS is odd. A signed
  sign(u_s)·u_s² would claim left/right turns change drag oppositely, and a
  signed linear CD term was tried and made CD worse.
