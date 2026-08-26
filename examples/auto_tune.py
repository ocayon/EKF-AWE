"""Automatic EKF tuning for a kite flight: the decision ladder as a driver.

Encodes the tuning methodology developed on the LEI V3 (2026-08) and
validated on the LEI V9 (2026-08-26, both flights refereed by a profiling
lidar). Phases, in dependency order, each with a primary objective and
do-no-harm guardrails:

  P0  audit      one solve at generic-loose defaults; sniff the flight
                 (airspeed? steering? depower activity? lidar columns?)
  P1  wind walk  vw grid on va-scale-INSENSITIVE criteria (lidar direction
                 RMS, or blind direction leak with a turbulence guard)
  P2  pitot      dynamic-pressure scale k, b = 0 (sensor physics): secant
                 on the 60-s-block lidar speed bias, or blind on the
                 wind-va loop-correlation zero (flagged unanchored, the
                 model-bias floor is ~3-4 % in k)
  P3  stages     steering-dependent CS/CL/CD terms + tightened walks;
                 each constant must reproduce across flight halves
  P4  lag        actuation-lag sweep: smallest lag with >=70 % of the max
                 guarded CS-leak reduction
  P5  cd walk    near-frozen CD only if the dip rate improves and the
                 wind/stiffness guards hold
  P6  final      full solve, pitot re-verify, evidence report + proposed
                 ekf_config_autotuned.yaml (the existing config is never
                 touched)

Fixed by methodology, never tuned here: meas_stdv (sensor properties),
NIS as a target (guardrail band only -- the least-squares pseudo-
measurements corrupt it), enforce_vertical_wind_to_0 (mean w_z is
unobservable), vwz = 0.005, tether n_elements = 30.

Usage
-----
    python examples/auto_tune.py --base-h5 results/v9/v9_2023-11-27.h5
    python examples/auto_tune.py --config-folder data/MY-KITE \
        --flight-model mykite --date 2026-01-01 [--no-lidar] [--max-parallel 3]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR / "src"))

# ------------------------------------------------------------ thresholds
SLOPE_PRIOR = 19.0  # m/s wind bias per unit va speed-scale (V3+V9)
PITOT_BIAS_TOL = 0.08  # m/s: lidar secant convergence
DRIFT_KEEP_PCT = 30.0  # split-half drift below which a constant is trusted
LAG_GRID = [0.5, 1.0, 1.5]
LAG_ELBOW = 0.70  # smallest lag with this fraction of max CS-leak reduction
CD_TIGHT = 0.0005
NIS_BAND = (0.2, 3.0)


# ------------------------------------------------------------ config I/O
def _plain(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def config_from_h5(path):
    import h5py

    def convert(v):
        if isinstance(v, bytes):
            return v.decode()
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    def walk(g):
        out = {k: convert(v) for k, v in g.attrs.items() if k != "description"}
        for k in g:
            out[k] = walk(g[k])
        return out

    with h5py.File(path, "r") as hf:
        return walk(hf["config_data"])


def start_overrides(config):
    """Reset to the generic-loose ladder start, independent of whatever
    tuning the base config carries."""
    sp = config["simulation_parameters"]
    sp["enforce_vertical_wind_to_0"] = True
    sp["calibrate_apparent_windspeed"] = False
    sp["find_offset_angle_of_attack"] = True
    sp["steering_dependent_cs"] = False
    sp["steering_dependent_clcd"] = False
    sp["steering_dependent_cl_asym"] = False
    sp["steering_input_lag"] = 0.0
    sp["drag_polar"] = False
    sp.pop("pitot_calibration_k", None)
    sp.pop("pitot_calibration_b", None)
    ms = config["tuning_parameters"]["model_stdv"]
    ms.update({"vw": 0.1, "vwz": 0.005, "CL": 0.01, "CD": 0.003, "CS": 0.01})
    config["tether"]["n_elements"] = 30
    return config


# ------------------------------------------------------------ worker
def solve_worker(jobfile):
    """Run one EKF solve described by a job json. Executed in a subprocess."""
    job = json.loads(Path(jobfile).read_text())
    os.chdir(job["project_dir"])

    import time as _t

    from awes_ekf.ekf.ekf_output import convert_ekf_output_to_df
    from awes_ekf.ekf.initialize_and_update_ekf import (
        initialize_ekf,
        propagate_state_EKF,
    )
    from awes_ekf.load_data.create_input_from_csv import (
        create_input_from_csv,
        find_initial_state_vector,
    )
    from awes_ekf.load_data.read_data import read_processed_flight_data
    from awes_ekf.load_data.save_data import save_results
    from awes_ekf.postprocess.postprocessing import postprocess_results
    from awes_ekf.setup.kcu import KCU
    from awes_ekf.setup.kite import PointMassEKF
    from awes_ekf.setup.settings import SimulationConfig, TuningParameters
    from awes_ekf.setup.tether import Tether

    config = job["config"]
    year, month, day = config["year"], config["month"], config["day"]
    kite_model = config["kite"]["model_name"]
    flight_data = read_processed_flight_data(year, month, day, kite_model)
    dt = float(flight_data["time"].diff().mean())
    lo = int(job["start_min"] * 60 / dt)
    hi = int(job["end_min"] * 60 / dt) if job["end_min"] > 0 else len(flight_data)
    flight_data = flight_data.iloc[lo:hi].reset_index(drop=True)

    simConfig = SimulationConfig(**config["simulation_parameters"])
    kite = PointMassEKF(simConfig, **config["kite"])
    kcu = KCU(**config["kcu"]) if config["kcu"] else None
    tether = Tether(kite, kcu, simConfig.obsData, **config["tether"])
    kite.calc_fx = kite.get_fx_fun(tether)
    tuningParams = TuningParameters(config["tuning_parameters"], simConfig)
    ekf_input_list = create_input_from_csv(
        flight_data, kite, kcu, tether, simConfig, kite_sensor=0
    )
    x0 = find_initial_state_vector(
        tether, ekf_input_list[0], simConfig,
        wind_velocity=simConfig.initial_wind_velocity,
    )
    ekf, ekf_input_list = initialize_ekf(
        ekf_input_list, simConfig, tuningParams, x0, kite, kcu, tether
    )
    ekf_output_list, n_reset = [], 0
    t0 = _t.time()
    for k, ekf_input in enumerate(ekf_input_list):
        try:
            ekf, out = propagate_state_EKF(ekf, ekf_input, simConfig, tether, kite, kcu)
            ekf_output_list.append(out)
        except Exception as exc:
            n_reset += 1
            print(f"  reset {n_reset} at {k}: {exc}", flush=True)
            try:
                x0 = find_initial_state_vector(tether, ekf_input, simConfig)
            except Exception:
                x0 = ekf.x_k1_k1
            ekf, ekf_input_list[k::] = initialize_ekf(
                ekf_input_list[k::], simConfig, tuningParams, x0, kite, kcu,
                tether, find_offsets=False,
            )
            flight_data.drop(k, inplace=True)
            continue
        if k and k % 12000 == 0:
            r = k / (_t.time() - t0)
            print(f"  {k}/{len(ekf_input_list)} ({r:.0f}/s)", flush=True)
    ekf_output_df = convert_ekf_output_to_df(ekf_output_list)
    ekf_output_df.dropna(subset=["kite_pitch"], inplace=True)
    ekf_output_df.reset_index(drop=True, inplace=True)
    flight_data = flight_data.iloc[ekf_output_df.index].reset_index(drop=True)
    ekf_output_df, flight_data = postprocess_results(
        ekf_output_df, flight_data, kite, kcu, config
    )
    save_results(
        ekf_output_df, flight_data, kite_model, year, month, day, config,
        addition=job["addition"],
    )
    print(f"done ({n_reset} resets)")


# ------------------------------------------------------------ orchestration
class Ladder:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.model = config["kite"]["model_name"]
        self.date = f"{config['year']}-{config['month']}-{config['day']}"
        self.tag = args.tag
        self.out_dir = (
            PROJECT_DIR / "results" / self.model
            / f"autotune_{self.date}{('_' + self.tag) if self.tag else ''}"
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = []
        self.solved = {}

    def _job_h5(self, name, cfg):
        """h5 path for a candidate; the name carries a hash of its config so
        --reuse can never pick up a run from a different decision path."""
        import hashlib

        h = hashlib.md5(
            json.dumps(cfg, sort_keys=True, default=str).encode()
        ).hexdigest()[:6]
        prefix = f"{self.tag}_" if self.tag else ""
        stem = f"{prefix}{name}_{h}"
        return stem, (
            PROJECT_DIR / "results" / self.model
            / f"{self.model}_{self.date}_auto_{stem}.h5"
        )

    def solve_many(self, jobs):
        """jobs: list of (name, config_dict). Runs missing ones in parallel."""
        pending = []
        self._expected = getattr(self, "_expected", {})
        for name, cfg in jobs:
            if name in self.solved:
                continue
            stem, path = self._job_h5(name, cfg)
            self._expected[name] = path
            if path.exists() and self.args.reuse:
                self.solved[name] = path
                continue
            jf = self.out_dir / f"job_{stem}.json"
            jf.write_text(json.dumps({
                "project_dir": str(PROJECT_DIR),
                "config": cfg,
                "start_min": self.args.start_min,
                "end_min": self.args.end_min,
                "addition": f"_auto_{stem}",
            }, default=str))
            pending.append((name, jf))
        procs = {}
        for name, jf in pending:
            while len(procs) >= self.args.max_parallel:
                self._reap(procs)
                time.sleep(5)
            logf = open(self.out_dir / f"log_{name}.txt", "w")
            procs[name] = (
                subprocess.Popen(
                    [sys.executable, str(HERE / "auto_tune.py"), "--solve-one", str(jf)],
                    stdout=logf, stderr=subprocess.STDOUT, cwd=str(PROJECT_DIR),
                ),
                logf,
            )
            print(f"  solving {name} ...", flush=True)
        while procs:
            self._reap(procs)
            time.sleep(5)
        for name, _ in jobs:
            if name not in self.solved:
                raise RuntimeError(f"solve {name} produced no h5")

    def _reap(self, procs):
        for name in list(procs):
            p, logf = procs[name]
            if p.poll() is not None:
                logf.close()
                del procs[name]
                if p.returncode == 0 and self._expected[name].exists():
                    self.solved[name] = self._expected[name]
                    print(f"  {name} done", flush=True)
                else:
                    print(f"  {name} FAILED (exit {p.returncode})", flush=True)

    def score(self, name):
        from autotune_metrics import score_run

        s = score_run(self.solved[name])
        if self.args.no_lidar:
            s["lidar"] = None
        return s

    def decide(self, phase, choice, reason, table):
        entry = {"phase": phase, "choice": choice, "reason": reason, "table": table}
        self.log.append(entry)
        print(f"[{phase}] -> {choice}: {reason}", flush=True)

    # ---------------------------------------------------------- phases
    def sniff_flight(self):
        import pandas as pd

        p = (
            PROJECT_DIR / "processed_data" / "flight_data" / self.model
            / f"{self.model}_{self.date}.csv"
        )
        fd = pd.read_csv(p, low_memory=False)
        va = fd.get("kite_apparent_windspeed")
        has_va = va is not None and np.isfinite(va).mean() > 0.5 and np.nanmean(va) > 1
        dep = fd.get("kcu_actual_depower")
        dep_active = dep is not None and np.nanstd(dep) > 3.0
        lidar = any("Wind Speed" in c or "Wind_Speed" in c for c in fd.columns)
        us = fd.get("kcu_actual_steering")
        has_us = us is not None and np.nanstd(us) > 1.0
        return {"has_va": bool(has_va), "dep_active": bool(dep_active),
                "lidar": bool(lidar and not self.args.no_lidar),
                "has_us": bool(has_us)}

    def cfg(self, **kw):
        """Deep-copied config with simulation/tuning overrides applied."""
        c = json.loads(json.dumps(self.config))
        for k, v in kw.items():
            if k in ("vw", "CL", "CD", "CS"):
                c["tuning_parameters"]["model_stdv"][k] = v
            else:
                c["simulation_parameters"][k] = v
        return c

    def run(self):
        flight = self.sniff_flight()
        self.config["simulation_parameters"]["measurements"][
            "kite_apparent_windspeed"
        ] = flight["has_va"]
        self.config["simulation_parameters"]["measurements"]["dynamic_depower"] = (
            not flight["has_va"] and flight["dep_active"]
        )
        self.decide("P0-sniff", flight, "flight capabilities", {})

        # ---- P0 audit + P1 wind walk
        if flight["lidar"]:
            # grid decided by the va-scale-insensitive lidar criterion
            vw_grid = [0.1, 0.05] + ([0.02] if flight["has_va"] else [])
            self.solve_many([(f"vw{v}", self.cfg(vw=v)) for v in vw_grid])
            scores = {v: self.score(f"vw{v}") for v in vw_grid}
            tab = {v: {"dir_rms": s["lidar"]["dir_rms_deg"],
                       "nis": s["internal"]["nis_median"]}
                   for v, s in scores.items()}
            ok = [v for v in vw_grid
                  if NIS_BAND[0] < tab[v]["nis"] < NIS_BAND[1]] or vw_grid
            vw = min(ok, key=lambda v: tab[v]["dir_rms"])
            reason = "min lidar direction RMS (va-scale-insensitive), NIS in band"
        else:
            # BLIND: vw is a methodological prior, not per-flight tunable.
            # Internal criteria are weakly identified for it: the loose-vw
            # wind error is mostly NOT loop-locked (low-frequency wander),
            # and the turbulence reference is itself inflated by the leak.
            # 0.02 (va measured) / 0.05 (no va) were lidar-validated on the
            # V3 (2019+2025) and V9 (2023+2024) at two sites.
            vw = 0.02 if flight["has_va"] else 0.05
            self.solve_many([(f"vw{vw}", self.cfg(vw=vw))])
            s = self.score(f"vw{vw}")["internal"]
            tab = {"nis": s["nis_median"], "ti": s["ti_proxy"]}
            reason = ("TRANSFERRED PRIOR (lidar-validated V3+V9); blind "
                      "internal criteria cannot identify vw")
            if not (NIS_BAND[0] < s["nis_median"] < NIS_BAND[1]):
                reason += f" -- WARNING: NIS {s['nis_median']:.2f} out of band"
        self.decide("P1-vw", vw, reason, tab)
        best = f"vw{vw}"
        base_kw = {"vw": vw}

        # ---- P2 pitot (dynamic-pressure scale on the measured va, b = 0)
        pitot_k, anchored = None, False
        if flight["has_va"]:
            if flight["lidar"]:
                b1 = self.score(best)["lidar"]["speed_bias"]
                s_hist = [(1.0, b1)]
                k_hist = []
                for it in range(3):
                    if abs(s_hist[-1][1]) < PITOT_BIAS_TOL and k_hist:
                        break
                    if len(s_hist) == 1:
                        s_next = 1.0 - np.clip(b1 / SLOPE_PRIOR, -0.12, 0.12)
                    else:
                        (sa, ba), (sb, bb) = s_hist[-2], s_hist[-1]
                        s_next = sb - bb * (sb - sa) / (bb - ba)
                    k_next = float(round(1.0 / s_next**2, 4))
                    name = f"pitot{k_next}"
                    self.solve_many([(name, self.cfg(**base_kw,
                                                     pitot_calibration_k=k_next,
                                                     pitot_calibration_b=0.0))])
                    b = self.score(name)["lidar"]["speed_bias"]
                    s_hist.append((s_next, b))
                    k_hist.append((k_next, name, b))
                pitot_k, best, bias = min(
                    ((k, n, b) for k, n, b in k_hist), key=lambda x: abs(x[2])
                )
                anchored = True
                base_kw.update(pitot_calibration_k=pitot_k, pitot_calibration_b=0.0)
                self.decide("P2-pitot", pitot_k,
                            f"lidar secant, final bias {bias:+.2f} m/s (ANCHORED)",
                            {"points": s_hist})
            else:
                r1 = self.score(best)["internal"]["ro_corr"]
                name2 = "pitotblind"
                self.solve_many([(name2, self.cfg(**base_kw,
                                                  pitot_calibration_k=0.826,
                                                  pitot_calibration_b=0.0))])
                r2 = self.score(name2)["internal"]["ro_corr"]
                s1, s2 = 1.0, 1.0 / np.sqrt(0.826)
                s_zero = s1 - r1 * (s2 - s1) / (r2 - r1)
                pitot_k = float(round(1.0 / s_zero**2, 4))
                name = f"pitot{pitot_k}"
                self.solve_many([(name, self.cfg(**base_kw,
                                                 pitot_calibration_k=pitot_k,
                                                 pitot_calibration_b=0.0))])
                best = name
                base_kw.update(pitot_calibration_k=pitot_k, pitot_calibration_b=0.0)
                self.decide("P2-pitot", pitot_k,
                            "blind wind-va loop-correlation zero -- UNANCHORED, "
                            "model-bias floor ~3-4 % in k, +-0.04",
                            {"ro_corr": [(1.0, r1), (0.826, r2)]})

        # ---- P3 steering stages
        if flight["has_us"]:
            pre = self.score(best)["internal"]
            stage_kw = dict(steering_dependent_cs=True, steering_dependent_clcd=True,
                            steering_dependent_cl_asym=True,
                            CL=0.005, CD=0.002, CS=0.003)
            self.solve_many([("stages", self.cfg(**base_kw, **stage_kw))])
            post = self.score("stages")["internal"]
            keep_asym = (post.get("drift_k_cl_us_odd") or 999) < DRIFT_KEEP_PCT
            if not keep_asym:
                stage_kw["steering_dependent_cl_asym"] = False
                self.solve_many([("stages_noasym", self.cfg(**base_kw, **stage_kw))])
                stage_name = "stages_noasym"
                post = self.score(stage_name)["internal"]
            else:
                stage_name = "stages"
            gain = pre["leak_cs_us"] - post["leak_cs_us"]
            if gain > 0.05 or post["leak_cs_us"] < pre["leak_cs_us"]:
                best = stage_name
                base_kw.update(stage_kw)
                self.decide("P3-stages", stage_name,
                            f"CS-us leak {pre['leak_cs_us']:.2f}->"
                            f"{post['leak_cs_us']:.2f}, asym "
                            f"{'kept' if keep_asym else 'dropped (drift)'}",
                            {"pre": pre["leak_cs_us"], "post": post["leak_cs_us"]})
            else:
                self.decide("P3-stages", "off", "no CS-leak improvement", {})

        # ---- P4 actuation lag
        if flight["has_us"] and base_kw.get("steering_dependent_cs"):
            self.solve_many([(f"lag{l}", self.cfg(**base_kw, steering_input_lag=l))
                             for l in LAG_GRID])
            s0 = self.score(best)
            cands = {0.0: s0}
            for l in LAG_GRID:
                cands[l] = self.score(f"lag{l}")
            cs0 = s0["internal"]["leak_cs_us"]
            red = {l: cs0 - s["internal"]["leak_cs_us"] for l, s in cands.items()}

            def guarded(l):
                s = cands[l]
                if flight["lidar"]:
                    L0, L = s0["lidar"], s["lidar"]
                    return (L["speed_rms"] <= L0["speed_rms"] + 0.05
                            and L["dir_rms_deg"] <= L0["dir_rms_deg"] + 0.3
                            and abs(L["speed_bias"]) <= abs(L0["speed_bias"]) + 0.15)
                i0, i = s0["internal"], s["internal"]
                return (i["wind_hp_std"] <= 1.10 * i0["wind_hp_std"]
                        and i["leak_dir_amp_deg"] <= 1.10 * i0["leak_dir_amp_deg"]
                        + 0.1)

            ok = [l for l in cands if guarded(l)]
            max_red = max(red[l] for l in ok)
            if max_red <= 0.02:
                lag = 0.0
                reason = "no guarded CS-leak reduction"
            else:
                lag = min(l for l in ok if red[l] >= LAG_ELBOW * max_red)
                reason = (f"smallest lag with >={LAG_ELBOW:.0%} of max guarded "
                          f"CS-leak reduction")
            if lag > 0:
                best = f"lag{lag}"
                base_kw["steering_input_lag"] = lag
            self.decide("P4-lag", lag, reason,
                        {l: {"cs_leak": cands[l]["internal"]["leak_cs_us"],
                             "guard_ok": l in ok} for l in cands})

        # ---- P5 CD walk
        s_cur = self.score(best)
        self.solve_many([("cdtight", self.cfg(**{**base_kw, "CD": CD_TIGHT}))])
        s_t = self.score("cdtight")
        i0, it = s_cur["internal"], s_t["internal"]
        dip_gain = (i0.get("cd_dip_frac", 0) - it.get("cd_dip_frac", 0))
        guards = (it["wind_hp_std"] <= 1.10 * i0["wind_hp_std"]
                  and it["ti_proxy"] >= 0.85 * i0["ti_proxy"]
                  and it["leak_wz_us"] <= i0["leak_wz_us"] + 0.20)
        if flight["lidar"]:
            guards = guards and (
                abs(s_t["lidar"]["speed_bias"]) <= abs(s_cur["lidar"]["speed_bias"]) + 0.10
                and s_t["lidar"]["dir_rms_deg"] <= s_cur["lidar"]["dir_rms_deg"] + 0.3)
        if dip_gain >= 0.3 * max(i0.get("cd_dip_frac", 0), 1e-9) and guards:
            best = "cdtight"
            base_kw["CD"] = CD_TIGHT
            self.decide("P5-cd", CD_TIGHT,
                        f"dip fraction {i0.get('cd_dip_frac'):.3f}->"
                        f"{it.get('cd_dip_frac'):.3f}, guards hold", {})
        else:
            self.decide("P5-cd", base_kw.get("CD", 0.002),
                        "tight CD rejected (insufficient dip gain or guards)",
                        {"dip": (i0.get("cd_dip_frac"), it.get("cd_dip_frac")),
                         "guards_ok": guards})

        # ---- P6 final: re-verify pitot, emit config + report
        final = self.score(best)
        if flight["lidar"] and flight["has_va"]:
            b = final["lidar"]["speed_bias"]
            if abs(b) > 0.10 and pitot_k:
                s_now = 1.0 / np.sqrt(pitot_k)
                k_new = float(round(1.0 / (s_now - b / SLOPE_PRIOR) ** 2, 4))
                self.solve_many([("final", self.cfg(**{**base_kw,
                                                       "pitot_calibration_k": k_new,
                                                       "pitot_calibration_b": 0.0}))])
                base_kw["pitot_calibration_k"] = pitot_k = k_new
                best = "final"
                final = self.score(best)
                self.decide("P6-pitot-recheck", k_new,
                            f"re-anchored at final config, bias "
                            f"{final['lidar']['speed_bias']:+.2f}", {})

        cfg_final = self.cfg(**base_kw)
        out_yaml = self.out_dir / "ekf_config_autotuned.yaml"
        header = (
            f"# Auto-tuned {self.date} ({self.model}) by examples/auto_tune.py\n"
            f"# winning run: {self.solved[best].name}\n"
            f"# pitot: k={pitot_k} "
            f"({'LIDAR-ANCHORED' if anchored else 'unanchored/none'})\n"
        )
        out_yaml.write_text(header + yaml.safe_dump(
            _plain({"simulation_parameters": cfg_final["simulation_parameters"],
                    "tuning_parameters": cfg_final["tuning_parameters"]}),
            sort_keys=False))
        (self.out_dir / "decisions.json").write_text(
            json.dumps({"decisions": self.log, "final_metrics": final,
                        "winning_run": str(self.solved[best])},
                       indent=1, default=str))
        print(f"\nFinal run: {self.solved[best]}")
        print(f"Config:    {out_yaml}")
        print(f"Evidence:  {self.out_dir / 'decisions.json'}")
        if final["lidar"]:
            L = final["lidar"]
            print(f"Lidar: bias {L['speed_bias']:+.2f}, RMS {L['speed_rms']:.2f}, "
                  f"dir RMS {L['dir_rms_deg']:.1f} deg")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solve-one", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--base-h5", default=None)
    ap.add_argument("--config-folder", default=None)
    ap.add_argument("--flight-model", default=None)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (with --config-folder)")
    ap.add_argument("--no-lidar", action="store_true",
                    help="force the blind branch even if lidar columns exist")
    ap.add_argument("--start-min", type=float, default=0.0)
    ap.add_argument("--end-min", type=float, default=0.0)
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing _auto_* h5s instead of re-solving")
    ap.add_argument("--tag", default="",
                    help="namespace for run names/outputs, e.g. --tag blind")
    args = ap.parse_args()

    if args.solve_one:
        solve_worker(args.solve_one)
        return

    os.chdir(PROJECT_DIR)
    if args.base_h5:
        config = config_from_h5(Path(args.base_h5))
    else:
        from awes_ekf.setup.settings import load_config

        config = load_config(args.config_folder)
        if not args.date:
            raise SystemExit("--date is required with --config-folder")
        config["year"], config["month"], config["day"] = args.date.split("-")
    if args.flight_model:
        config["kite"]["model_name"] = args.flight_model
    config = start_overrides(config)
    Ladder(args, config).run()


if __name__ == "__main__":
    main()
