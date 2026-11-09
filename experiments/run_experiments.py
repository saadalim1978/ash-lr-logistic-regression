"""Run the pre-registered experiments E1-E5 (see EXPERIMENT_PLAN.md).

Usage:
    python experiments/run_experiments.py e1 e2 e3 e4 e5     # full study
    python experiments/run_experiments.py all --quick         # 2 seeds, 1 rep (smoke test)

Each experiment writes results/<name>.csv and results/raw/<name>_histories.pkl.gz.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import (ABLATIONS, LAM, MAIN_METHODS, REGIMES, SEEDS, LogisticObjective,
                    load_regime, log, opt, run_problem, save, start_point, wrap)


def e1(seeds, reps):
    """Main comparison: all regimes, w0 = 0, lambda = 1e-6."""
    rows, hists = [], {}
    for rid in REGIMES:
        for seed in seeds:
            t0 = time.time()
            ds = load_regime(rid, seed)
            obj = LogisticObjective(ds.X_train, ds.y_train, LAM)
            w0 = np.zeros(ds.X_train.shape[1])
            r, h = run_problem(obj, w0, MAIN_METHODS, reps=reps, ds=ds,
                               rng=np.random.default_rng(seed),
                               extra={"regime": rid, "seed": seed, "n": ds.X_train.shape[0],
                                      "d": ds.X_train.shape[1],
                                      "hess_cond0": obj.hessian_condition(w0)})
            rows += r
            hists.update({(rid, seed, m): v for m, v in h.items()})
            log(f"E1 {rid} seed {seed}: {time.time() - t0:.1f}s")
    return save(rows, hists, "e1_main")


def e2(seeds, reps):
    """Robustness: starting points x lambda, relative stopping rule."""
    methods = {k: MAIN_METHODS[k] for k in ("Newton", "Newton-CG", "L-BFGS", "ASH-LR")}
    rows, hists = [], {}
    for rid in ("R2", "R5", "R8"):
        for seed in seeds[:3]:
            ds = load_regime(rid, seed)
            for lam in (1e-6, 1e-4, 1e-2):
                obj = LogisticObjective(ds.X_train, ds.y_train, lam)
                for start in ("zero", "near", "far"):
                    t0 = time.time()
                    w0 = start_point(ds, start, seed)
                    r, h = run_problem(obj, w0, methods, reps=reps, ds=ds, relative=True,
                                       rng=np.random.default_rng(seed),
                                       extra={"regime": rid, "seed": seed, "lam": lam, "start": start})
                    rows += r
                    hists.update({(rid, seed, lam, start, m): v for m, v in h.items()})
                    log(f"E2 {rid} seed {seed} lam {lam:g} {start}: {time.time() - t0:.1f}s")
    return save(rows, hists, "e2_robustness")


def e3(seeds, reps):
    """Ablation of ASH-LR components."""
    rows, hists = [], {}
    for rid in ("R2", "R5", "R6", "R8"):
        for seed in seeds:
            t0 = time.time()
            ds = load_regime(rid, seed)
            obj = LogisticObjective(ds.X_train, ds.y_train, LAM)
            w0 = np.zeros(ds.X_train.shape[1])
            r, h = run_problem(obj, w0, ABLATIONS, reps=reps, ds=ds,
                               rng=np.random.default_rng(seed),
                               extra={"regime": rid, "seed": seed})
            rows += r
            hists.update({(rid, seed, m): v for m, v in h.items()})
            log(f"E3 {rid} seed {seed}: {time.time() - t0:.1f}s")
    return save(rows, hists, "e3_ablation")


def e4(seeds, reps):
    """float32 sensitivity: same problems in float32 and float64, tol 1e-4."""
    names = ("Newton", "Newton-CG", "ASH-LR")
    fns = {"Newton": opt.newton, "Newton-CG": opt.newton_cg, "ASH-LR": opt.ash_lr}
    rows, hists = [], {}
    for rid in ("R2", "R7", "R8"):
        for seed in seeds[:3]:
            ds = load_regime(rid, seed)
            obj64 = LogisticObjective(ds.X_train, ds.y_train, LAM)
            d = ds.X_train.shape[1]
            f_star = opt.newton(obj64, np.zeros(d), tol=1e-10).f
            for dtype in (np.float64, np.float32):
                obj = LogisticObjective(ds.X_train, ds.y_train, LAM, dtype=dtype)
                methods = {nm: wrap(fns[nm]) for nm in names}
                r, h = run_problem(obj, np.zeros(d, dtype=dtype), methods, reps=reps, ds=ds,
                                   tol=1e-4, f_star=f_star, rng=np.random.default_rng(seed),
                                   extra={"regime": rid, "seed": seed,
                                          "dtype": np.dtype(dtype).name})
                for row, nm in zip(r, names):  # objective of the solution in float64
                    res = fns[nm](obj, np.zeros(d, dtype=dtype), tol=1e-4, max_iter=1000)
                    f64 = obj64.value(res.w.astype(np.float64))
                    row["f64_at_solution"] = f64
                    row["gap64"] = (f64 - f_star) / max(1.0, abs(f_star))
                rows += r
                hists.update({(rid, seed, np.dtype(dtype).name, m): v for m, v in h.items()})
            log(f"E4 {rid} seed {seed} done")
    return save(rows, hists, "e4_float32")


def e5(seeds, reps):
    """Numerical stability: extreme start (logit std 1e3), stable vs naive formulas."""
    rows, hists = [], {}
    for rid in ("R2", "R8"):
        for seed in seeds[:3]:
            ds = load_regime(rid, seed)
            ref = LogisticObjective(ds.X_train, ds.y_train, LAM)
            d = ds.X_train.shape[1]
            f_star = opt.newton(ref, np.zeros(d), tol=1e-10).f
            w0 = start_point(ds, "extreme", seed)
            for dtype in (np.float64, np.float32):
                for stable in (True, False):
                    obj = LogisticObjective(ds.X_train, ds.y_train, LAM, stable=stable, dtype=dtype)
                    tol = 1e-6 if dtype == np.float64 else 1e-4
                    methods = {"ASH-LR": wrap(opt.ash_lr), "Newton": wrap(opt.newton)}
                    with np.errstate(all="ignore"):
                        r, h = run_problem(obj, w0.astype(dtype), methods, reps=reps, ds=ds,
                                           tol=tol, f_star=f_star, warmup=False,
                                           rng=np.random.default_rng(seed),
                                           extra={"regime": rid, "seed": seed, "stable": stable,
                                                  "dtype": np.dtype(dtype).name,
                                                  "f0": obj.value(w0.astype(dtype))})
                    rows += r
                    hists.update({(rid, seed, np.dtype(dtype).name, stable, m): v for m, v in h.items()})
            log(f"E5 {rid} seed {seed} done")
    return save(rows, hists, "e5_stability")


EXPERIMENTS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", choices=[*EXPERIMENTS, "all"])
    ap.add_argument("--quick", action="store_true", help="2 seeds, 1 timed repetition")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    seeds = SEEDS[:2] if a.quick else SEEDS
    reps = 1 if a.quick else a.reps
    todo = list(EXPERIMENTS) if "all" in a.experiments else a.experiments
    for name in todo:
        t0 = time.time()
        df = EXPERIMENTS[name](seeds, reps)
        log(f"== {name}: {len(df)} rows in {time.time() - t0:.0f}s")
