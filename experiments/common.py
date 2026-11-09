"""Shared definitions for the experiments in EXPERIMENT_PLAN.md.

Regimes R1-R8, problem construction, the timing protocol (warm-up + timed
repetitions in randomized order) and time-to-accuracy computation.
"""
from __future__ import annotations

import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashlr import LogisticObjective, load_spambase, make_synthetic  # noqa: E402
from ashlr import optimizers as opt  # noqa: E402

RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
RESULTS.mkdir(exist_ok=True)
RAW.mkdir(exist_ok=True)

SEEDS = [1000 + s for s in range(10)]
LAM = 1e-6
TOL = 1e-6
MAX_ITER = 1000
GAP_LEVELS = (1e-4, 1e-8)

# --------------------------------------------------------------------------
# Regimes (Section 3 of the plan)
# --------------------------------------------------------------------------
REGIMES = {
    "R1": dict(kind="syn", n=1000, d=10, kappa=1e1, desc="synthetic, kappa=1e1"),
    "R2": dict(kind="syn", n=1000, d=10, kappa=1e3, desc="synthetic, kappa=1e3"),
    "R3": dict(kind="syn", n=1000, d=10, kappa=1e6, desc="synthetic, kappa=1e6"),
    "R4": dict(kind="syn", n=4000, d=200, kappa=1e4, desc="synthetic d=200, kappa=1e4"),
    "R5": dict(kind="syn", n=6000, d=500, kappa=1e6, desc="synthetic d=500, kappa=1e6"),
    "R6": dict(kind="syn", n=2000, d=50, kappa=1e2, scale_range=(1e-2, 1e2),
               desc="synthetic d=50, scaled columns"),
    "R7": dict(kind="spam", standardize=True, desc="Spambase, standardized"),
    "R8": dict(kind="spam", standardize=False, desc="Spambase, raw"),
}
PRIMARY_ILL = ["R3", "R4", "R5", "R6", "R8"]


def load_regime(rid, seed):
    cfg = REGIMES[rid]
    if cfg["kind"] == "syn":
        ds, _ = make_synthetic(n=cfg["n"], d=cfg["d"], kappa=cfg["kappa"], seed=seed,
                               scale_range=cfg.get("scale_range"), name=rid)
    else:
        ds = load_spambase(standardize=cfg["standardize"], seed=seed)
        ds.name = rid
    return ds


def start_point(ds, kind, seed):
    """w0 = 0, or a random direction scaled to a target logit std."""
    d = ds.X_train.shape[1]
    if kind == "zero":
        return np.zeros(d)
    target = {"near": 5.0, "far": 50.0, "extreme": 1e3}[kind]
    rng = np.random.default_rng(seed + 7919)
    w = rng.standard_normal(d)
    return w * target / np.std(ds.X_train @ w)


# --------------------------------------------------------------------------
# Running a set of methods on one problem
# --------------------------------------------------------------------------
def gap_crossings(history, f_star):
    """Passes and time when the relative gap first drops below each level."""
    out = {}
    denom = max(1.0, abs(f_star))
    for lvl in GAP_LEVELS:
        hit = next((h for h in history if (h["f"] - f_star) / denom <= lvl), None)
        tag = f"{lvl:.0e}"
        out[f"passes_to_{tag}"] = hit["passes"] if hit else np.nan
        out[f"time_to_{tag}"] = hit["time"] if hit else np.nan
    return out


def run_problem(obj, w0, methods, reps=3, warmup=True, rng=None, ds=None,
                f_star=None, extra=None, tol=TOL, relative=False):
    """Run each method (name -> callable(obj, w0, tol, relative)) on one problem.

    Protocol: one untimed warm-up of every method, then ``reps`` timed rounds,
    each running all methods in a fresh random order. Metrics other than time
    come from the (deterministic) runs; time is the median over the rounds,
    and time-to-accuracy uses the history of the median-time run.
    Returns (rows, histories).
    """
    rng = rng or np.random.default_rng(0)
    names = list(methods)
    if f_star is None:
        f_star = opt.newton(obj, w0, tol=1e-10, max_iter=200).f
    if warmup:
        for nm in names:
            methods[nm](obj, w0, tol, relative)
    runs = {nm: [] for nm in names}
    for _ in range(reps):
        for nm in rng.permutation(names):
            runs[nm].append(methods[nm](obj, w0, tol, relative))
    rows, hists = [], {}
    for nm in names:
        rs = sorted(runs[nm], key=lambda r: r.time)
        r = rs[len(rs) // 2]
        row = {
            "method": nm,
            "iterations": r.n_iter,
            "converged": r.converged,
            "status": r.status,
            "loss": r.f,
            "gap": (r.f - f_star) / max(1.0, abs(f_star)),
            "grad_norm": r.grad_norm,
            "time_ms": 1e3 * r.time,
            "time_ms_min": 1e3 * rs[0].time,
            "time_ms_max": 1e3 * rs[-1].time,
            **r.work,
            "cg_total": r.info.get("cg_total", np.nan),
            "rejected": r.info.get("rejected", np.nan),
            "fallbacks": r.info.get("fallbacks", np.nan),
            "switch_iter": r.info.get("switch_iter", np.nan),
            "finite": bool(np.isfinite(r.f) and np.all(np.isfinite(r.w))),
            **gap_crossings(r.history, f_star),
            "f_star": f_star,
        }
        if ds is not None:
            w = r.w.astype(np.float64)
            row["test_acc"] = float(np.mean(np.where(ds.X_test @ w >= 0, 1, -1) == ds.y_test))
            row["train_acc"] = float(np.mean(np.where(ds.X_train @ w >= 0, 1, -1) == ds.y_train))
        row.update(extra or {})
        rows.append(row)
        hists[nm] = r.history
    return rows, hists


def wrap(fn, **kw):
    """Adapt an optimizer to the (obj, w0, tol, relative) signature."""
    return lambda obj, w0, tol, relative: fn(obj, w0, tol=tol, max_iter=MAX_ITER,
                                             relative=relative, **kw)


MAIN_METHODS = {
    "GD": wrap(opt.gd),
    "GD-fixed": wrap(opt.gd_fixed),
    "Newton": wrap(opt.newton),
    "Newton-CG": wrap(opt.newton_cg),
    "L-BFGS": wrap(opt.lbfgs),
    "ASH-LR": wrap(opt.ash_lr),
}

ABLATIONS = {
    "(a) M2: direct+LS, fixed mu": wrap(opt.ash_lr, inner="direct", damping="linesearch", mu_rule="fixed"),
    "(b) CG+LS, fixed mu": wrap(opt.ash_lr, inner="cg", damping="linesearch", mu_rule="fixed"),
    "(c) CG+ratio, fixed mu": wrap(opt.ash_lr, inner="cg", damping="ratio", mu_rule="fixed"),
    "(d) ASH-LR (full)": wrap(opt.ash_lr),
    "(e) full, no GD phase": wrap(opt.ash_lr, gd_phase=False),
    "(f) direct+ratio, grad mu": wrap(opt.ash_lr, inner="direct"),
}


def save(rows, hists, name):
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    with gzip.open(RAW / f"{name}_histories.pkl.gz", "wb") as fh:
        pickle.dump(hists, fh)
    return df


def log(msg):
    print(msg, flush=True)
