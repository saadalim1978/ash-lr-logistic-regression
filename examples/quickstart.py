"""Quickstart: fit L2-regularized logistic regression with ASH-LR and compare
with two baselines on an ill-conditioned synthetic problem.

Run from the repository root:
    python examples/quickstart.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ashlr import LogisticObjective, ash_lr, lbfgs, make_synthetic, newton_cg  # noqa: E402

# 1. Data: 1000 samples, 10 features, condition number of X^T X / n equal to 1e4
ds, w_true = make_synthetic(n=1000, d=10, kappa=1e4, seed=0)
print(f"train {ds.X_train.shape}, design condition number {ds.meta['kappa_design']:.0f}")

# 2. Objective with lambda = 1e-6 (numerically stable formulas by default)
obj = LogisticObjective(ds.X_train, ds.y_train, lam=1e-6)
w0 = np.zeros(ds.X_train.shape[1])

# 3. Optimize and compare
for method in (ash_lr, newton_cg, lbfgs):
    res = method(obj, w0, tol=1e-6)
    acc = np.mean(np.sign(ds.X_test @ res.w) == ds.y_test)
    print(f"{res.method:10s} converged={res.converged!s:5s} iters={res.n_iter:4d} "
          f"passes={res.work['passes']:7.0f} time={1e3 * res.time:6.1f} ms "
          f"loss={res.f:.8f} test acc={acc:.3f}")

# 4. ASH-LR diagnostics: where it switched and how the damping evolved
res = ash_lr(obj, w0)
print("switched to damped Newton-CG at iteration", res.info["switch_iter"])
print("mu per Newton iteration:",
      ", ".join(f"{h['mu']:.1e}" for h in res.history if h.get("phase") == "newton"))
