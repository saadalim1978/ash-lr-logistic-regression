"""Optimizers for L2-regularized logistic regression.

All methods share the same interface::

    result = method(obj, w0, tol=1e-6, max_iter=1000, **options)

and return an :class:`OptResult` with the final weights, a convergence flag,
the reason for stopping, the work counters of the objective and a per-iteration
history (objective, gradient norm, cumulative data passes, wall time, ...).

Methods
-------
gd           gradient descent with Armijo backtracking
gd_fixed     gradient descent with fixed step 1/L
newton       Newton's method, direct linear solve, Armijo backtracking
newton_cg    inexact Newton-CG (Hessian-vector products), Armijo backtracking
lbfgs        limited-memory BFGS (two-loop recursion), Armijo backtracking
ash_lr       Adaptive Stable Hybrid: gradient phase, then damped Newton
             (direct or CG inner solver; line-search or ratio-test damping)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .objectives import LogisticObjective


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class OptResult:
    method: str
    w: np.ndarray
    f: float
    grad_norm: float
    n_iter: int
    converged: bool
    status: str
    time: float
    work: dict
    history: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "method": self.method,
            "iterations": self.n_iter,
            "loss": self.f,
            "grad_norm": self.grad_norm,
            "time_ms": 1e3 * self.time,
            "converged": self.converged,
            "status": self.status,
            **self.work,
            **{k: v for k, v in self.info.items() if np.isscalar(v)},
        }


class _Recorder:
    """Records one history row per iteration and keeps the clock."""

    def __init__(self, obj: LogisticObjective):
        self.obj = obj
        self.rows = []
        self.t0 = time.perf_counter()

    def __call__(self, k, f, gnorm, **extra):
        self.rows.append(
            {
                "iter": k,
                "f": f,
                "grad_norm": gnorm,
                "passes": self.obj.work.passes,
                "time": time.perf_counter() - self.t0,
                **extra,
            }
        )

    def elapsed(self):
        return time.perf_counter() - self.t0


def _stop_threshold(gnorm0, tol, relative):
    """Absolute rule ||g|| <= tol, or the proposal's relative rule
    ||g|| / max(1, ||g0||) <= tol."""
    return tol * max(1.0, gnorm0) if relative else tol


# ---------------------------------------------------------------------------
# Armijo backtracking (shared by every method)
# ---------------------------------------------------------------------------
def armijo(obj, w, f, g, p, alpha0=1.0, c=1e-4, shrink=0.5, max_trials=40):
    """Backtracking line search with sufficient-decrease condition

        F(w + a p) <= F(w) + c * a * g^T p.

    Returns (alpha, f_new, n_trials); alpha = 0.0 signals failure.
    """
    slope = float(g @ p)
    if not slope < 0:  # not a descent direction (or NaN)
        return 0.0, f, 0
    a = alpha0
    for trial in range(1, max_trials + 1):
        f_new = obj.value(w + a * p)
        if np.isfinite(f_new) and f_new <= f + c * a * slope:
            return a, f_new, trial
        a *= shrink
    return 0.0, f, max_trials


# ---------------------------------------------------------------------------
# Conjugate gradient for (H + mu I) p = -g using Hessian-vector products
# ---------------------------------------------------------------------------
def cg_solve(obj, w, g, mu=0.0, rtol=0.5, max_iter=None, D=None):
    """Solve (H + mu I) p = -g approximately with linear CG.

    Stops when ||r|| <= rtol * ||g|| or after max_iter iterations.
    H + mu I is symmetric positive definite because lam > 0, so plain CG
    applies (no negative-curvature handling is needed).

    Returns p, number of CG iterations, final relative residual ||r||/||g||,
    and the residual vector r = -g - (H + mu I) p.
    """
    d = g.size
    max_iter = d if max_iter is None else max_iter
    if D is None:
        D = obj.curvature_weights(w)
    gnorm = np.linalg.norm(g)
    p = np.zeros_like(g)
    r = -g.copy()  # residual b - A p with b = -g, p = 0
    s = r.copy()
    rr = r @ r
    target = rtol * gnorm
    it = 0
    while it < max_iter and np.sqrt(rr) > target:
        As = obj.hess_vec(w, s, D=D) + mu * s
        sAs = s @ As
        if sAs <= 0:  # cannot happen in exact arithmetic; guard anyway
            break
        a = rr / sAs
        p += a * s
        r -= a * As
        rr_new = r @ r
        s = r + (rr_new / rr) * s
        rr = rr_new
        it += 1
    if it == 0:  # tolerance met by p = 0 is impossible unless g = 0; fall back
        p = -g.copy()
    return p, it, float(np.sqrt(rr) / max(gnorm, 1e-300)), r


def forcing_term(gnorm, eta_max=0.5):
    """Inexact-Newton forcing sequence eta_k = min(eta_max, sqrt(||g||)),
    which gives local superlinear convergence (Nocedal & Wright, Thm 7.2)."""
    return min(eta_max, np.sqrt(gnorm))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def gd(obj, w0, tol=1e-6, max_iter=1000, relative=False, c=1e-4, max_trials=40):
    """Gradient descent, Armijo backtracking from alpha = 1."""
    obj.reset_work()
    rec = _Recorder(obj)
    w = np.array(w0, dtype=obj.dtype)
    f, g = obj.value_grad(w)
    gn = float(np.linalg.norm(g))
    thr = _stop_threshold(gn, tol, relative)
    rec(0, f, gn)
    status, k = "max_iter", 0
    for k in range(1, max_iter + 1):
        if gn <= thr:
            status, k = "converged", k - 1
            break
        p = -g
        a, f_new, _ = armijo(obj, w, f, g, p, c=c, max_trials=max_trials)
        if a == 0.0:
            status, k = "line_search_failed", k - 1
            break
        w = w + a * p
        f, g = f_new, obj.grad(w)
        gn = float(np.linalg.norm(g))
        rec(k, f, gn, alpha=a)
    else:
        if gn <= thr:
            status = "converged"
    return _finish("GD", obj, w, f, gn, k, status, rec)


def gd_fixed(obj, w0, tol=1e-6, max_iter=1000, relative=False, step=None):
    """Gradient descent with fixed step 1/L (L = ||X||^2/(4n) + lam)."""
    L = obj.lipschitz()  # computed before the clock starts
    step = 1.0 / L if step is None else step
    obj.reset_work()
    rec = _Recorder(obj)
    w = np.array(w0, dtype=obj.dtype)
    f, g = obj.value_grad(w)
    gn = float(np.linalg.norm(g))
    thr = _stop_threshold(gn, tol, relative)
    rec(0, f, gn)
    status, k = "max_iter", 0
    for k in range(1, max_iter + 1):
        if gn <= thr:
            status, k = "converged", k - 1
            break
        w = w - step * g
        f, g = obj.value_grad(w)
        gn = float(np.linalg.norm(g))
        if not np.isfinite(f):
            status = "non_finite"
            break
        rec(k, f, gn, alpha=step)
    else:
        if gn <= thr:
            status = "converged"
    return _finish("GD-fixed", obj, w, f, gn, k, status, rec, info={"step": step})


def newton(obj, w0, tol=1e-6, max_iter=1000, relative=False, c=1e-4, max_trials=40):
    """Newton's method: solve H p = -g directly (numpy.linalg.solve)."""
    obj.reset_work()
    rec = _Recorder(obj)
    w = np.array(w0, dtype=obj.dtype)
    f, g = obj.value_grad(w)
    gn = float(np.linalg.norm(g))
    thr = _stop_threshold(gn, tol, relative)
    rec(0, f, gn)
    status, k = "max_iter", 0
    for k in range(1, max_iter + 1):
        if gn <= thr:
            status, k = "converged", k - 1
            break
        H = obj.hessian(w)
        p = np.linalg.solve(H, -g)
        obj.count_solve()
        a, f_new, _ = armijo(obj, w, f, g, p, c=c, max_trials=max_trials)
        if a == 0.0:
            status, k = "line_search_failed", k - 1
            break
        w = w + a * p
        f, g = f_new, obj.grad(w)
        gn = float(np.linalg.norm(g))
        rec(k, f, gn, alpha=a)
    else:
        if gn <= thr:
            status = "converged"
    return _finish("Newton", obj, w, f, gn, k, status, rec)


def newton_cg(obj, w0, tol=1e-6, max_iter=1000, relative=False, c=1e-4,
              max_trials=40, cg_max_iter=None, eta_max=0.5):
    """Inexact (truncated) Newton-CG with Armijo backtracking, no damping."""
    obj.reset_work()
    rec = _Recorder(obj)
    w = np.array(w0, dtype=obj.dtype)
    f, g = obj.value_grad(w)
    gn = float(np.linalg.norm(g))
    thr = _stop_threshold(gn, tol, relative)
    rec(0, f, gn)
    status, k, total_cg = "max_iter", 0, 0
    for k in range(1, max_iter + 1):
        if gn <= thr:
            status, k = "converged", k - 1
            break
        p, it, res, _ = cg_solve(obj, w, g, 0.0, forcing_term(gn, eta_max), cg_max_iter)
        total_cg += it
        a, f_new, _ = armijo(obj, w, f, g, p, c=c, max_trials=max_trials)
        if a == 0.0:
            status, k = "line_search_failed", k - 1
            break
        w = w + a * p
        f, g = f_new, obj.grad(w)
        gn = float(np.linalg.norm(g))
        rec(k, f, gn, alpha=a, cg_iter=it, cg_res=res)
    else:
        if gn <= thr:
            status = "converged"
    return _finish("Newton-CG", obj, w, f, gn, k, status, rec, info={"cg_total": total_cg})


def lbfgs(obj, w0, tol=1e-6, max_iter=1000, relative=False, m=10, c=1e-4, max_trials=40):
    """L-BFGS (two-loop recursion, memory m) with Armijo backtracking.

    Pairs with s^T y <= 1e-10 * ||s|| ||y|| are skipped to keep the
    implicit inverse-Hessian approximation positive definite.
    """
    obj.reset_work()
    rec = _Recorder(obj)
    w = np.array(w0, dtype=obj.dtype)
    f, g = obj.value_grad(w)
    gn = float(np.linalg.norm(g))
    thr = _stop_threshold(gn, tol, relative)
    rec(0, f, gn)
    S, Y = deque(maxlen=m), deque(maxlen=m)
    status, k = "max_iter", 0
    for k in range(1, max_iter + 1):
        if gn <= thr:
            status, k = "converged", k - 1
            break
        # two-loop recursion: p = -H_k g
        q = g.copy()
        alphas = []
        for s, yv in zip(reversed(S), reversed(Y)):
            rho = 1.0 / (yv @ s)
            a_i = rho * (s @ q)
            q -= a_i * yv
            alphas.append((rho, a_i))
        gamma = (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1]) if S else 1.0
        r = gamma * q
        for (s, yv), (rho, a_i) in zip(zip(S, Y), reversed(alphas)):
            b = rho * (yv @ r)
            r += (a_i - b) * s
        p = -r
        a, f_new, _ = armijo(obj, w, f, g, p, c=c, max_trials=max_trials)
        if a == 0.0:  # restart once with steepest descent before giving up
            S.clear(); Y.clear()
            p = -g
            a, f_new, _ = armijo(obj, w, f, g, p, c=c, max_trials=max_trials)
            if a == 0.0:
                status, k = "line_search_failed", k - 1
                break
        w_new = w + a * p
        g_new = obj.grad(w_new)
        s, yv = w_new - w, g_new - g
        if s @ yv > 1e-10 * np.linalg.norm(s) * np.linalg.norm(yv):
            S.append(s); Y.append(yv)
        w, f, g = w_new, f_new, g_new
        gn = float(np.linalg.norm(g))
        rec(k, f, gn, alpha=a)
    else:
        if gn <= thr:
            status = "converged"
    return _finish("L-BFGS", obj, w, f, gn, k, status, rec)


# ---------------------------------------------------------------------------
# ASH-LR
# ---------------------------------------------------------------------------
def ash_lr(
    obj,
    w0,
    tol=1e-6,
    max_iter=1000,
    relative=False,
    *,
    inner="cg",              # "cg" (Hessian-vector products) or "direct"
    damping="ratio",         # "ratio" (actual vs predicted) or "linesearch"
    mu_rule="gradient",      # "gradient": mu = sqrt(theta*||g||); "fixed": mu adapted directly
    gd_phase=True,           # False: start directly in the damped-Newton phase
    switch_grad_frac=0.1,    # switch when ||g|| <= frac * ||g0|| ...
    switch_min_steps=5,      # ... or after >= this many GD steps ...
    switch_rel_decrease=1e-3,  # ... when (F_old - F_new)/max(1,|F_new|) <= this
    mu0=1e-3,
    mu_min=1e-12,
    mu_max=1e8,
    mu_down=0.5,             # mu <- mu * mu_down after a very successful step
    mu_up=4.0,               # mu <- mu * mu_up after a poor/shortened step
    eta_accept=1e-4,         # ratio test: accept step if rho >= eta_accept
    rho_low=0.25,            # rho < rho_low  -> increase mu
    rho_high=0.75,           # rho > rho_high -> decrease mu
    max_reject=30,           # rejected trials per iteration before Armijo fallback
    cg_max_iter=None,
    eta_max=0.5,
    c=1e-4,
    max_trials=40,
    name=None,
):
    """Adaptive Stable Hybrid Logistic Regression.

    Phase 1 (gradient): p = -g with Armijo backtracking. The method switches
    once ||g|| <= switch_grad_frac * ||g0||, or after at least
    ``switch_min_steps`` gradient steps when the relative objective decrease
    of the last step is at most ``switch_rel_decrease``. The switch is one-way.

    Phase 2 (damped Newton): solve (H + mu I) p = -g, directly or with
    inexact CG (forcing term min(eta_max, sqrt(||g||))).

    Damping update
      * "linesearch" (Milestone 2): Armijo on p; mu is halved after a full
        step (alpha = 1) and multiplied by four after a shortened step.
      * "ratio" (final method): trust-region style test on
            rho = (F(w) - F(w+p)) / (m(0) - m(p)),
            m(0) - m(p) = -(g^T p + 0.5 p^T H p).
        The step is accepted if rho >= eta_accept; mu grows by mu_up when
        rho < rho_low and shrinks by mu_down when rho > rho_high. After
        ``max_reject`` rejections the last direction is used with Armijo.
    mu is always clipped to [mu_min, mu_max].

    mu_rule
      * "fixed": the update factors act on mu itself.
      * "gradient": gradient-regularized damping mu = sqrt(theta * ||g||)
        (cf. Mishchenko 2023; Doikov, Mishchenko & Nesterov 2024). The factors
        act on theta, initialised so that mu = mu0 when the Newton phase starts;
        mu then vanishes automatically as ||g|| -> 0.
    """
    if (inner not in ("cg", "direct") or damping not in ("ratio", "linesearch")
            or mu_rule not in ("fixed", "gradient")):
        raise ValueError("invalid inner/damping option")
    obj.reset_work()
    rec = _Recorder(obj)
    w = np.array(w0, dtype=obj.dtype)
    f, g = obj.value_grad(w)
    gn = float(np.linalg.norm(g))
    gn0 = gn
    thr = _stop_threshold(gn, tol, relative)
    rec(0, f, gn, phase="gd" if gd_phase else "newton", mu=mu0)

    phase = "gd" if gd_phase else "newton"
    mu = mu0
    theta = None  # gradient rule: set when the Newton phase starts
    gd_steps = 0
    switch_iter = 0 if not gd_phase else None
    stats = {"cg_total": 0, "rejected": 0, "fallbacks": 0, "newton_iters": 0}
    status, k = "max_iter", 0
    clip = lambda v: min(mu_max, max(mu_min, v))

    def direction(w, g, gn, mu, D, H):
        """Damped Newton direction and the quantity p^T H p for the model."""
        if inner == "direct":
            p = np.linalg.solve(H + mu * np.eye(obj.d, dtype=obj.dtype), -g)
            obj.count_solve()
            pHp = float(p @ (H @ p))  # O(d^2), negligible next to forming H
            return p, pHp, 0, 0.0
        p, it, res, r = cg_solve(obj, w, g, mu, forcing_term(gn, eta_max), cg_max_iter, D)
        # (H + mu I) p = -g - r  =>  p^T H p = -p^T g - p^T r - mu ||p||^2 (no extra Hv)
        pHp = float(-(p @ g) - (p @ r) - mu * (p @ p))
        return p, pHp, it, res

    for k in range(1, max_iter + 1):
        if gn <= thr:
            status, k = "converged", k - 1
            break

        # ---------------- gradient phase ----------------
        if phase == "gd":
            p = -g
            a, f_new, _ = armijo(obj, w, f, g, p, c=c, max_trials=max_trials)
            if a == 0.0:
                status, k = "line_search_failed", k - 1
                break
            rel_dec = (f - f_new) / max(1.0, abs(f_new))
            w = w + a * p
            f, g = f_new, obj.grad(w)
            gn = float(np.linalg.norm(g))
            gd_steps += 1
            rec(k, f, gn, phase="gd", alpha=a, mu=mu)
            if gn <= switch_grad_frac * gn0 or (
                gd_steps >= switch_min_steps and rel_dec <= switch_rel_decrease
            ):
                phase, switch_iter = "newton", k
            continue

        # ---------------- damped Newton phase ----------------
        stats["newton_iters"] += 1
        if mu_rule == "gradient":
            if theta is None:
                theta = mu0**2 / max(gn, 1e-300)
            mu = clip(np.sqrt(theta * gn))
        D = obj.curvature_weights(w) if inner == "cg" else None
        H = obj.hessian(w) if inner == "direct" else None
        cg_it_iter = 0

        if damping == "linesearch":
            p, _, it, res = direction(w, g, gn, mu, D, H)
            cg_it_iter += it
            a, f_new, _ = armijo(obj, w, f, g, p, c=c, max_trials=max_trials)
            if a == 0.0:
                status, k = "line_search_failed", k - 1
                break
            fac = mu_down if a == 1.0 else mu_up
            if mu_rule == "gradient":
                theta *= fac
            mu = clip(mu * fac)
            rho = np.nan
        else:  # ratio test
            accepted = False
            for _ in range(max_reject):
                p, pHp, it, res = direction(w, g, gn, mu, D, H)
                cg_it_iter += it
                pred = -(float(g @ p) + 0.5 * pHp)
                f_try = obj.value(w + p)
                rho = (f - f_try) / pred if pred > 0 else -np.inf
                if not np.isfinite(f_try):
                    rho = -np.inf
                fac = mu_up if rho < rho_low else (mu_down if rho > rho_high else 1.0)
                if mu_rule == "gradient":
                    theta *= fac
                    mu = clip(np.sqrt(theta * gn))
                else:
                    mu = clip(mu * fac)
                if rho >= eta_accept:
                    a, f_new, accepted = 1.0, f_try, True
                    break
                stats["rejected"] += 1
            if not accepted:  # safeguard: Armijo along the last direction
                stats["fallbacks"] += 1
                a, f_new, _ = armijo(obj, w, f, g, p, c=c, max_trials=max_trials)
                if a == 0.0:
                    status, k = "line_search_failed", k - 1
                    break

        stats["cg_total"] += cg_it_iter
        w = w + a * p
        f, g = f_new, obj.grad(w)
        gn = float(np.linalg.norm(g))
        rec(k, f, gn, phase="newton", alpha=a, mu=mu, rho=rho, cg_iter=cg_it_iter)
    else:
        if gn <= thr:
            status = "converged"

    if name is None:
        cfg = (inner, damping, mu_rule, gd_phase)
        name = "ASH-LR" if cfg == ("cg", "ratio", "gradient", True) else \
            f"ASH-LR({inner},{damping},{mu_rule}{'' if gd_phase else ',noGD'})"
    info = {**stats, "switch_iter": -1 if switch_iter is None else switch_iter, "mu_final": mu}
    return _finish(name, obj, w, f, gn, k, status, rec, info=info)


# ---------------------------------------------------------------------------
def _finish(name, obj, w, f, gn, k, status, rec, info=None):
    elapsed = rec.elapsed()
    if not (np.isfinite(f) and np.all(np.isfinite(w))):
        status = "non_finite"
    return OptResult(
        method=name,
        w=w,
        f=float(f),
        grad_norm=float(gn),
        n_iter=int(k),
        converged=(status == "converged"),
        status=status,
        time=elapsed,
        work=obj.work.as_dict(),
        history=rec.rows,
        info=info or {},
    )


METHODS = {
    "GD": gd,
    "GD-fixed": gd_fixed,
    "Newton": newton,
    "Newton-CG": newton_cg,
    "L-BFGS": lbfgs,
    "ASH-LR": ash_lr,
}
