"""Test suite for ASH-LR.  Run with:  pytest -q"""
import numpy as np
import pytest

from ashlr import LogisticObjective, METHODS, ash_lr, make_synthetic, newton
from ashlr.objectives import sigmoid_naive, sigmoid_stable
from ashlr.optimizers import cg_solve


@pytest.fixture(scope="module")
def small():
    ds, _ = make_synthetic(n=200, d=6, kappa=100, seed=1)
    return LogisticObjective(ds.X_train, ds.y_train, lam=1e-3)


# ---------------------------------------------------------------- derivatives
def test_gradient_finite_difference(small):
    rng = np.random.default_rng(0)
    w = rng.standard_normal(small.d)
    g = small.grad(w)
    h = 1e-6
    fd = np.array([(small.value(w + h * e) - small.value(w - h * e)) / (2 * h)
                   for e in np.eye(small.d)])
    assert np.allclose(g, fd, rtol=1e-6, atol=1e-8)


def test_hessian_finite_difference(small):
    rng = np.random.default_rng(1)
    w = rng.standard_normal(small.d)
    H = small.hessian(w)
    h = 1e-6
    fd = np.column_stack([(small.grad(w + h * e) - small.grad(w - h * e)) / (2 * h)
                          for e in np.eye(small.d)])
    assert np.allclose(H, fd, rtol=1e-5, atol=1e-8)
    assert np.allclose(H, H.T)


def test_hessian_vector_product_matches_dense(small):
    rng = np.random.default_rng(2)
    w, v = rng.standard_normal(small.d), rng.standard_normal(small.d)
    assert np.allclose(small.hess_vec(w, v), small.hessian(w) @ v)


def test_cg_solves_damped_system(small):
    rng = np.random.default_rng(3)
    w = rng.standard_normal(small.d)
    g = small.grad(w)
    mu = 1e-2
    p, it, res, _ = cg_solve(small, w, g, mu, rtol=1e-12, max_iter=100)
    exact = np.linalg.solve(small.hessian(w) + mu * np.eye(small.d), -g)
    assert res < 1e-10 and np.allclose(p, exact, rtol=1e-8)


# ---------------------------------------------------------- numerical stability
def test_stable_formulas_finite_at_extreme_logits():
    t = np.array([-1e4, -800.0, -40.0, 0.0, 40.0, 800.0, 1e4])
    s = sigmoid_stable(t)
    assert np.all(np.isfinite(s)) and np.all((s >= 0) & (s <= 1))
    X = np.array([[1e4], [-1e4]])
    y = np.array([-1, -1])
    obj = LogisticObjective(X, y, lam=1e-6)
    f, g = obj.value_grad(np.array([1.0]))
    assert np.isfinite(f) and np.all(np.isfinite(g))
    # mean(softplus(1e4), softplus(-1e4)) + lam/2 * w^2 = 5000 + 5e-7
    assert f == pytest.approx(0.5 * 1e4 + 0.5e-6, rel=1e-14)


def test_naive_formulas_overflow():
    X = np.array([[1e4]])
    obj = LogisticObjective(X, np.array([-1]), lam=1e-6, stable=False)
    assert not np.isfinite(obj.value(np.array([1.0])))
    assert np.isfinite(sigmoid_naive(np.array([-1e4]))).all()  # underflows to 0


def test_labels_validated():
    with pytest.raises(ValueError):
        LogisticObjective(np.ones((2, 1)), np.array([0, 1]))


# ------------------------------------------------------------------ optimizers
@pytest.mark.parametrize("name", list(METHODS))
def test_methods_converge_and_agree(small, name):
    w0 = np.zeros(small.d)
    ref = newton(small, w0, tol=1e-12)
    res = METHODS[name](small, w0, tol=1e-6, max_iter=5000)
    assert res.converged, res.status
    assert res.f - ref.f < 1e-9
    # objective is non-increasing along the iterations
    fs = np.array([r["f"] for r in res.history])
    assert np.all(np.diff(fs) <= 1e-12)


@pytest.mark.parametrize("inner", ["cg", "direct"])
@pytest.mark.parametrize("damping", ["ratio", "linesearch"])
@pytest.mark.parametrize("gd_phase", [True, False])
def test_ash_lr_variants(small, inner, damping, gd_phase):
    res = ash_lr(small, np.zeros(small.d), inner=inner, damping=damping, gd_phase=gd_phase)
    assert res.converged
    assert 1e-12 <= res.info["mu_final"] <= 1e8


def test_ash_lr_switches_to_newton(small):
    res = ash_lr(small, np.zeros(small.d))
    phases = [r.get("phase") for r in res.history[1:]]
    assert "gd" in phases and "newton" in phases
    first_newton = phases.index("newton")
    assert all(p == "newton" for p in phases[first_newton:])  # one-way switch


def test_work_counter_increases(small):
    res = ash_lr(small, np.zeros(small.d))
    assert res.work["n_hvp"] > 0 and res.work["n_hess"] == 0
    res = ash_lr(small, np.zeros(small.d), inner="direct")
    assert res.work["n_hess"] > 0 and res.work["n_hvp"] == 0


def test_deterministic(small):
    a = ash_lr(small, np.zeros(small.d))
    b = ash_lr(small, np.zeros(small.d))
    assert np.array_equal(a.w, b.w) and a.n_iter == b.n_iter


def test_synthetic_condition_number():
    ds, _ = make_synthetic(n=500, d=8, kappa=1e4, seed=3)
    assert ds.meta["kappa_design"] == pytest.approx(1e4, rel=1e-6)


def test_float32_runs(small):
    obj32 = LogisticObjective(small.X, np.where(small.y > 0, 1, -1), lam=1e-3, dtype=np.float32)
    res = ash_lr(obj32, np.zeros(small.d, dtype=np.float32), tol=1e-4)
    assert res.w.dtype == np.float32 and np.isfinite(res.f)
