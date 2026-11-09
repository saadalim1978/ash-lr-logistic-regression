"""L2-regularized logistic regression objective.

    F(w) = (1/n) * sum_i log(1 + exp(-y_i x_i^T w)) + (lam/2) * ||w||^2,   y_i in {-1,+1}

The class provides the value, gradient, Hessian, and Hessian-vector products
(Hv), and it counts the computational work spent on each.

Numerical stability
-------------------
``stable=True`` (default) evaluates the loss with the softplus identity
``log(1+exp(-z)) = logaddexp(0, -z)`` and the sigmoid with sign-dependent
branches, so no exponential of a large positive number is ever formed.
``stable=False`` uses the textbook formulas ``log(1+exp(-z))`` and
``1/(1+exp(-z))``; it exists only for the ablation and float32 experiments.

Work accounting
---------------
Work is measured in *data passes*: one pass equals one product with X or X^T
(about n*d multiply-adds). Typical costs:

  * value F(w)            : 1 pass  (margins X w)
  * gradient (margins hit): 1 pass  (X^T r)
  * Hessian-vector product: 2 passes (X v, then X^T(D X v))
  * dense Hessian X^T D X : d passes
  * dense solve (Cholesky/LU, ~d^3/3 flops): d^2 / (3 n) passes
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Elementwise helpers
# ---------------------------------------------------------------------------
def sigmoid_stable(t: np.ndarray) -> np.ndarray:
    """sigma(t) = 1/(1+exp(-t)) without overflow (sign-dependent branches)."""
    out = np.empty_like(t)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    e = np.exp(t[~pos])  # t < 0, so exp(t) <= 1 and cannot overflow
    out[~pos] = e / (1.0 + e)
    return out


def sigmoid_naive(t: np.ndarray) -> np.ndarray:
    """Textbook sigmoid; exp(-t) overflows for very negative t."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-t))


def softplus_neg_stable(z: np.ndarray) -> np.ndarray:
    """log(1 + exp(-z)) evaluated as logaddexp(0, -z)."""
    return np.logaddexp(0.0, -z)


def softplus_neg_naive(z: np.ndarray) -> np.ndarray:
    """Textbook log(1 + exp(-z)); returns inf once exp(-z) overflows."""
    with np.errstate(over="ignore"):
        return np.log(1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Work counter
# ---------------------------------------------------------------------------
@dataclass
class WorkCounter:
    """Counts oracle calls and converts them to data passes."""

    n: int
    d: int
    fun: int = 0
    grad: int = 0
    hvp: int = 0
    hess: int = 0
    solve: int = 0
    extra_passes: float = 0.0
    _hist: list = field(default_factory=list, repr=False)

    @property
    def passes(self) -> float:
        return (
            self.fun
            + self.grad
            + 2 * self.hvp
            + self.d * self.hess
            + self.solve * self.d**2 / (3.0 * self.n)
            + self.extra_passes
        )

    def as_dict(self) -> dict:
        return {
            "n_fun": self.fun,
            "n_grad": self.grad,
            "n_hvp": self.hvp,
            "n_hess": self.hess,
            "n_solve": self.solve,
            "passes": self.passes,
        }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
class LogisticObjective:
    """Binary L2-regularized logistic loss with cached margins.

    Parameters
    ----------
    X : (n, d) array
    y : (n,) array with entries in {-1, +1}
    lam : regularization parameter (> 0)
    stable : use overflow-safe formulas (see module docstring)
    dtype : floating type used for all computations (float64 or float32)
    """

    def __init__(self, X, y, lam: float = 1e-6, stable: bool = True, dtype=np.float64):
        y = np.asarray(y)
        if not np.all(np.isin(y, (-1, 1))):
            raise ValueError("labels must be in {-1, +1}")
        if lam <= 0:
            raise ValueError("lam must be positive")
        self.dtype = np.dtype(dtype)
        self.X = np.ascontiguousarray(X, dtype=self.dtype)
        self.y = y.astype(self.dtype)
        self.n, self.d = self.X.shape
        self.lam = self.dtype.type(lam)
        self.stable = stable
        self._sig = sigmoid_stable if stable else sigmoid_naive
        self._sp = softplus_neg_stable if stable else softplus_neg_naive
        self.work = WorkCounter(self.n, self.d)
        self._w_cache = None
        self._z_cache = None

    # -- internal ---------------------------------------------------------
    def _margins(self, w):
        """z = y * (X w); cached so value and gradient share one pass."""
        if self._w_cache is not None and np.array_equal(w, self._w_cache):
            return self._z_cache
        z = self.y * (self.X @ w)
        self.work.fun += 1  # counted as the "value" pass
        self._w_cache = w.copy()
        self._z_cache = z
        return z

    # -- oracles ----------------------------------------------------------
    def value(self, w):
        w = np.asarray(w, dtype=self.dtype)
        z = self._margins(w)
        return float(np.mean(self._sp(z)) + 0.5 * self.lam * (w @ w))

    def grad(self, w):
        w = np.asarray(w, dtype=self.dtype)
        z = self._margins(w)
        r = -self.y * self._sig(-z)  # d/dz_i of loss times y_i
        self.work.grad += 1
        return self.X.T @ r / self.n + self.lam * w

    def value_grad(self, w):
        return self.value(w), self.grad(w)

    def curvature_weights(self, w):
        """D_ii = sigma(z_i)(1 - sigma(z_i)), the Hessian weights."""
        z = self._margins(w)
        s = self._sig(z)
        return s * (1.0 - s)

    def hessian(self, w):
        """Dense Hessian X^T D X / n + lam I (costs d passes)."""
        w = np.asarray(w, dtype=self.dtype)
        D = self.curvature_weights(w)
        self.work.hess += 1
        H = (self.X.T * D) @ self.X / self.n
        H[np.diag_indices_from(H)] += self.lam
        return H

    def hess_vec(self, w, v, D=None):
        """Hessian-vector product without forming H (2 passes)."""
        if D is None:
            D = self.curvature_weights(np.asarray(w, dtype=self.dtype))
        self.work.hvp += 1
        return self.X.T @ (D * (self.X @ v)) / self.n + self.lam * v

    def count_solve(self):
        self.work.solve += 1

    def reset_work(self):
        self.work = WorkCounter(self.n, self.d)
        self._w_cache = None
        self._z_cache = None

    # -- diagnostics ------------------------------------------------------
    def accuracy(self, w, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        pred = np.where(X @ w >= 0, 1.0, -1.0)
        return float(np.mean(pred == y))

    def hessian_condition(self, w):
        """Condition number of the Hessian (diagnostic, not counted as work)."""
        saved = self.work.hess
        H = self.hessian(w)
        self.work.hess = saved
        ev = np.linalg.eigvalsh(H.astype(np.float64))
        return float(ev[-1] / ev[0])

    def lipschitz(self):
        """Upper bound L = ||X||_2^2 / (4n) + lam on the gradient Lipschitz constant."""
        s = np.linalg.norm(self.X.astype(np.float64), 2)
        return float(s**2 / (4.0 * self.n) + self.lam)
