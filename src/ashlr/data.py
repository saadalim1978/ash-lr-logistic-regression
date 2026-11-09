"""Data sets: synthetic designs with a target condition number, and Spambase.

Synthetic design
----------------
X = Z diag(sqrt(e)) V^T, where Z has orthonormal columns scaled so that
Z^T Z / n = I, V is a random orthogonal matrix and the eigenvalues e are
log-spaced between 1 and 1/kappa. Hence X^T X / n = V diag(e) V^T has
condition number exactly kappa. Using the same seed for Z and V across kappa
gives *matched* designs that differ only in their spectrum.

Labels: a random true weight vector is rescaled so that std(X w*) equals
``logit_std``; then y_i = +1 with probability sigma(x_i^T w*), else -1.

Optional ``scale_range`` multiplies column j by a log-spaced factor, which
creates differently scaled features (bad scaling on top of correlation).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .objectives import sigmoid_stable

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DEFAULT_SEED = 448540005


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.X_train.shape


# ---------------------------------------------------------------------------
def stratified_split(y, test_frac=0.25, seed=DEFAULT_SEED):
    """Fixed stratified split; returns (train_idx, test_idx), both sorted."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_test = int(round(test_frac * idx.size))
        test.append(idx[:n_test])
        train.append(idx[n_test:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def random_orthogonal(d, rng):
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))  # Haar-distributed


def make_synthetic(
    n=1000,
    d=10,
    kappa=10.0,
    logit_std=1.5,
    seed=DEFAULT_SEED,
    scale_range=None,
    test_frac=0.25,
    name=None,
):
    """Synthetic logistic data whose design X^T X / n has condition number kappa."""
    rng = np.random.default_rng(seed)
    Z, _ = np.linalg.qr(rng.standard_normal((n, d)))
    Z *= np.sqrt(n)                                   # Z^T Z / n = I
    V = random_orthogonal(d, rng)
    eig = np.logspace(0.0, -np.log10(kappa), d)        # 1 ... 1/kappa
    X = (Z * np.sqrt(eig)) @ V.T
    if scale_range is not None:                        # differently scaled columns
        lo, hi = scale_range
        X = X * np.logspace(np.log10(lo), np.log10(hi), d)[rng.permutation(d)]
    w_star = rng.standard_normal(d)
    w_star *= logit_std / np.std(X @ w_star)
    prob = sigmoid_stable(X @ w_star)
    y = np.where(rng.random(n) < prob, 1, -1)
    tr, te = stratified_split(y, test_frac, seed)
    cond = np.linalg.cond(X.T @ X / n)
    meta = {
        "n": n, "d": d, "kappa_target": kappa, "kappa_design": float(cond),
        "logit_std": logit_std, "seed": seed, "scale_range": scale_range,
        "pos_frac": float(np.mean(y == 1)),
    }
    name = name or f"synthetic_d{d}_k{kappa:g}"
    return Dataset(name, X[tr], y[tr], X[te], y[te], meta), w_star


def save_synthetic_csv(ds: Dataset, path: Path):
    """Save a synthetic set (train+test with a split column) and metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f"x{j}" for j in range(ds.X_train.shape[1])]
    df = pd.concat(
        [
            pd.DataFrame(ds.X_train, columns=cols).assign(y=ds.y_train, split="train"),
            pd.DataFrame(ds.X_test, columns=cols).assign(y=ds.y_test, split="test"),
        ]
    )
    df.to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(ds.meta, indent=2))


# ---------------------------------------------------------------------------
def load_spambase(
    path=None,
    standardize=True,
    drop_duplicates=False,
    seed=DEFAULT_SEED,
    test_frac=0.25,
):
    """Load Spambase (UCI, CC BY 4.0) and return a stratified 75/25 split.

    The CSV must contain the 57 feature columns and a ``target`` column with
    labels in {0,1} or {-1,+1}. ``standardize`` uses training statistics only.
    ``drop_duplicates`` removes exact duplicate rows (features + label) before
    splitting, so identical e-mails cannot appear in both train and test.
    """
    path = Path(path) if path else DATA_DIR / "raw" / "spambase.csv"
    df = pd.read_csv(path)
    n_raw = len(df)
    n_dup = int(df.duplicated().sum())
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    y = df["target"].to_numpy()
    y = np.where(y > 0, 1, -1)
    X = df.drop(columns="target").to_numpy(dtype=np.float64)
    tr, te = stratified_split(y, test_frac, seed)
    Xtr, Xte = X[tr], X[te]
    mean = std = None
    if standardize:
        mean = Xtr.mean(axis=0)
        std = Xtr.std(axis=0)
        std[std == 0] = 1.0
        Xtr = (Xtr - mean) / std
        Xte = (Xte - mean) / std
    meta = {
        "n_raw": n_raw, "n_duplicates": n_dup, "dropped_duplicates": drop_duplicates,
        "n_used": len(df), "d": X.shape[1], "standardized": standardize,
        "pos_frac": float(np.mean(y == 1)), "seed": seed,
        "kappa_design": float(np.linalg.cond(Xtr.T @ Xtr / Xtr.shape[0])),
    }
    tag = ("std" if standardize else "raw") + ("_dedup" if drop_duplicates else "")
    return Dataset(f"spambase_{tag}", Xtr, y[tr], Xte, y[te], meta)
