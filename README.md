# ASH-LR: Adaptive and Numerically Stable Optimization for Ill-Conditioned Logistic Regression

Course project, College of Computer Engineering and Science, Prince Sattam bin Abdulaziz University
Student: Saad Almusfir (448540005) · Instructor: Dr. Mahdi Khemakhem

**Repository:** https://github.com/saadalim1978/ash-lr-logistic-regression

ASH-LR (*Adaptive Stable Hybrid Logistic Regression*) minimizes the L2-regularized
logistic loss

$$F(w)=\frac1n\sum_{i=1}^n \log\bigl(1+e^{-y_i x_i^\top w}\bigr)+\frac\lambda2\|w\|_2^2 ,\qquad y_i\in\{-1,+1\},$$

with three ingredients:

1. **Stable evaluation** – the loss via `logaddexp`, the sigmoid with sign-dependent branches.
2. **Hybrid switching** – Armijo gradient steps first; a one-way switch to Newton once
   ‖g‖ ≤ 0.1‖g₀‖ or progress stalls.
3. **Damped Newton–CG** – (H + μI)p = −g solved inexactly with conjugate gradients
   using Hessian-vector products only, gradient-regularized damping
   μ = √(θ‖g‖), and a ratio test (actual vs. predicted decrease) that adapts θ.

The repository contains the implementation, the baselines (GD, GD with fixed step,
Newton, Newton–CG, L-BFGS), a test suite and the scripts that produce every table
and figure of the report.

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.24, pandas ≥ 1.5, scipy ≥ 1.9, matplotlib ≥ 3.6, pytest ≥ 7 (tests only)

## Installation

```bash
git clone https://github.com/saadalim1978/ash-lr-logistic-regression.git
cd ash-lr-logistic-regression
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                    # optional: makes `import ashlr` work everywhere
```

## Quick start

```bash
python examples/quickstart.py
```

```python
import numpy as np
from ashlr import LogisticObjective, ash_lr, make_synthetic

ds, _ = make_synthetic(n=1000, d=10, kappa=1e4, seed=0)
obj = LogisticObjective(ds.X_train, ds.y_train, lam=1e-6)
res = ash_lr(obj, np.zeros(10), tol=1e-6)
print(res.converged, res.n_iter, res.f, res.work["passes"])
```

Every optimizer has the signature `method(obj, w0, tol=1e-6, max_iter=1000, relative=False, **options)`
and returns an `OptResult` with the weights, loss, gradient norm, status, wall time,
work counters (`passes` = products with X or Xᵀ) and a per-iteration `history`.

ASH-LR options (all keyword arguments of `ash_lr`):

| Option | Default | Meaning |
|---|---|---|
| `inner` | `"cg"` | `"cg"` (Hessian-vector products) or `"direct"` (dense solve) |
| `damping` | `"ratio"` | `"ratio"` (actual/predicted decrease) or `"linesearch"` (Milestone 2) |
| `mu_rule` | `"gradient"` | `"gradient"`: μ = √(θ‖g‖); `"fixed"`: μ adapted directly |
| `gd_phase` | `True` | start with gradient steps (False: damped Newton from the first iteration) |
| `switch_grad_frac`, `switch_min_steps`, `switch_rel_decrease` | 0.1, 5, 1e-3 | switching rule |
| `mu0`, `mu_min`, `mu_max` | 1e-3, 1e-12, 1e8 | damping start value and bounds |

## Tests

```bash
pytest -q
```

The 26 tests check the gradient and Hessian against centered finite differences,
Hessian-vector products against the dense Hessian, the CG solver, finite outputs
at extreme logits (and the overflow of the naive formulas), convergence and
monotone decrease of every method, all ASH-LR variants, the one-way switch, the
work counters, determinism, the synthetic condition number and float32 runs.

## Reproducing the experiments

The study follows the pre-registered plan in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)
(regimes R1–R8, experiments E1–E5, success criterion H1).

```bash
cd experiments
python run_experiments.py all --quick     # smoke test: 2 seeds, 1 timed repetition (~3 min)
python run_experiments.py e1 e2 e3 e4 e5  # full study (~1 h on a laptop)
python analyze.py                         # tables (results/tables), figures (results/figures), H1 test
```

| Experiment | Content | Output |
|---|---|---|
| E1 | 6 methods × 8 regimes × 10 seeds | `results/e1_main.csv` |
| E2 | starting points × λ (R2, R5, R8) | `results/e2_robustness.csv` |
| E3 | ablation of ASH-LR components | `results/e3_ablation.csv` |
| E4 | float32 vs float64 | `results/e4_float32.csv` |
| E5 | stable vs naive formulas from an extreme start | `results/e5_stability.csv` |

## Repository layout

```
ash-lr-logistic-regression/
├── src/ashlr/
│   ├── objectives.py     # stable loss, gradient, Hessian, Hessian-vector products, work counter
│   ├── optimizers.py     # Armijo, CG, GD, GD-fixed, Newton, Newton-CG, L-BFGS, ASH-LR
│   └── data.py           # synthetic generator (exact condition number), Spambase loader
├── experiments/
│   ├── common.py         # regimes, timing protocol, time-to-accuracy
│   ├── run_experiments.py
│   └── analyze.py        # summaries, H1 test, LaTeX tables, figures
├── tests/test_ashlr.py
├── examples/quickstart.py
├── data/raw/spambase.csv # UCI Spambase, CC BY 4.0
├── results/              # CSV results, tables, figures
├── EXPERIMENT_PLAN.md
└── README.md
```

## Results

All numbers below come from `results/` and are reproduced by the commands above.

**Main comparison (E1, median over 10 seeds, `‖g‖ ≤ 1e-6`)** – data passes / converged seeds:

| Regime | Newton | Newton-CG | L-BFGS | ASH-LR |
|---|---|---|---|---|
| R2 synthetic κ=1e3 | 50 (10/10) | 92 (10/10) | 128 (10/10) | 140 (10/10) |
| R3 synthetic κ=1e6 | 50 (10/10) | 126 (10/10) | 416 (10/10) | 266 (10/10) |
| R5 synthetic d=500, κ=1e6 | 2,605 (10/10) | 3,082 (10/10) | – (0/10) | 6,990 (10/10) |
| R6 synthetic, scaled columns | 265 (10/10) | 46,162 (10/10) | – (0/10) | 54,120 (10/10) |
| R8 Spambase raw | 773 (10/10) | 2,647 (10/10) | – (0/10) | 3,122 (10/10) |

Key findings:

- ASH-LR, Newton and Newton-CG converged on every problem; GD failed on all
  ill-conditioned regimes and L-BFGS on the larger and badly scaled ones.
- The pre-registered hypothesis H1 (≥ 20 % fewer passes than the best matrix-free
  baseline in ≥ 2 of 5 ill-conditioned regimes) is **not supported**: Newton-CG was more
  efficient than ASH-LR in four regimes and on a par in R6 (`results/h1.json`).
- Direct Newton is the fastest method in wall time for all dimensions studied (d ≤ 500).
- Stable formulas are essential: from an extreme start, all ASH-LR runs with the naive
  formulas failed (non-finite loss), all runs with the stable formulas converged in float64.
- Ablation: the gradient phase is the main overhead; on badly scaled data the
  unpreconditioned CG solve, not the damping, limits progress.


## Data

Spambase: M. Hopkins, E. Reeber, G. Forman, J. Suermondt. UCI Machine Learning
Repository, 1999. doi:10.24432/C53G6X. License CC BY 4.0. The file in
`data/raw/spambase.csv` is the original data with a header row; labels are mapped
to {−1, +1} when loaded. The 391 exact duplicate rows are kept for optimization
(they are part of the objective) and can be removed with
`load_spambase(drop_duplicates=True)`.

## License

Code: MIT. Data: CC BY 4.0 (see above).
