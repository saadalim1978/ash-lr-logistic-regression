# Pre-registered Experiment Plan (final study)

Fixed on 2026-11-09, **before** the final experiments were run, and committed to
the repository so that later changes are visible in the history.

## 0. Disclosure of exploratory runs

Before this plan was written, exploratory runs were made with the default seed
(448540005) on: the Milestone 2 data sets (synthetic d = 10, kappa = 10 and 1e3;
Spambase standardized), synthetic d in {10, 200, 500} with kappa in {1e2, 1e4, 1e6},
and Spambase without standardization. These runs motivated two changes to the
method (Section 1). They are not part of the reported results; the final study
uses the fresh seeds listed in Section 3.

## 1. Final method (frozen)

`ash_lr(obj, w0)` with its default options:

| Component | Setting |
|---|---|
| Phase 1 | gradient steps, Armijo (c = 1e-4, alpha0 = 1, halving, <= 40 trials) |
| Switch (one-way) | ‖g‖ <= 0.1 ‖g0‖, or >= 5 GD steps and relative decrease <= 1e-3 |
| Phase 2 direction | (H + mu I) p = -g by inexact CG with Hessian-vector products |
| CG forcing term | ‖r‖ <= min(0.5, sqrt(‖g‖)) ‖g‖, at most d CG iterations |
| Damping rule | mu = sqrt(theta ‖g‖), theta set so mu = 1e-3 at the switch |
| Acceptance | ratio test rho = actual / predicted decrease; accept if rho >= 1e-4 |
| theta update | x4 if rho < 0.25, x0.5 if rho > 0.75; mu clipped to [1e-12, 1e8] |
| Safeguard | after 30 rejections, Armijo along the last direction |

Changes relative to Milestone 2, and why:
1. Direct solve -> inexact Newton-CG and line-search damping -> ratio test
   (both were already planned in Milestone 2).
2. mu adapted directly -> gradient-regularized mu = sqrt(theta ‖g‖)
   (motivated by [Doikov, Mishchenko & Nesterov 2024] and by the exploratory
   observation that a slowly halving mu keeps steps over-damped).

## 2. Research questions and hypotheses

- **Q1 (efficiency).** Does ASH-LR reduce the work to reach the tolerance
  compared with the matrix-free baselines (Newton-CG, L-BFGS)?
  *Success criterion (H1):* in at least two of the five primary ill-conditioned
  regimes R3-R6, R8, the median number of data passes of ASH-LR to reach
  ‖g‖ <= 1e-6 is at least 20 % lower than that of the best matrix-free
  baseline that converges in that regime.
- **Q2 (robustness).** Does ASH-LR converge from distant starting points and
  across lambda without non-finite values?
- **Q3 (components).** What do the stable formulas, the gradient phase, the
  inner solver and the damping rule each contribute?
- Direct Newton is reported in every table as a reference. It is expected to be
  fastest in wall time for d <= 500 with dense data; this is reported, not hidden.

## 3. Regimes and data

| ID | Data | n | d | kappa(XᵀX/n) | Notes |
|---|---|---|---|---|---|
| R1 | synthetic | 1000 | 10 | 1e1 | well conditioned (sanity) |
| R2 | synthetic | 1000 | 10 | 1e3 | Milestone 2 ill-conditioned |
| R3 | synthetic | 1000 | 10 | 1e6 | severe ill-conditioning |
| R4 | synthetic | 4000 | 200 | 1e4 | moderate dimension |
| R5 | synthetic | 6000 | 500 | 1e6 | larger dimension, severe |
| R6 | synthetic, scaled columns | 2000 | 50 | 1e2 | column scales log-spaced in [1e-2, 1e2] |
| R7 | Spambase, standardized | 4601 | 57 | — | real data |
| R8 | Spambase, raw (unscaled) | 4601 | 57 | — | real data, bad scaling |

Synthetic: logit std 1.5, stratified 75/25 split. Spambase: stratified 75/25
split, standardization with training statistics only (R7).
Seeds: 10 seeds, `1000 + s` for s = 0..9 (data generation for R1-R6; split for R7-R8).
A duplicate-free Spambase variant is evaluated only for test accuracy.

## 4. Experiments

| Exp. | Regimes | Factors | Methods |
|---|---|---|---|
| E1 main | R1-R8 | 10 seeds, w0 = 0, lambda = 1e-6 | GD, GD-fixed, Newton, Newton-CG, L-BFGS, ASH-LR |
| E2 robustness | R2, R5, R8 | 3 seeds x w0 in {0, random near, random far} x lambda in {1e-6, 1e-4, 1e-2} | Newton, Newton-CG, L-BFGS, ASH-LR |
| E3 ablation | R2, R5, R6, R8 | 10 seeds, w0 = 0 | see below |
| E4 float32 | R2, R7, R8 | 3 seeds, tol 1e-4 | Newton, Newton-CG, ASH-LR (float32 vs float64) |
| E5 stability | R2, R8 | far start (logit std 1e3) | ASH-LR stable vs naive formulas |

Random starts: w0 ~ N(0, I) rescaled so that std(X w0) = 5 (near) or 50 (far).
For E2 the relative stopping rule ‖g‖ / max(1, ‖g0‖) <= 1e-6 is used.

E3 variants: (a) Milestone 2 version: direct solve, line-search damping, fixed mu;
(b) CG + line-search damping, fixed mu; (c) CG + ratio test, fixed mu;
(d) full ASH-LR; (e) full ASH-LR without gradient phase; (f) direct solve +
ratio test + gradient mu.

## 5. Metrics and protocol

- Stop: ‖g‖ <= 1e-6, 1000 iterations, or line-search failure.
- Reported: iterations, data passes, wall time, converged flag, objective gap to a
  reference (Newton, tol 1e-10), test accuracy, CG iterations, rejected steps.
- Time-to-accuracy: passes and time until the relative objective gap
  (F - F*) / max(1, |F*|) first falls below 1e-4 and 1e-8.
- Timing: one untimed warm-up run, then 3 timed repetitions in randomized method
  order; median reported. All arithmetic float64 except E4. Single machine.
- Statistics: per regime, median over seeds of the ratio
  passes(ASH-LR) / passes(baseline), and a two-sided Wilcoxon signed-rank test
  over the 10 seeds (alpha = 0.05).
