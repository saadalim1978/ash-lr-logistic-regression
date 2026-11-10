"""Summaries, hypothesis test (H1), LaTeX tables and figures from results/*.csv.

Usage:  python experiments/analyze.py
Writes results/summary_*.csv, results/h1.json, results/tables/*.tex, results/figures/*.pdf|png
"""
from __future__ import annotations

import gzip
import json
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from common import PRIMARY_ILL, RAW, REGIMES, RESULTS  # noqa: E402

TAB = RESULTS / "tables"
FIG = RESULTS / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

ORDER = ["GD", "GD-fixed", "Newton", "Newton-CG", "L-BFGS", "ASH-LR"]
MATRIX_FREE = ["Newton-CG", "L-BFGS"]
# fixed categorical order (validated reference palette) + secondary encoding
STYLE = {
    "ASH-LR":    dict(color="#2a78d6", ls="-",  marker="o"),
    "Newton-CG": dict(color="#eb6834", ls="--", marker="s"),
    "L-BFGS":    dict(color="#1baf7a", ls="-.", marker="^"),
    "Newton":    dict(color="#eda100", ls=":",  marker="D"),
    "GD":        dict(color="#e87ba4", ls="-",  marker="v"),
    "GD-fixed":  dict(color="#008300", ls="--", marker="x"),
}
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3dc", "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8a83", "axes.labelcolor": "#2b2b28", "xtick.color": "#5c5c56",
    "ytick.color": "#5c5c56", "legend.frameon": False, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42,
})


def load(name):
    p = RESULTS / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def hist(name):
    p = RAW / f"{name}_histories.pkl.gz"
    if not p.exists():
        return {}
    with gzip.open(p, "rb") as fh:
        return pickle.load(fh)


def fmt_int(x):
    return "--" if pd.isna(x) else f"{x:,.0f}"


def fmt_ms(x):
    if pd.isna(x):
        return "--"
    return f"{x:,.1f}" if x < 100 else f"{x:,.0f}"


# ---------------------------------------------------------------------------
# E1: main table and H1
# ---------------------------------------------------------------------------
def e1_summary(df):
    g = df.groupby(["regime", "method"])
    s = g.agg(conv=("converged", "sum"), seeds=("seed", "nunique"),
              iters=("iterations", "median"), passes=("passes", "median"),
              time_ms=("time_ms", "median"), gap=("gap", "median"),
              test_acc=("test_acc", "median"), cg=("cg_total", "median"),
              p4=("passes_to_1e-04", "median"), p8=("passes_to_1e-08", "median"),
              t8=("time_to_1e-08", "median")).reset_index()
    s.to_csv(RESULTS / "summary_e1.csv", index=False)

    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Regime & Method & Conv. & Iter. & Passes & Time (ms) & Test acc. \\", r"\midrule"]
    for rid in REGIMES:
        sub = s[s.regime == rid].set_index("method").reindex(ORDER)
        for i, m in enumerate(ORDER):
            r = sub.loc[m]
            name = rf"\textbf{{{m}}}" if m == "ASH-LR" else m
            lines.append(f"{rid if i == 0 else ''} & {name} & {int(r.conv)}/{int(r.seeds)} & "
                         f"{r.iters:.0f} & {fmt_int(r.passes)} & {fmt_ms(r.time_ms)} & {r.test_acc:.3f} \\\\")
        lines.append(r"\midrule" if rid != list(REGIMES)[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (TAB / "e1_main.tex").write_text("\n".join(lines))
    return s


def regimes_table(df):
    first = df.drop_duplicates(["regime", "seed"])
    lines = [r"\begin{tabular}{lllrrr}", r"\toprule",
             r"ID & Data & Description & $n_{\text{train}}$ & $d$ & $\kappa(H_0)$ \\", r"\midrule"]
    for rid, cfg in REGIMES.items():
        sub = first[first.regime == rid]
        data = "synthetic" if cfg["kind"] == "syn" else "Spambase"
        if cfg["kind"] == "syn":
            desc = rf"$\kappa=10^{{{int(np.log10(cfg['kappa']))}}}$"
            if cfg.get("scale_range"):
                desc += r", scaled columns $[10^{-2},10^{2}]$"
        else:
            desc = "standardized" if cfg["standardize"] else "raw features"
        lines.append(f"{rid} & {data} & {desc} & {int(sub.n.iloc[0]):,} & {int(sub.d.iloc[0])} & "
                     f"{sub.hess_cond0.median():.1e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "regimes.tex").write_text("\n".join(lines))


def h1_test(df):
    """Pre-registered H1: >= 20% fewer median passes than the best converging
    matrix-free baseline in >= 2 of the primary ill-conditioned regimes."""
    out = {"regimes": {}}
    wins = 0
    for rid in PRIMARY_ILL:
        sub = df[df.regime == rid]
        conv_all = {m: bool(sub[sub.method == m].converged.all()) for m in MATRIX_FREE}
        cands = [m for m in MATRIX_FREE if conv_all[m]]
        res = {"baselines_converged": conv_all}
        if not cands:
            res["note"] = "no matrix-free baseline converged on all seeds"
            base = None
        else:
            med = {m: sub[sub.method == m].passes.median() for m in cands}
            base = min(med, key=med.get)
        ash = sub[sub.method == "ASH-LR"].set_index("seed")
        res["ash_converged"] = int(ash.converged.sum())
        comparisons = {}
        for m in MATRIX_FREE + ["Newton"]:
            b = sub[sub.method == m].set_index("seed")
            for metric in ("passes", "time_ms"):
                ratio = (ash[metric] / b[metric]).dropna()
                try:
                    p = float(wilcoxon(ash[metric], b[metric]).pvalue)
                except ValueError:
                    p = float("nan")
                comparisons[f"{m}:{metric}"] = {"median_ratio": float(ratio.median()),
                                                "q1": float(ratio.quantile(.25)),
                                                "q3": float(ratio.quantile(.75)),
                                                "wilcoxon_p": p,
                                                "baseline_converged": int(b.converged.sum())}
        res["comparisons"] = comparisons
        if base is not None:
            reduction = 1 - sub[sub.method == "ASH-LR"].passes.median() / \
                sub[sub.method == base].passes.median()
            res.update(best_baseline=base, reduction=float(reduction),
                       success=bool(reduction >= 0.20 and ash.converged.all()))
            wins += res["success"]
        else:
            # no converging baseline: count as success only if ASH-LR converges on all seeds
            res.update(best_baseline=None, reduction=None, success=False)
        out["regimes"][rid] = res
    out["n_success"] = wins
    out["H1_supported"] = wins >= 2
    (RESULTS / "h1.json").write_text(json.dumps(out, indent=2))

    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Regime & Baseline & Conv. (base) & Passes ratio & Time ratio & $p$ (passes) \\",
             r"\midrule"]
    for rid in PRIMARY_ILL:
        c = out["regimes"][rid]["comparisons"]
        for i, m in enumerate(MATRIX_FREE + ["Newton"]):
            a, t = c[f"{m}:passes"], c[f"{m}:time_ms"]
            if a["baseline_converged"] < 10:  # baseline stopped at max_iter: ratio not comparable
                lines.append(f"{rid if i == 0 else ''} & {m} & {a['baseline_converged']}/10 & "
                             r"\multicolumn{3}{c}{baseline not converged} \\")
                continue
            lines.append(f"{rid if i == 0 else ''} & {m} & {a['baseline_converged']}/10 & "
                         f"{a['median_ratio']:.2f} [{a['q1']:.2f}, {a['q3']:.2f}] & "
                         f"{t['median_ratio']:.2f} & {a['wilcoxon_p']:.3f} \\\\")
        lines.append(r"\midrule" if rid != PRIMARY_ILL[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (TAB / "h1_ratios.tex").write_text("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# E2-E5 summaries
# ---------------------------------------------------------------------------
def e2_summary(df):
    s = df.groupby(["regime", "lam", "start", "method"]).agg(
        conv=("converged", "sum"), n=("seed", "nunique"), iters=("iterations", "median"),
        passes=("passes", "median"), finite=("finite", "all")).reset_index()
    s.to_csv(RESULTS / "summary_e2.csv", index=False)
    piv = df.groupby(["regime", "start", "method"]).converged.agg(["sum", "count"]).reset_index()
    meth = ["Newton", "Newton-CG", "L-BFGS", "ASH-LR"]
    lines = [r"\begin{tabular}{ll" + "r" * len(meth) + "}", r"\toprule",
             "Regime & Start & " + " & ".join(meth) + r" \\", r"\midrule"]
    for rid in ("R2", "R5", "R8"):
        for i, st in enumerate(("zero", "near", "far")):
            cells = []
            for m in meth:
                r = piv[(piv.regime == rid) & (piv.start == st) & (piv.method == m)]
                q = s[(s.regime == rid) & (s.start == st) & (s.method == m)].passes.median()
                cells.append(f"{int(r['sum'].iloc[0])}/{int(r['count'].iloc[0])} ({fmt_int(q)})")
            lines.append(f"{rid if i == 0 else ''} & {st} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if rid != "R8" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (TAB / "e2_robustness.tex").write_text("\n".join(lines))
    return s


def e3_summary(df):
    s = df.groupby(["regime", "method"]).agg(
        conv=("converged", "sum"), iters=("iterations", "median"), passes=("passes", "median"),
        time_ms=("time_ms", "median"), rejected=("rejected", "median")).reset_index()
    s.to_csv(RESULTS / "summary_e3.csv", index=False)
    variants = list(dict.fromkeys(df.method))
    regs = ["R2", "R5", "R6", "R8"]
    lines = [r"\begin{tabular}{l" + "rr" * len(regs) + "}", r"\toprule",
             "Variant & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{r}}}" for r in regs) + r" \\",
             " & " + " & ".join(["Conv. & Passes"] * len(regs)) + r" \\", r"\midrule"]
    for v in variants:
        cells = []
        for r in regs:
            row = s[(s.regime == r) & (s.method == v)].iloc[0]
            cells += [f"{int(row.conv)}/10", fmt_int(row.passes)]
        lines.append(f"{v.replace(' mu', r' $\mu$')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "e3_ablation.tex").write_text("\n".join(lines))
    return s


def e4_summary(df):
    s = df.groupby(["regime", "dtype", "method"]).agg(
        conv=("converged", "sum"), n=("seed", "nunique"), iters=("iterations", "median"),
        gap64=("gap64", "median"), time_ms=("time_ms", "median")).reset_index()
    s.to_csv(RESULTS / "summary_e4.csv", index=False)
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Regime & Method & \multicolumn{2}{c}{Conv.} & \multicolumn{2}{c}{Median gap} & Time ratio \\",
             r" & & f64 & f32 & f64 & f32 & f32/f64 \\", r"\midrule"]
    for rid in ("R2", "R7", "R8"):
        for i, m in enumerate(("Newton", "Newton-CG", "ASH-LR")):
            a = s[(s.regime == rid) & (s.method == m) & (s.dtype == "float64")].iloc[0]
            b = s[(s.regime == rid) & (s.method == m) & (s.dtype == "float32")].iloc[0]
            lines.append(f"{rid if i == 0 else ''} & {m} & {int(a.conv)}/{int(a.n)} & {int(b.conv)}/{int(b.n)} & "
                         f"{a.gap64:.1e} & {b.gap64:.1e} & {b.time_ms / a.time_ms:.2f} \\\\")
        lines.append(r"\midrule" if rid != "R8" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (TAB / "e4_float32.tex").write_text("\n".join(lines))
    return s


def e5_summary(df):
    s = df.groupby(["regime", "dtype", "stable", "method"]).agg(
        conv=("converged", "sum"), n=("seed", "nunique"), finite_f0=("f0", lambda x: np.isfinite(x).all()),
        status=("status", lambda x: ",".join(sorted(set(x)))), iters=("iterations", "median")).reset_index()
    s.to_csv(RESULTS / "summary_e5.csv", index=False)
    lines = [r"\begin{tabular}{lllrrl}", r"\toprule",
             r"Regime & Precision & Formulas & Newton & ASH-LR & Failure mode \\", r"\midrule"]
    for rid in ("R2", "R8"):
        first = True
        for dt in ("float64", "float32"):
            for st in (True, False):
                sub = s[(s.regime == rid) & (s.dtype == dt) & (s.stable == st)].set_index("method")
                fails = sorted({x for v in sub.status for x in v.split(",") if x != "converged"})
                lines.append(f"{rid if first else ''} & {dt} & {'stable' if st else 'naive'} & "
                             f"{int(sub.loc['Newton'].conv)}/{int(sub.loc['Newton'].n)} & "
                             f"{int(sub.loc['ASH-LR'].conv)}/{int(sub.loc['ASH-LR'].n)} & "
                             f"{', '.join(fails).replace('non_finite', 'non-finite').replace('_', ' ') or '--'} \\\\")
                first = False
        lines.append(r"\midrule" if rid != "R8" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (TAB / "e5_stability.tex").write_text("\n".join(lines))
    return s


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_convergence(h, df, seed=1000, xkey="passes", fname="convergence_passes"):
    regs = ["R2", "R5", "R6", "R8"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
    for ax, rid in zip(axes, regs):
        f_star = df[(df.regime == rid) & (df.seed == seed)].f_star.iloc[0]
        for m in ORDER:
            rows = h.get((rid, seed, m))
            if not rows:
                continue
            x = np.array([r[xkey] for r in rows]) * (1e3 if xkey == "time" else 1)
            gap = np.maximum(np.array([r["f"] for r in rows]) - f_star, 1e-16) / max(1, abs(f_star))
            st = STYLE[m]
            ax.plot(x, gap, color=st["color"], ls=st["ls"], label=m,
                    lw=2.0 if m == "ASH-LR" else 1.4)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(1e-14, 1)
        desc = re.sub(r"kappa=1e(\d+)", r"$\\kappa=10^{\1}$", REGIMES[rid]["desc"])
        ax.set_title(f"{rid}: {desc}", fontsize=8.5)
        ax.set_xlabel("data passes" if xkey == "passes" else "time (ms)")
    axes[0].set_ylabel(r"relative gap $(F-F^*)/\max(1,|F^*|)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}")
    plt.close(fig)


def fig_ratios(df):
    regs = list(REGIMES)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    pos = np.arange(len(regs))
    for j, (m, off) in enumerate((("Newton-CG", -0.2), ("L-BFGS", 0.0), ("Newton", 0.2))):
        data, where = [], []
        for i, rid in enumerate(regs):
            a = df[(df.regime == rid) & (df.method == "ASH-LR")].set_index("seed")
            b = df[(df.regime == rid) & (df.method == m)].set_index("seed")
            if not b.converged.all():  # ratio against a non-converged run is meaningless
                ax.text(pos[i] + off, 0.42, "n.c.", ha="center", fontsize=6, color="#5c5c56")
                continue
            data.append((a.passes / b.passes).dropna().values)
            where.append(pos[i] + off)
        bp = ax.boxplot(data, positions=where, widths=0.16, patch_artist=True,
                        showfliers=False, medianprops=dict(color="#2b2b28", lw=1.2))
        for b in bp["boxes"]:
            b.set(facecolor=STYLE[m]["color"], alpha=0.85, edgecolor="white", lw=1)
        ax.plot([], [], color=STYLE[m]["color"], lw=6, label=f"vs {m}")
    ax.axhline(1.0, color="#5c5c56", lw=0.8)
    ax.axhline(0.8, color="#5c5c56", lw=0.8, ls=":")
    ax.text(-0.45, 0.78, "20% target", fontsize=7, va="top", ha="left", color="#5c5c56")
    ax.set_xticks(pos, regs)
    ax.set_yscale("log")
    ax.set_ylim(0.38, 400)
    ax.set_ylabel("passes ratio (ASH-LR / baseline)")
    ax.legend(ncol=3, loc="upper left", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"ratios.{ext}")
    plt.close(fig)


def fig_ablation(s3):
    regs = ["R2", "R5", "R6", "R8"]
    variants = list(dict.fromkeys(s3.method))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
    for ax, rid in zip(axes, regs):
        sub = s3[s3.regime == rid].set_index("method").reindex(variants)
        bars = ax.bar(range(len(variants)), sub.passes, color=cols[:len(variants)],
                      edgecolor="white", linewidth=1.5)
        for b, c in zip(bars, sub.conv):
            if c < 10:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(c)}/10",
                        ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(variants)), [v.split(")")[0] + ")" for v in variants])
        ax.set_title(rid, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("median data passes")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"ablation.{ext}")
    plt.close(fig)


def fig_trace(h, rid="R8", seed=1000):
    rows = h.get((rid, seed, "ASH-LR"))
    if not rows:
        return
    k = np.array([r["iter"] for r in rows])
    g = np.array([r["grad_norm"] for r in rows])
    mu = np.array([r.get("mu", np.nan) for r in rows])
    newton = np.array([r.get("phase") == "newton" for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
    axes[0].semilogy(k, g, color=STYLE["ASH-LR"]["color"])
    axes[0].set_ylabel(r"$\|\nabla F(w_k)\|_2$")
    axes[1].semilogy(k[newton], mu[newton], color="#6250d6")
    axes[1].set_ylabel(r"damping $\mu_k$")
    for ax in axes:
        ax.set_xlabel("iteration")
        if newton.any():
            ax.axvline(k[newton][0] - 0.5, color="#5c5c56", lw=0.8, ls=":")
    axes[0].text(k[newton][0], g.max(), " switch", fontsize=7, va="top", color="#5c5c56")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"trace_{rid}.{ext}")
    plt.close(fig)


def main():
    out = {}
    df1 = load("e1_main")
    if df1 is not None:
        e1_summary(df1)
        regimes_table(df1)
        out["h1"] = h1_test(df1)
        h = hist("e1_main")
        fig_convergence(h, df1)
        fig_convergence(h, df1, xkey="time", fname="convergence_time")
        fig_ratios(df1)
        fig_trace(h)
    for name, fn in (("e2_robustness", e2_summary), ("e4_float32", e4_summary),
                     ("e5_stability", e5_summary)):
        d = load(name)
        if d is not None:
            fn(d)
    d3 = load("e3_ablation")
    if d3 is not None:
        fig_ablation(e3_summary(d3))
    if "h1" in out:
        print(json.dumps({k: (v["reduction"], v["best_baseline"], v["success"])
                          for k, v in out["h1"]["regimes"].items()}, indent=1))
        print("H1 supported:", out["h1"]["H1_supported"])


if __name__ == "__main__":
    main()
