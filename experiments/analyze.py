"""Turn results/raw_results.csv into tables, statistical tests and plots.

    python experiments/analyze.py
Outputs (in results/): summary_table.csv, statistical_tests.csv, pruning.csv,
approximation.csv, summary.md and fig_*.png
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ALGS = ["Greedy", "DSATUR", "Naive BT", "BT+FC+MRV", "B&B", "Hybrid B&B+LS"]
EXACT = ["Naive BT", "BT+FC+MRV", "B&B", "Hybrid B&B+LS"]
COLORS = dict(zip(ALGS, ["#8c8c8c", "#c49c00", "#d62728", "#1f77b4", "#2ca02c", "#9467bd"]))
MARK = dict(zip(ALGS, ["s", "D", "x", "o", "^", "P"]))
FAMILY_TITLE = {"random": "Random (G(n,p)-like)", "uz": "UZ-inspired (synthetic)"}


def ci95(x: pd.Series) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(np.mean(x)), float("nan")
    h = stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))
    return float(np.mean(x)), float(h)


def load() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RES, "raw_results.csv"))
    df["inst"] = df["family"] + "|" + df["n"].astype(str) + "|" + df["seed"].astype(str)
    # reference value per instance: proved optimum if any solver proved one,
    # otherwise the best timetable found by any solver (upper bound on OPT)
    proved = df[df.status == "optimal"].groupby("inst")["slots"].min()
    best = df.groupby("inst")["slots"].min()
    df["ref"] = df["inst"].map(best)
    df["opt_known"] = df["inst"].map(proved).notna()
    df["ref"] = np.where(df["opt_known"], df["inst"].map(proved), df["ref"])
    df["gap_lb"] = df["slots"] - df["lb"]
    df["gap_ref"] = df["slots"] - df["ref"]
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, n, alg), g in df.groupby(["family", "n", "algorithm"], sort=False):
        m_nodes, h_nodes = ci95(g["nodes"])
        m_t, h_t = ci95(g["seconds"])
        m_s, h_s = ci95(g["slots"])
        rows.append(dict(
            family=fam, n=n, algorithm=alg, instances=len(g),
            slots_mean=m_s, slots_ci=h_s, gap_to_best=g["gap_ref"].mean(),
            gap_to_lb=g["gap_lb"].mean(),
            proved_optimal_pct=100 * (g["status"] == "optimal").mean(),
            timed_out_pct=100 * g["timed_out"].mean(),
            nodes_mean=m_nodes, nodes_ci=h_nodes, time_mean=m_t, time_ci=h_t,
            conflicts=int(g["conflicts"].sum() + g["overloads"].sum()),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "summary_table.csv"), index=False)
    return out


# ------------------------------------------------------------------- pruning
def pruning(df: pd.DataFrame) -> pd.DataFrame:
    """% reduction in nodes explored relative to Naive BT, per instance, over
    the instances where a search was needed (DSATUR was not already optimal).
    Naive BT is censored at the time limit on large instances, so there the
    reduction is a LOWER bound on the true reduction."""
    piv = df.pivot_table(index=["family", "n", "seed"], columns="algorithm", values="nodes")
    done = df.pivot_table(index=["family", "n", "seed"], columns="algorithm", values="timed_out")
    rows = []
    for (fam, n, seed), r in piv.iterrows():
        if r["Naive BT"] <= 0:
            continue
        for alg in ["BT+FC+MRV", "B&B", "Hybrid B&B+LS"]:
            rows.append(dict(family=fam, n=n, seed=seed, algorithm=alg,
                             reduction_pct=100 * (1 - r[alg] / r["Naive BT"]),
                             naive_censored=bool(done.loc[(fam, n, seed), "Naive BT"]),
                             alg_censored=bool(done.loc[(fam, n, seed), alg])))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "pruning.csv"), index=False)
    return out


# ------------------------------------------------------------ approximation
def approximation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alg in ["Greedy", "DSATUR"]:
        g = df[df.algorithm == alg].copy()
        g["ratio_best"] = g["slots"] / g["ref"]
        g["ratio_lb"] = g["slots"] / g["lb"]
        for (fam, n), h in g.groupby(["family", "n"]):
            rows.append(dict(algorithm=alg, family=fam, n=n,
                             mean_ratio_vs_best=h["ratio_best"].mean(),
                             max_ratio_vs_best=h["ratio_best"].max(),
                             mean_ratio_vs_lb=h["ratio_lb"].mean(),
                             max_ratio_vs_lb=h["ratio_lb"].max()))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "approximation.csv"), index=False)
    return out


# ------------------------------------------------------------------- tests
def paired_test(df, fam, alg_a, alg_b, metric, min_n, label, log=False):
    sub = df[(df.family == fam) & (df.n >= min_n) & df.algorithm.isin([alg_a, alg_b])]
    piv = sub.pivot_table(index="inst", columns="algorithm", values=metric)
    if log:
        piv = np.log10(piv + 1)
    d = (piv[alg_a] - piv[alg_b]).dropna()
    if len(d) < 3:
        return None
    if (d == 0).all():
        stat, p = 0.0, 1.0
    else:
        stat, p = stats.wilcoxon(d, zero_method="wilcox")
    return dict(family=fam, comparison=f"{alg_a} vs {alg_b}", metric=label, n_min=min_n,
                pairs=len(d), median_diff=float(np.median(d)),
                a_better_pct=100 * float((d < 0).mean()), b_better_pct=100 * float((d > 0).mean()),
                tie_pct=100 * float((d == 0).mean()), wilcoxon_W=float(stat), p_value=float(p))


def tests(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for fam in df.family.unique():
        specs = [
            ("DSATUR", "Greedy", "slots", 50, "slots", False),
            ("Hybrid B&B+LS", "DSATUR", "slots", 50, "slots", False),
            ("Hybrid B&B+LS", "B&B", "slots", 100, "slots", False),
            ("BT+FC+MRV", "Naive BT", "nodes", 20, "log10(nodes+1)", True),
            ("B&B", "BT+FC+MRV", "nodes", 30, "log10(nodes+1)", True),
            ("Hybrid B&B+LS", "B&B", "nodes", 100, "log10(nodes+1)", True),
        ]
        for a, b, metric, min_n, label, log in specs:
            r = paired_test(df, fam, a, b, metric, min_n, label, log)
            if r:
                out.append(r)
    t = pd.DataFrame(out)
    if len(t):
        t["p_bonferroni"] = np.minimum(1.0, t["p_value"] * len(t))
        t["significant_5pct"] = t["p_bonferroni"] < 0.05
    t.to_csv(os.path.join(RES, "statistical_tests.csv"), index=False)
    return t


# --------------------------------------------------------------------- plots
def panels(fig_name, title, ylabel, fn, families, logy=False, logx=True, sharey=False):
    fig, axes = plt.subplots(1, len(families), figsize=(6.2 * len(families), 4.4), sharey=sharey)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, families):
        fn(ax, fam)
        ax.set_title(FAMILY_TITLE.get(fam, fam), fontsize=10)
        ax.set_xlabel("number of exams n")
        ax.set_ylabel(ylabel)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=.3, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, fig_name), dpi=150)
    plt.close(fig)


def make_plots(df, summ, prune, approx):
    fams = list(df.family.unique())

    def line(metric, ci, algs, floor=None):
        def f(ax, fam):
            for alg in algs:
                s = summ[(summ.family == fam) & (summ.algorithm == alg)].sort_values("n")
                y = s[metric].clip(lower=floor) if floor else s[metric]
                ax.errorbar(s["n"], y, yerr=s[ci] if ci else None, label=alg, color=COLORS[alg],
                            marker=MARK[alg], ms=5, lw=1.4, capsize=2)
        return f

    panels("fig_nodes.png", "Search nodes explored (mean, 95% CI; capped by the 5 s limit)",
           "nodes explored", line("nodes_mean", "nodes_ci", EXACT, floor=1), fams, logy=True)
    panels("fig_time.png", "Running time (mean, 95% CI)", "seconds",
           line("time_mean", "time_ci", ALGS, floor=1e-5), fams, logy=True)
    panels("fig_quality.png", "Solution quality: slots above the lower bound (mean)",
           "slots - lower bound", line("gap_to_lb", None, ALGS), fams)

    def solved(ax, fam):
        for alg in EXACT:
            s = summ[(summ.family == fam) & (summ.algorithm == alg)].sort_values("n")
            ax.plot(s["n"], s["proved_optimal_pct"], label=alg, color=COLORS[alg],
                    marker=MARK[alg], ms=5)
        ax.set_ylim(-3, 103)
    panels("fig_solved.png", "Instances solved to proven optimality within the time limit",
           "% proven optimal", solved, fams)

    def ratio(ax, fam):
        for alg in ["Greedy", "DSATUR"]:
            s = approx[(approx.family == fam) & (approx.algorithm == alg)].sort_values("n")
            ax.plot(s["n"], s["mean_ratio_vs_best"], color=COLORS[alg], marker=MARK[alg],
                    label=f"{alg} mean")
            ax.plot(s["n"], s["max_ratio_vs_best"], color=COLORS[alg], ls="--", alpha=.6,
                    label=f"{alg} max")
        ax.axhline(1.0, color="k", lw=.6)
    panels("fig_ratio.png", "Empirical approximation ratio (slots / best known)",
           "ratio", ratio, fams)

    # pruning effectiveness
    fig, axes = plt.subplots(1, len(fams), figsize=(6.2 * len(fams), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        p = prune[prune.family == fam].groupby(["n", "algorithm"])["reduction_pct"].mean().unstack()
        p = p[[c for c in ["BT+FC+MRV", "B&B", "Hybrid B&B+LS"] if c in p]]
        p.plot.bar(ax=ax, color=[COLORS[c] for c in p.columns], width=.8)
        ax.set_title(FAMILY_TITLE.get(fam, fam), fontsize=10)
        ax.set_xlabel("number of exams n")
        ax.set_ylabel("% fewer nodes than Naive BT")
        ax.set_ylim(0, 101)
        ax.grid(alpha=.3, axis="y")
    fig.suptitle("Pruning effectiveness vs naive backtracking (instances needing search)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_pruning.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- markdown
def md_table(df: pd.DataFrame, floatfmt: dict[str, str] | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt.get(c, ".2f")) if not np.isnan(v) else "-")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(df, summ, prune, approx, tst):
    out = ["# ETE benchmark summary\n"]
    out.append(f"{df.inst.nunique()} instances x {df.algorithm.nunique()} algorithms = {len(df)} runs. "
               f"Every returned timetable verified: "
               f"{int(df.valid.sum())}/{len(df)} valid, {int(df.conflicts.sum())} conflicts, "
               f"{int(df.overloads.sum())} capacity overloads.\n")
    for fam in df.family.unique():
        s = summ[summ.family == fam]
        t = s[["n", "algorithm", "instances", "slots_mean", "gap_to_lb", "proved_optimal_pct",
               "nodes_mean", "time_mean"]].copy()
        t.columns = ["n", "algorithm", "inst", "mean slots", "slots above LB", "% proved opt",
                     "mean nodes", "mean time (s)"]
        out.append(f"\n## {FAMILY_TITLE.get(fam, fam)}\n")
        out.append(md_table(t, {"mean nodes": ".0f", "mean time (s)": ".3f", "% proved opt": ".0f"}))
    out.append("\n\n## Approximation ratios (slots / best known)\n")
    out.append(md_table(approx[["algorithm", "family", "n", "mean_ratio_vs_best", "max_ratio_vs_best",
                                "mean_ratio_vs_lb", "max_ratio_vs_lb"]], {}))
    out.append("\n\n## Pruning effectiveness (mean % node reduction vs Naive BT)\n")
    pr = prune.groupby(["family", "n", "algorithm"])["reduction_pct"].mean().reset_index()
    out.append(md_table(pr, {"reduction_pct": ".2f"}))
    out.append("\n\n## Paired Wilcoxon signed-rank tests (Bonferroni corrected)\n")
    if len(tst):
        out.append(md_table(tst[["family", "comparison", "metric", "n_min", "pairs", "median_diff",
                                 "a_better_pct", "b_better_pct", "p_value", "p_bonferroni"]],
                            {"p_value": ".2e", "p_bonferroni": ".2e", "median_diff": ".3f"}))
    with open(os.path.join(RES, "summary.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")


def main() -> None:
    df = load()
    summ = summary_table(df)
    prune = pruning(df)
    approx = approximation(df)
    tst = tests(df)
    make_plots(df, summ, prune, approx)
    write_markdown(df, summ, prune, approx, tst)
    print("analysis written to", os.path.abspath(RES))


if __name__ == "__main__":
    main()
