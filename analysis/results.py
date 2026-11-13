import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.evaluate import save_rows

METHODS = {
    "DAgger p=0.5": "runs/dagger_p0.5",
    "BC": "runs/bc",
    "RLPD": "runs/rlpd_ratio0.5",
    "DAgger p=0": "runs/dagger_p0.0",
    "SAC seul": "runs/rlpd_ratio0.0",
}
CURVE_METHODS = ["DAgger p=0.5", "DAgger p=0", "RLPD", "SAC seul"]
COLORS = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
DASHED = {"DAgger p=0"}  # se superpose souvent à DAgger p=0.5
INK, MUTED, GRID = "#3d3d3a", "#5f5f5a", "#e5e5e2"
BUDGET = 12_000  # budget en ligne de DAgger : comparaison à budget égal
LAST_K = 5  # analyse exploratoire : moyenne des K dernières évaluations
COMPARISONS = [
    ("DAgger p=0.5", "RLPD"), ("DAgger p=0", "RLPD"),
    ("DAgger p=0.5", "RLPD @12k"), ("DAgger p=0", "RLPD @12k"),
    ("RLPD", "SAC seul"), ("DAgger p=0", "DAgger p=0.5"),
    ("BC", "DAgger p=0"), ("BC", "DAgger p=0.5"),
]
N_BOOT = 10_000
rng = np.random.default_rng(0)


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def curves(name):
    return [read(p) for p in sorted(Path(METHODS[name]).glob("seed_*/curve.csv"))]


def column(rows, key):
    return np.array([float(r[key]) for r in rows])


def at_budget(curve, budget):
    return [r for r in curve if float(r["env_steps"]) <= budget][-1]


def first_lap(curve):
    finished = [float(r["env_steps"]) for r in curve if r["finished"] == "1"]
    return finished[0] if finished else np.nan


def load_final():
    final = {name: read(f"{d}/eval.csv") for name, d in METHODS.items() if Path(d, "eval.csv").exists()}
    if "RLPD" in final:
        final["RLPD @12k"] = [at_budget(c, BUDGET) for c in curves("RLPD")]
    return final


def bootstrap_mean(x):
    return x[rng.integers(len(x), size=(N_BOOT, len(x)))].mean(axis=1)


def ci95(x):
    return np.percentile(bootstrap_mean(x), [2.5, 97.5])


def summarize(final):
    rows = []
    for name, runs in final.items():
        x = column(runs, "return")
        low, high = ci95(x)
        row = {"method": name, "n_seeds": len(x), "mean": x.mean(), "ci95_low": low, "ci95_high": high,
               "std": x.std(ddof=1), "median": np.median(x), "iqm": stats.trim_mean(x, 0.25),
               "success_rate": column(runs, "finished").mean(), "lap_time_finished": np.nanmean(column(runs, "lap_time")),
               "first_lap_median": np.nan, "never_finished": np.nan, "last5_mean": np.nan, "last5_ci_low": np.nan, "last5_ci_high": np.nan}
        if name == "BC":
            row["first_lap_median"], row["never_finished"] = 0.0, 0
        if name in CURVE_METHODS:
            c = curves(name)
            laps = np.array([first_lap(curve) for curve in c])
            last = np.array([column(curve, "return")[-LAST_K:].mean() for curve in c])
            row["first_lap_median"] = np.nanmedian(laps) if (~np.isnan(laps)).any() else np.nan
            row["never_finished"] = int(np.isnan(laps).sum())
            row["last5_mean"] = last.mean()
            row["last5_ci_low"], row["last5_ci_high"] = ci95(last)
        rows.append(row)
    return rows


def holm(p):
    order, adjusted, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def compare(final, expert):
    rows = []
    for a, b in COMPARISONS:
        if a in final and b in final:
            x, y = column(final[a], "return"), column(final[b], "return")
            low, high = np.percentile(bootstrap_mean(y) - bootstrap_mean(x), [2.5, 97.5])
            rows.append({"comparison": f"{b} - {a}", "diff_mean": y.mean() - x.mean(), "ci95_low": low, "ci95_high": high,
                         "p_welch": stats.ttest_ind(x, y, equal_var=False).pvalue,
                         "p_mannwhitney": stats.mannwhitneyu(x, y, alternative="two-sided").pvalue})
    for name, runs in final.items():
        x = column(runs, "return") - expert
        low, high = ci95(x)
        rows.append({"comparison": f"{name} - Expert", "diff_mean": x.mean(), "ci95_low": low, "ci95_high": high,
                     "p_welch": stats.ttest_1samp(x, 0.0).pvalue,  # expert déterministe : test à un échantillon
                     "p_mannwhitney": 1.0 if np.allclose(x, 0) else stats.wilcoxon(x).pvalue})
    for key in ["p_welch", "p_mannwhitney"]:
        adjusted = holm(np.array([r[key] for r in rows]))  # une seule famille de tests
        for r, p in zip(rows, adjusted):
            r[f"{key}_holm"] = p
    return rows


def curve_matrix(name, grid, key):
    values = []
    for c in curves(name):
        steps = column(c, "env_steps")
        idx = np.searchsorted(steps, grid, side="right") - 1  # politique évaluée la plus récente
        v = np.where(idx >= 0, column(c, key)[np.maximum(idx, 0)], np.nan)
        values.append(np.where(grid <= steps.max(), v, np.nan))
    return np.array(values)


def style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_mean_band(ax, name, grid, key, scale=1.0):
    v = curve_matrix(name, grid, key) * scale
    ok = ~np.isnan(v).any(axis=0)
    boot = v[:, ok][rng.integers(len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    ax.fill_between(grid[ok], *np.percentile(boot, [2.5, 97.5], axis=0), color=COLORS[name], alpha=0.2, linewidth=0)
    ax.plot(grid[ok], v[:, ok].mean(axis=0), color=COLORS[name], linewidth=1.8, linestyle="--" if name in DASHED else "-", label=name)


def plot_median(ax, name, grid, key):
    v = curve_matrix(name, grid, key)
    ok = ~np.isnan(v).any(axis=0)
    ax.plot(grid[ok], np.median(v[:, ok], axis=0), color=COLORS[name], linewidth=1.8, linestyle="--" if name in DASHED else "-", label=name)


def available(names):
    return [n for n in names if list(Path(METHODS[n]).glob("seed_*/curve.csv"))]


def plot_curves(expert, path):
    grid = np.arange(0, 30_001, 250)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for ax, ylim in zip(axes, [None, (90, 106)]):
        style(ax)
        ax.axhline(expert, color=MUTED, linewidth=1.0, linestyle="--")
        ax.axvline(BUDGET, color=MUTED, linewidth=0.8, linestyle=":")
        for name in available(CURVE_METHODS):
            (plot_mean_band if ylim is None else plot_median)(ax, name, grid, "return")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel("Retour (moyenne, IC 95 %)" if ylim is None else "Médiane (zoom)")
    axes[0].text(BUDGET, 1.01, "budget DAgger", transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", color=MUTED, fontsize=8)
    axes[1].text(30_000, expert, "expert ", ha="right", va="bottom", color=MUTED, fontsize=8)
    axes[1].set_xlabel("Pas d'environnement en ligne")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=4, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)


def plot_success(path):
    grid = np.arange(0, 30_001, 250)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    style(ax)
    ax.axvline(BUDGET, color=MUTED, linewidth=0.8, linestyle=":")
    for name in available(CURVE_METHODS):
        plot_mean_band(ax, name, grid, "finished", scale=100)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("Graines qui finissent le tour (%)")
    ax.set_xlabel("Pas d'environnement en ligne")
    ax.text(BUDGET, 1.01, "budget DAgger", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=MUTED, fontsize=8)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=4, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)


def plot_final(final, expert, path):
    names = list(final)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, ylim in zip(axes, [None, (95, 106)]):
        style(ax)
        ax.axhline(expert, color=MUTED, linewidth=1.0, linestyle="--")
        for i, name in enumerate(names):
            x = column(final[name], "return")
            color = COLORS[name.replace(" @12k", "")]
            jitter = np.linspace(-0.15, 0.15, len(x))
            hollow = name.endswith("@12k")  # même méthode que RLPD, lue à budget égal
            ax.scatter(i + jitter, x, s=24, facecolor="white" if hollow else color, edgecolor=color if hollow else "white", linewidth=1.0, zorder=3)
            low, high = ci95(x)
            ax.errorbar(i + 0.3, x.mean(), yerr=[[x.mean() - low], [high - x.mean()]], color=INK, capsize=3, fmt="o", markersize=4)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel("Retour final par graine" if ylim is None else "Zoom autour de l'expert")
    axes[1].set_xticks(range(len(names)), names)
    axes[1].text(-0.45, expert, "expert", ha="left", va="bottom", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def fr(x, digits=2):
    if np.isnan(x):
        return "--"
    text = f"{x:.{digits}f}".replace(".", "{,}")
    return "$-$" + text[1:] if text.startswith("-") else text


def write_tables(summary, tests, expert_row, folder):
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Méthode & Retour [IC 95\,\%] & Médiane & IQM & Réussite & Temps (s) & 1\textsuperscript{er} tour (pas) & 5 dern. [IC 95\,\%] \\",
             r"\midrule",
             f"Expert & {fr(expert_row['return'])} & -- & -- & 100\\,\\% & {fr(expert_row['lap_time'], 1)} & -- & -- \\\\"]
    order = ["BC", "DAgger p=0", "DAgger p=0.5", "RLPD @12k", "RLPD", "SAC seul"]
    for r in sorted(summary, key=lambda r: order.index(r["method"])):
        last5 = "--" if np.isnan(r["last5_mean"]) else f"{fr(r['last5_mean'])} [{fr(r['last5_ci_low'])} ; {fr(r['last5_ci_high'])}]"
        lap = "--" if np.isnan(r["first_lap_median"]) else f"{r['first_lap_median']:.0f}"
        if not np.isnan(r["never_finished"]) and r["never_finished"] > 0:
            lap += f" ({int(r['never_finished'])} jamais)"
        lines.append(f"{r['method']} & {fr(r['mean'])} [{fr(r['ci95_low'])} ; {fr(r['ci95_high'])}] & {fr(r['median'])} & {fr(r['iqm'])} & "
                     f"{100 * r['success_rate']:.0f}\\,\\% & {fr(r['lap_time_finished'], 1)} & {lap} & {last5} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(folder, "summary_table.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Comparaison & $\Delta$ [IC 95\,\%] & $p$ Welch (Holm) & $p$ Mann--Whitney (Holm) \\", r"\midrule"]
    for r in tests:
        lines.append(f"{r['comparison'].replace(' - ', ' $-$ ')} & {fr(r['diff_mean'])} [{fr(r['ci95_low'])} ; {fr(r['ci95_high'])}] & "
                     f"{fr(r['p_welch_holm'], 3)} & {fr(r['p_mannwhitney_holm'], 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(folder, "tests_table.tex").write_text("\n".join(lines) + "\n")


def main():
    expert_row = {k: float(v) for k, v in read("runs/expert/eval.csv")[0].items()}
    expert = expert_row["return"]
    final = load_final()
    summary, tests = summarize(final), compare(final, expert)
    save_rows("runs/analysis/summary.csv", summary)
    save_rows("runs/analysis/tests.csv", tests)
    plot_curves(expert, "report/figures/learning_curves.pdf")
    plot_success("report/figures/success_rate.pdf")
    plot_final(final, expert, "report/figures/final_scores.pdf")
    write_tables(summary, tests, expert_row, "report/figures")

    print(f"Expert : {expert:.2f}")
    for r in summary:
        print(f"{r['method']:<13} | n={r['n_seeds']} | moyenne {r['mean']:7.2f} [{r['ci95_low']:7.2f}, {r['ci95_high']:7.2f}] | "
              f"médiane {r['median']:7.2f} | IQM {r['iqm']:7.2f} | réussite {r['success_rate']:.0%} | "
              f"1er tour {r['first_lap_median']:7.0f} ({r['never_finished']} jamais) | 5 dernières {r['last5_mean']:7.2f}")
    for r in tests:
        print(f"{r['comparison']:<30} | diff {r['diff_mean']:7.2f} [{r['ci95_low']:7.2f}, {r['ci95_high']:7.2f}] | "
              f"Welch-Holm {r['p_welch_holm']:.3f} | MW-Holm {r['p_mannwhitney_holm']:.3f}")


if __name__ == "__main__":
    main()
