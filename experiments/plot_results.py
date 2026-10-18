"""Plots for the report, from the CSV files written by run_experiments.py.

    python experiments/plot_results.py [--results-dir experiments/results]
Creates convergence.png (best score per generation, mean +- std over seeds) and
final_scores.png (box plot of final best scores per configuration, with the random baseline)."""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def convergence(results_dir):
    rows = read_rows(os.path.join(results_dir, "history.csv"))
    curves = defaultdict(lambda: defaultdict(dict))          # config -> seed -> {generation: best}
    for r in rows:
        curves[r["config"]][r["seed"]][int(r["generation"])] = float(r["best"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for config, by_seed in curves.items():
        max_gen = max(max(d) for d in by_seed.values())
        mat = []
        for d in by_seed.values():                          # runs that stopped early keep their last value
            last, row = None, []
            for g in range(max_gen + 1):
                last = d.get(g, last)
                row.append(last if last is not None else np.nan)
            mat.append(row)
        mat = np.array(mat, dtype=float)
        mean, std = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
        x = np.arange(max_gen + 1)
        ax.plot(x, mean, label=config)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("generation")
    ax.set_ylabel("best SP score")
    ax.set_title("Convergence (mean ± std over seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "convergence.png"), dpi=150)


def final_scores(results_dir):
    summary = read_rows(os.path.join(results_dir, "summary.csv"))
    random_scores = [float(r["score"]) for r in read_rows(os.path.join(results_dir, "random_baseline.csv"))]
    groups = defaultdict(list)
    for r in summary:
        groups[r["config"]].append(float(r["best_score"]))
    labels = ["random"] + list(groups)
    data = [random_scores] + list(groups.values())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, labels=labels)
    ax.set_ylabel("SP score")
    ax.set_title("Final best score per configuration vs random MSAs")
    plt.setp(ax.get_xticklabels(), rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "final_scores.png"), dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/results")
    a = ap.parse_args()
    convergence(a.results_dir)
    final_scores(a.results_dir)
    print(f"Plots saved in {a.results_dir}/")
