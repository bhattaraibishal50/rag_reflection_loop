"""Generate Chapter 4 figures from eval/results/benchmark_raw.csv.

Produces (into eval/results/):
  fig_rq1_metrics.png   grouped bars: hallucination rate + faithfulness by system (± sd)
  fig_rq2_latency.png   inference latency by system (± sd)
  fig_rq3_rejection.png cross-domain rejection accuracy by system (if any cross_domain cases)

Run:  python cli.py plots      (after `python cli.py benchmark`)
"""
from __future__ import annotations

import pandas as pd

from config.config import cfg

# Colour-blind-safe pair (Okabe-Ito): blue = baseline, orange = reflection.
_COLORS = {"A_baseline": "#0072B2", "B_reflection": "#E69F00"}


def _load() -> pd.DataFrame:
    path = cfg.results_dir / "benchmark_raw.csv"
    if not path.exists():
        raise SystemExit(f"No results at {path}. Run `python cli.py benchmark` first.")
    return pd.read_csv(path)


def _colors_for(systems) -> list:
    return [_COLORS.get(s, "#888888") for s in systems]


def make_plots(df: pd.DataFrame | None = None) -> list:
    import matplotlib
    matplotlib.use("Agg")  # headless — works in Docker/CI with no display
    import matplotlib.pyplot as plt
    import numpy as np

    if df is None:
        df = _load()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    outputs: list = []

    # --- RQ1: hallucination rate + faithfulness, grouped by system ---
    metrics = ["hallucination_rate", "faithfulness"]
    means = df.groupby("system")[metrics].mean()
    stds = df.groupby("system")[metrics].std().fillna(0.0)
    systems = list(means.index)
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(systems))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, s in enumerate(systems):
        ax.bar(x + i * width, means.loc[s, metrics].to_numpy(), width,
               yerr=stds.loc[s, metrics].to_numpy(), capsize=4,
               label=s, color=_COLORS.get(s, "#888888"))
    ax.set_xticks(x + width * (len(systems) - 1) / 2)
    ax.set_xticklabels(["Hallucination Rate", "Faithfulness"])
    ax.set_ylabel("score (0–1)")
    ax.set_ylim(0, 1)
    ax.set_title("RQ1: Hallucination rate & faithfulness by system")
    ax.legend()
    fig.tight_layout()
    p = cfg.results_dir / "fig_rq1_metrics.png"
    fig.savefig(p, dpi=150); plt.close(fig); outputs.append(p)

    # --- RQ2: latency ---
    lat = df.groupby("system")["latency_s"].agg(["mean", "std"]).fillna(0.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(lat.index, lat["mean"], yerr=lat["std"], capsize=4,
           color=_colors_for(lat.index))
    ax.set_ylabel("latency (s)")
    ax.set_title("RQ2: Inference latency by system")
    fig.tight_layout()
    p = cfg.results_dir / "fig_rq2_latency.png"
    fig.savefig(p, dpi=150); plt.close(fig); outputs.append(p)

    # --- RQ3: cross-domain rejection accuracy (only if such cases exist) ---
    cd = df[df["case_type"] == "cross_domain"]
    if not cd.empty:
        rej = cd.groupby("system")["rejection_correct"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(rej.index, rej.to_numpy(), color=_colors_for(rej.index))
        ax.set_ylabel("rejection accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("RQ3: Cross-domain rejection accuracy by system")
        fig.tight_layout()
        p = cfg.results_dir / "fig_rq3_rejection.png"
        fig.savefig(p, dpi=150); plt.close(fig); outputs.append(p)

    print("Wrote figures:")
    for p in outputs:
        print(f"  {p}")
    return outputs


if __name__ == "__main__":
    make_plots()
