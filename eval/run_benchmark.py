"""Run all cases through System A and System B and record the dependent variables.

Each case is run cfg.runs_per_case times (non-determinism — concern #7); report mean ± std.
Output: eval/results/benchmark_raw.csv and a printed summary with paired significance tests.

Run:  python cli.py benchmark
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import cfg
from eval.metrics import faithfulness_score, hallucination_rate, rejection_correct
from src.systems import baseline as system_a_baseline
from src.systems.reflection import graph as system_b


def _default_query(row) -> str:
    if row["case_type"] == "cross_domain":
        return str(row["notes"])  # e.g., "query asks about corn disease"
    return f"What disease affects this {row['crop']}?"


def run() -> pd.DataFrame:
    gt = pd.read_csv(cfg.ground_truth_csv, comment="#")
    rows = []
    for _, case in gt.iterrows():
        image = str(cfg.images_dir / case["image_filename"])
        query = _default_query(case)
        for run_i in range(cfg.runs_per_case):
            for system in (system_a_baseline, system_b):
                res = system.diagnose(image, query)
                hr = hallucination_rate(res["diagnosis"], res["context"],
                                        case["true_label"], image_path=image)
                try:
                    faith = faithfulness_score(res["diagnosis"], res["context"], query)
                except Exception as e:  # don't let one RAGAS failure abort the whole run
                    print(f"  [faithfulness failed for {case['image_filename']}: {e}]")
                    faith = None
                rows.append({
                    "image": case["image_filename"],
                    "true_label": case["true_label"],
                    "case_type": case["case_type"],
                    "system": res["system"],
                    "run": run_i,
                    "iterations": res["iterations"],
                    "latency_s": res["latency_s"],
                    "hallucination_rate": hr.get("hallucination_rate"),
                    "faithfulness": faith,
                    "rejection_correct": rejection_correct(res["diagnosis"], case["true_label"]),
                })
    df = pd.DataFrame(rows)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.results_dir / "benchmark_raw.csv", index=False)
    return df


def _paired_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot to one row per (image, run) with an A column and a B column.

    Pairing on the SAME image+run is what makes the A-vs-B test *paired* — each case
    sees both systems under the same conditions, so the comparison controls for
    case difficulty (concern #9).
    """
    wide = df.pivot_table(index=["image", "run"], columns="system", values=metric)
    return wide.dropna()


def _bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000) -> tuple[float, float]:
    """95% percentile bootstrap CI for the mean paired difference (B − A)."""
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)  # fixed seed → reproducible CI
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _mcnemar(df: pd.DataFrame, binary_col: str) -> None:
    """McNemar's test on a paired binary outcome (B correct vs A correct)."""
    from statsmodels.stats.contingency_tables import mcnemar

    wide = _paired_frame(df, binary_col)
    cols = list(wide.columns)
    if len(cols) < 2:
        print(f"  [{binary_col}: need both systems present to pair]")
        return
    a_col = next((c for c in cols if "A_" in c or "baseline" in c), cols[0])
    b_col = next((c for c in cols if "B_" in c or "reflection" in c), cols[-1])
    a = wide[a_col].astype(bool)
    b = wide[b_col].astype(bool)
    # 2x2 contingency of (A correct?, B correct?)
    table = [[int((a & b).sum()),  int((a & ~b).sum())],
             [int((~a & b).sum()), int((~a & ~b).sum())]]
    res = mcnemar(table, exact=len(wide) < 25)  # exact binomial for small n
    print(f"  {binary_col}: A={a.mean():.3f}  B={b.mean():.3f}  "
          f"discordant(b>a={table[1][0]}, a>b={table[0][1]})  "
          f"McNemar p={res.pvalue:.4f}")


def _paired_diff(df: pd.DataFrame, metric: str) -> None:
    """Mean paired difference (B − A) with a 95% bootstrap CI for a continuous metric."""
    wide = _paired_frame(df, metric)
    cols = list(wide.columns)
    if len(cols) < 2:
        print(f"  [{metric}: need both systems present to pair]")
        return
    a_col = next((c for c in cols if "A_" in c or "baseline" in c), cols[0])
    b_col = next((c for c in cols if "B_" in c or "reflection" in c), cols[-1])
    diffs = (wide[b_col] - wide[a_col]).to_numpy()
    lo, hi = _bootstrap_ci(diffs)
    sig = "" if (lo <= 0 <= hi) else "  *(95% CI excludes 0)*"
    print(f"  {metric}: mean(B−A)={diffs.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]{sig}")


def summarize(df: pd.DataFrame) -> None:
    print("\n=== Summary by system (mean over runs) ===")
    print(df.groupby("system")[
        ["hallucination_rate", "faithfulness", "latency_s", "iterations"]
    ].agg(["mean", "std"]))

    print("\n=== RQ1: paired A-vs-B on continuous metrics (mean diff + bootstrap CI) ===")
    _paired_diff(df, "hallucination_rate")
    _paired_diff(df, "faithfulness")

    print("\n=== RQ1: paired A-vs-B on 'any hallucination' (McNemar) ===")
    tmp = df.copy()
    tmp["no_halluc"] = tmp["hallucination_rate"].fillna(1.0) == 0  # 'correct' = zero hallucinations
    _mcnemar(tmp, "no_halluc")

    print("\n=== RQ2: latency overhead ===")
    lat = df.groupby("system")["latency_s"].mean()
    print(lat)

    print("\n=== RQ3: cross-domain rejection accuracy (McNemar) ===")
    cd = df[df["case_type"] == "cross_domain"]
    if not cd.empty:
        print(cd.groupby("system")["rejection_correct"].mean())
        _mcnemar(cd, "rejection_correct")
    else:
        print("  [no cross_domain cases in ground_truth.csv]")


if __name__ == "__main__":
    summarize(run())
