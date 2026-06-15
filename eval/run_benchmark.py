"""Run all cases through System A and System B and record the dependent variables.

Each case is run cfg.runs_per_case times (non-determinism — concern #7); report mean ± std.
Output: eval/results/benchmark_raw.csv and a printed summary.

Run:  python -m eval.run_benchmark
"""
from __future__ import annotations

import pandas as pd

from config.config import cfg
from eval.metrics import faithfulness_score, hallucination_rate, rejection_correct
from src import system_a_baseline
from src.system_b_reflection import graph as system_b


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


def summarize(df: pd.DataFrame) -> None:
    print("\n=== Summary by system (mean over runs) ===")
    print(df.groupby("system")[
        ["hallucination_rate", "faithfulness", "latency_s", "iterations"]
    ].agg(["mean", "std"]))
    print("\n=== Cross-domain rejection accuracy (RQ3) ===")
    cd = df[df["case_type"] == "cross_domain"]
    if not cd.empty:
        print(cd.groupby("system")["rejection_correct"].mean())
    print("\nTODO: add McNemar's test (paired A-vs-B) + confidence intervals (concern #9).")


if __name__ == "__main__":
    summarize(run())
