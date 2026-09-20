"""COST GATE: run a small stratified pilot and project full-run cost/time/tokens.

A full benchmark is 100 cases x 2 systems x cfg.runs_per_case(3) = 600 Actor calls on
gemini-2.5-pro, plus a Critic call per reflection iteration, plus a Judge call and a
RAGAS faithfulness call per output. That is not something to discover the price of
halfway through. This script runs the SAME code path as eval/run_benchmark.py on a
handful of cases with runs_per_case=1, measures real latency / tokens / 429s, and
extrapolates.

Outputs:
  eval/results/pilot_raw.csv    per-output measurements
  eval/results/pilot_cost.json  token totals, retry counts and the full-run projection

Run:  python cli.py pilot [n_cases]
"""
from __future__ import annotations

import json
import time

import pandas as pd

from config.config import cfg
from eval.metrics import faithfulness_score, hallucination_rate
from src.llm import client as llm_client
from src.systems import baseline as system_a
from src.systems.reflection import graph as system_b

# Published Gemini Developer API paid-tier rates, USD per 1M tokens.
# Recorded 2026-09-20 — VERIFY before quoting in the thesis, pricing changes.
# gemini-2.5-pro input is tiered: $1.25 <=200k prompt, $2.50 above.
_PRICES = {
    "gemini-2.5-pro":   {"in": 1.25, "out": 10.00},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
}


def _pilot_cases(gt: pd.DataFrame, n: int) -> pd.DataFrame:
    """Stratified pilot slice: spans the confusable triad + both cross-domain directions.

    Deliberately not random — a pilot that misses cross_domain would under-measure
    cost, because REJECT outputs are short and cheap while ambiguous outputs are long.
    """
    amb = gt[gt["case_type"] == "ambiguous"]
    cross = gt[gt["case_type"] == "cross_domain"]
    n_cross = max(2, round(n * cfg.adversarial_fraction))
    n_amb = n - n_cross

    picks = []
    # one per distinct (crop, true_label) among ambiguous, in order, until n_amb
    for _, grp in amb.groupby(["crop", "true_label"], sort=True):
        picks.append(grp.index[0])
    picks = picks[:n_amb]
    # cross-domain: Direction A (off-crop image: corn/grape) and Direction B (tomato/potato)
    dir_a = cross[cross["crop"].isin(["corn", "grape"])]
    dir_b = cross[cross["crop"].isin(["tomato", "potato"])]
    for src, k in ((dir_a, n_cross // 2 + n_cross % 2), (dir_b, n_cross // 2)):
        picks.extend(list(src.index[:k]))
    return gt.loc[picks]


def _query_for(row) -> str:
    """Must match run_benchmark._default_query exactly, or the pilot measures the wrong thing."""
    if row["case_type"] == "cross_domain":
        return str(row["notes"])
    return f"What disease affects this {row['crop']}?"


def _cost(usage: dict) -> float:
    total = 0.0
    for model, u in usage.items():
        p = _PRICES.get(model)
        if not p:
            continue
        total += u["input_tokens"] / 1e6 * p["in"] + u["output_tokens"] / 1e6 * p["out"]
    return total


def run_pilot(n_cases: int = 8) -> dict:
    gt = pd.read_csv(cfg.ground_truth_csv, comment="#")
    subset = _pilot_cases(gt, n_cases)
    llm_client.reset_usage()

    print(f"PILOT: {len(subset)} cases x 2 systems x 1 run = {len(subset)*2} diagnoses")
    print(f"  (full run will be 100 x 2 x {cfg.runs_per_case} = {100*2*cfg.runs_per_case})\n")
    for _, c in subset.iterrows():
        print(f"  - {c['image_filename']:44s} {c['case_type']:12s} {c['true_label']}")
    print()

    rows = []
    t_start = time.perf_counter()
    for i, (_, case) in enumerate(subset.iterrows(), 1):
        image = str(cfg.images_dir / case["image_filename"])
        query = _query_for(case)
        for sys_name, system in (("A", system_a), ("B", system_b)):
            t0 = time.perf_counter()
            res = system.diagnose(image, query)
            t_diag = time.perf_counter() - t0

            t1 = time.perf_counter()
            hr = hallucination_rate(res["diagnosis"], res["context"],
                                    case["true_label"], image_path=image)
            t_judge = time.perf_counter() - t1

            t2 = time.perf_counter()
            try:
                faith = faithfulness_score(res["diagnosis"], res["context"], query)
                faith_err = ""
            except Exception as e:  # noqa: BLE001
                faith, faith_err = None, f"{type(e).__name__}: {e}"
                print(f"    [faithfulness failed: {faith_err}]")
            t_faith = time.perf_counter() - t2

            rows.append({
                "image": case["image_filename"], "case_type": case["case_type"],
                "true_label": case["true_label"], "system": res["system"],
                "iterations": res["iterations"],
                "diagnose_s": round(t_diag, 2), "judge_s": round(t_judge, 2),
                "faithfulness_s": round(t_faith, 2),
                "total_s": round(t_diag + t_judge + t_faith, 2),
                "hallucination_rate": hr.get("hallucination_rate"),
                "n_claims": len(hr.get("claims", [])),
                "faithfulness": faith, "faithfulness_error": faith_err,
                "diagnosis_head": str(res["diagnosis"])[:120].replace("\n", " "),
            })
            print(f"  [{i}/{len(subset)}] {sys_name} {case['image_filename'][:34]:34s} "
                  f"iter={res['iterations']} diag={t_diag:5.1f}s judge={t_judge:4.1f}s "
                  f"faith={t_faith:4.1f}s hr={hr.get('hallucination_rate')}")
    wall = time.perf_counter() - t_start

    df = pd.DataFrame(rows)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.results_dir / "pilot_raw.csv", index=False)

    usage = {m: dict(u) for m, u in llm_client.USAGE.items()}
    pilot_outputs = len(df)
    full_outputs = 100 * 2 * cfg.runs_per_case
    scale = full_outputs / pilot_outputs

    pilot_cost = _cost(usage)
    projection = {
        "pilot_outputs": pilot_outputs,
        "full_run_outputs": full_outputs,
        "scale_factor": round(scale, 2),
        "pilot_wall_s": round(wall, 1),
        "projected_full_run_wall_s": round(wall * scale, 1),
        "projected_full_run_wall_h": round(wall * scale / 3600, 2),
        "pilot_cost_usd": round(pilot_cost, 4),
        "projected_full_run_cost_usd": round(pilot_cost * scale, 2),
        "note": "RAGAS faithfulness uses its own ChatGoogleGenerativeAI instance, so its "
                "tokens are NOT in `usage` and NOT in the cost figures; its wall time IS "
                "measured in faithfulness_s. Treat cost as a lower bound.",
    }
    report = {"usage_by_model": usage, "retries": dict(llm_client.RETRIES),
              "latency_by_system": df.groupby("system")[
                  ["diagnose_s", "judge_s", "faithfulness_s", "total_s"]
              ].mean().round(2).to_dict(), "projection": projection,
              "prices_usd_per_1m_tokens": _PRICES, "prices_recorded": "2026-09-20"}
    (cfg.results_dir / "pilot_cost.json").write_text(json.dumps(report, indent=2))

    print("\n=== TOKENS (Actor/Critic/Judge via LLMClient; RAGAS not counted) ===")
    for m, u in sorted(usage.items()):
        print(f"  {m:20s} calls={u['calls']:4d} in={u['input_tokens']:8d} "
              f"out={u['output_tokens']:7d} total={u['total_tokens']:8d}")
    print(f"\n=== RETRIES === {dict(llm_client.RETRIES)}")
    print("\n=== LATENCY (mean s per output) ===")
    print(df.groupby("system")[["diagnose_s", "judge_s", "faithfulness_s", "total_s"]]
          .mean().round(2).to_string())
    print("\n=== ITERATIONS (System B reflection loop) ===")
    print(df[df.system == "B_reflection"]["iterations"].describe().round(2).to_string())
    print("\n=== PROJECTION TO FULL RUN ===")
    for k, v in projection.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {cfg.results_dir/'pilot_raw.csv'} and {cfg.results_dir/'pilot_cost.json'}")
    return report


if __name__ == "__main__":
    import sys
    run_pilot(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
