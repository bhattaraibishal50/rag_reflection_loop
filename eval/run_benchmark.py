"""Run all cases through System A and System B and record the dependent variables.

Each case is run cfg.runs_per_case times (non-determinism — concern #7); report mean ± std.
Output: eval/results/benchmark_raw.csv and a printed summary with paired significance tests.

Run:  python cli.py benchmark
"""
from __future__ import annotations

import concurrent.futures
import threading
import time

import numpy as np
import pandas as pd

from config.config import cfg
from eval.metrics import faithfulness_score, hallucination_rate, rejection_correct
from src.systems import baseline as system_a_baseline
from src.systems.reflection import graph as system_b

_print_lock = threading.Lock()


def _default_query(row) -> str:
    if row["case_type"] == "cross_domain":
        return str(row["notes"])  # e.g., "query asks about corn disease"
    return f"What disease affects this {row['crop']}?"


def _one_unit(case, run_i: int) -> list[dict]:
    """Run BOTH systems for one (case, run) pair and return their result rows.

    A and B stay together inside a single unit of work on purpose: the analysis
    pairs them on (image, run), so both must exist for that key or the pair is
    dropped by `_paired_frame`. Keeping them in one task means a failure loses a
    whole pair rather than orphaning half of one.
    """
    image = str(cfg.images_dir / case["image_filename"])
    query = _default_query(case)
    out = []
    for system in (system_a_baseline, system_b):
        res = system.diagnose(image, query)
        hr = hallucination_rate(res["diagnosis"], res["context"],
                                case["true_label"], image_path=image)
        try:
            faith = faithfulness_score(res["diagnosis"], res["context"], query)
        except Exception as e:  # don't let one RAGAS failure abort the whole run
            with _print_lock:
                print(f"  [faithfulness failed for {case['image_filename']}: {e}]")
            faith = None
        out.append({
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
    return out


def run(workers: int = 1) -> pd.DataFrame:
    """Execute the benchmark, optionally running (case, run) units concurrently.

    `workers` only changes WALL TIME, never cost or results: billing is per token,
    and each unit is independent. Latency per call is still measured inside the
    system modules, so `latency_s` stays valid under concurrency — but note that
    with many workers the API may slow individual calls, so latency comparisons
    should come from a consistent worker count (RQ2 reports A vs B measured in the
    same run, so the comparison stays internally fair).

    Rate limits are the real ceiling. The pilot recorded zero 429s, and
    `with_retry` backs off if they appear; raise workers only while 429s stay low.
    """
    # Build the Chroma client ONCE, here, before any worker starts. Constructing
    # it concurrently races inside chromadb's shared-client registry.
    from src.rag.retriever import warm_retriever
    warm_retriever()

    gt = pd.read_csv(cfg.ground_truth_csv, comment="#")

    # --- RESUME ---------------------------------------------------------------
    # A (image, run) pair counts as done only when BOTH systems are present: the
    # analysis pairs on that key, so a half-finished pair is useless. Completed
    # rows are carried forward and merged with the new ones.
    done_keys: set = set()
    prior_rows: list[dict] = []
    out_csv = cfg.results_dir / "benchmark_raw.csv"
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        if not prev.empty:
            counts = prev.groupby(["image", "run"])["system"].nunique()
            done_keys = {k for k, v in counts.items() if v >= 2}
            keep = prev.set_index(["image", "run"]).index.isin(list(done_keys))
            prior_rows = prev[keep].to_dict("records")
            print(f"RESUME: {len(done_keys)} complete (image, run) pairs found; "
                  f"carrying {len(prior_rows)} rows forward")

    units = [(case, run_i)
             for _, case in gt.iterrows()
             for run_i in range(cfg.runs_per_case)
             if (case["image_filename"], run_i) not in done_keys]

    # --- PRIORITY ORDER -------------------------------------------------------
    # cross_domain first (RQ3 has no data at all and is the most fragile result),
    # then non-tomato crops (cross-crop generality), then the rest. If the budget
    # runs out again, what is lost is additional tomato/ambiguous data we already
    # have plenty of, rather than a research question.
    def _priority(u):
        case, _ = u
        if case["case_type"] == "cross_domain":
            return 0
        return 1 if case["crop"] != "tomato" else 2

    units.sort(key=_priority)
    if units:
        from collections import Counter
        print("ORDER:", dict(Counter(
            "cross_domain" if c["case_type"] == "cross_domain"
            else ("non-tomato" if c["crop"] != "tomato" else "tomato-ambiguous")
            for c, _ in units)))
    total = len(units)
    print(f"BENCHMARK: {len(gt)} cases x {cfg.runs_per_case} runs x 2 systems; "
          f"{total} units remaining = {total * 2} diagnoses  (workers={workers})")
    if total == 0:
        print("Nothing to do — all units already complete.")

    rows: list[dict] = list(prior_rows)
    done = 0
    t0 = time.perf_counter()

    def _record(result: list[dict]) -> None:
        nonlocal done
        rows.extend(result)
        done += 1
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed else 0
        eta = (total - done) / rate / 60 if rate else float("nan")
        print(f"  [{done}/{total}] {result[0]['image'][:38]:38s} "
              f"elapsed={elapsed/60:5.1f}m eta={eta:5.1f}m", flush=True)

    if workers <= 1:
        for case, run_i in units:
            _record(_one_unit(case, run_i))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one_unit, case, run_i): (case, run_i)
                       for case, run_i in units}
            for fut in concurrent.futures.as_completed(futures):
                case, run_i = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:  # noqa: BLE001 — one unit must not kill the run
                    with _print_lock:
                        print(f"  [UNIT FAILED {case['image_filename']} run={run_i}: "
                              f"{type(e).__name__}: {e}]", flush=True)
                    continue
                with _print_lock:
                    _record(result)

    df = pd.DataFrame(rows)
    # Deterministic order regardless of completion order, so the CSV is stable.
    if not df.empty:
        df = df.sort_values(["image", "run", "system"]).reset_index(drop=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.results_dir / "benchmark_raw.csv", index=False)
    try:
        from src.llm.embeddings import cache_stats
        print("embedding cache:", cache_stats())
    except Exception:
        pass
    print(f"\nWrote {cfg.results_dir / 'benchmark_raw.csv'} "
          f"({len(df)} rows, {time.perf_counter() - t0:.0f}s)")
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
    import sys
    summarize(run(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
