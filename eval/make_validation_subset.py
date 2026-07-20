"""Generate the judge-validation subset (proposal §4.5).

This is the piece that FEEDS judge_validation.py. It runs a stratified subset of cases
through BOTH systems, has the LLM judge label each claim, and writes:

  eval/results/judge_labels.csv          (claim_id, judge_label)   ← the judge's calls
  eval/results/human_labels_template.csv (claim_id, system, image, claim_text, human_label)

Workflow (proposal §4.5):
  1. python cli.py make-validation-subset      # this script — needs API key + built index
  2. 1-2 annotators fill the `human_label` column (1 = hallucinated, 0 = supported) per
     prompts/judge_rubric.md, then save the file as data/human_labels.csv
  3. python cli.py validate-judge              # computes Cohen's / Fleiss' kappa

Run:  python cli.py make-validation-subset [n_cases]
"""
from __future__ import annotations

import sys

import pandas as pd

from config.config import cfg
from eval.metrics import hallucination_rate
from src.systems import baseline as system_a
from src.systems.reflection import graph as system_b


def _stratified_cases(gt: pd.DataFrame, n_cases: int, seed: int = 0) -> pd.DataFrame:
    """Pick ~n_cases spanning difficulty levels (proportional per case_type)."""
    import random

    rng = random.Random(seed)  # fixed seed → reproducible subset
    picked: list = []
    for _, grp in gt.groupby("case_type"):
        idx = list(grp.index)
        rng.shuffle(idx)
        share = max(1, round(n_cases * len(grp) / len(gt)))
        picked.extend(idx[:share])
    return gt.loc[picked[:n_cases]]


def _query_for(row) -> str:
    if row["case_type"] == "cross_domain":
        return str(row["notes"])
    return f"What disease affects this {row['crop']}?"


def make_subset(n_cases: int = 13) -> None:
    gt = pd.read_csv(cfg.ground_truth_csv, comment="#")
    subset = _stratified_cases(gt, n_cases)
    print(f"Selected {len(subset)} cases (of {len(gt)}); running both systems + judge "
          f"→ ~{len(subset) * 2} outputs...")

    judge_rows, human_rows = [], []
    for _, case in subset.iterrows():
        image = str(cfg.images_dir / case["image_filename"])
        query = _query_for(case)
        for sys_name, system in (("A", system_a), ("B", system_b)):
            res = system.diagnose(image, query)
            scored = hallucination_rate(res["diagnosis"], res["context"],
                                        case["true_label"], image_path=image)
            for i, claim in enumerate(scored.get("claims", [])):
                cid = f"{case['image_filename']}__{sys_name}__{i}"
                judge_rows.append({"claim_id": cid, "judge_label": claim.get("label")})
                human_rows.append({
                    "claim_id": cid, "system": sys_name,
                    "image": case["image_filename"],
                    "claim_text": claim.get("text", ""),
                    "human_label": "",  # ← annotators fill this
                })

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    jpath = cfg.results_dir / "judge_labels.csv"
    hpath = cfg.results_dir / "human_labels_template.csv"
    pd.DataFrame(judge_rows).to_csv(jpath, index=False)
    pd.DataFrame(human_rows).to_csv(hpath, index=False)

    print(f"\nWrote {len(judge_rows)} claim-level labels:")
    print(f"  {jpath}   (judge labels)")
    print(f"  {hpath}   (blank human_label column — annotators fill this)")
    print("\nNEXT: annotators fill human_label (1=hallucinated, 0=supported) per "
          "prompts/judge_rubric.md,")
    print(f"      save as {cfg.root / 'data' / 'human_labels.csv'}, then: "
          "python cli.py validate-judge")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    make_subset(n)
