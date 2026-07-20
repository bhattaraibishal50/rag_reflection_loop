"""GATE (proposal §3.1): validate the LLM judge against human annotators BEFORE trusting it.

Workflow:
1. Generate diagnoses for a stratified subset (25-30) and save them.
2. Humans label each claim using prompts/judge_rubric.md -> data/human_labels.csv
3. Run the LLM judge on the same claims.
4. Compute Cohen's kappa. Only proceed to the full benchmark if kappa >= 0.6.

Run:  python cli.py validate-judge
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from config.config import cfg

HUMAN_LABELS = cfg.root / "data" / "human_labels.csv"  # columns: claim_id, human_label
JUDGE_LABELS = cfg.results_dir / "judge_labels.csv"     # columns: claim_id, judge_label


def compute_agreement(human: list[int], judge: list[int]) -> None:
    kappa = cohen_kappa_score(human, judge)
    print(f"Cohen's kappa: {kappa:.3f}")
    print("(target >= 0.60 'substantial' on the Landis & Koch scale)\n")
    print("Confusion matrix (rows=human, cols=judge):")
    print(confusion_matrix(human, judge))
    print("\nJudge performance treating human labels as ground truth:")
    print(classification_report(human, judge, target_names=["supported", "hallucinated"]))
    if kappa < 0.60:
        print("\n[!] kappa below 0.60 — tighten the rubric / judge prompt or change the "
              "judge model, then re-run. Do NOT run the full benchmark yet.")
    else:
        print("\n[OK] Judge validated. Safe to run eval/run_benchmark.py.")


def main() -> None:
    if not HUMAN_LABELS.exists() or not JUDGE_LABELS.exists():
        print("Missing label files. Expected:")
        print(f"  {HUMAN_LABELS}  (you fill this in by hand)")
        print(f"  {JUDGE_LABELS}  (produced by running the judge on the subset)")
        print("\nStep 1: generate subset diagnoses, have 1-2 humans label them per the "
              "rubric, then run the judge on the same claims.")
        sys.exit(1)

    h = pd.read_csv(HUMAN_LABELS).sort_values("claim_id")
    j = pd.read_csv(JUDGE_LABELS).sort_values("claim_id")
    merged = h.merge(j, on="claim_id")
    compute_agreement(merged["human_label"].tolist(), merged["judge_label"].tolist())


if __name__ == "__main__":
    main()
