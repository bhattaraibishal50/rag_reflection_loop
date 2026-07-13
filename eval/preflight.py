"""Preflight readiness check — run BEFORE spending API budget on a full benchmark.

Verifies, without making any paid API call, that the pieces a real run needs are in
place: API key, knowledge-base PDFs, a built Chroma index, images, and a ground-truth
file whose rows line up with the images on disk.

Run:  python -m eval.preflight
Exit code 0 = ready; 1 = something is missing.
"""
from __future__ import annotations

import sys

import pandas as pd

from config.config import cfg

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


def _check_api_key() -> bool:
    if cfg.api_key:
        print(f"{OK} GEMINI_API_KEY is set.")
        return True
    print(f"{FAIL} GEMINI_API_KEY not set — copy .env.example to .env and fill it in.")
    return False


def _check_kb() -> bool:
    pdfs = list(cfg.kb_dir.glob("*.pdf"))
    if pdfs:
        print(f"{OK} Knowledge base: {len(pdfs)} PDF(s) in {cfg.kb_dir}.")
        return True
    print(f"{FAIL} No PDFs in {cfg.kb_dir} — add FAO/agronomy handbooks before ingest.")
    return False


def _check_index() -> bool:
    if not cfg.chroma_dir.exists():
        print(f"{WARN} No Chroma index yet at {cfg.chroma_dir} — run `python -m src.ingest.ingest`.")
        return False
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
        count = client.get_collection(cfg.collection_name).count()
        if count > 0:
            print(f"{OK} Chroma index: {count} chunks in '{cfg.collection_name}'.")
            return True
        print(f"{FAIL} Chroma collection '{cfg.collection_name}' is empty — re-run ingest.")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} Could not open Chroma index: {exc}")
        return False


def _check_images_and_ground_truth() -> bool:
    if not cfg.ground_truth_csv.exists():
        print(f"{FAIL} Missing {cfg.ground_truth_csv}.")
        return False
    gt = pd.read_csv(cfg.ground_truth_csv, comment="#")
    required = {"image_filename", "true_label", "crop", "case_type"}
    missing_cols = required - set(gt.columns)
    if missing_cols:
        print(f"{FAIL} ground_truth.csv missing columns: {sorted(missing_cols)}")
        return False

    on_disk = {p.name for p in cfg.images_dir.glob("*") if p.suffix.lower() in
               {".jpg", ".jpeg", ".png"}}
    listed = set(gt["image_filename"])
    missing_files = sorted(listed - on_disk)
    orphan_files = sorted(on_disk - listed)

    n_cross = int((gt["case_type"] == "cross_domain").sum())
    frac = n_cross / len(gt) if len(gt) else 0.0

    ok = True
    print(f"{OK} ground_truth.csv: {len(gt)} rows "
          f"({n_cross} cross-domain = {frac:.0%}; target {cfg.adversarial_fraction:.0%}).")
    if missing_files:
        ok = False
        print(f"{FAIL} {len(missing_files)} row(s) reference images not on disk, e.g. "
              f"{missing_files[:3]}")
    if orphan_files:
        print(f"{WARN} {len(orphan_files)} image(s) on disk not in ground_truth.csv, e.g. "
              f"{orphan_files[:3]}")
    if len(gt) < 100:
        print(f"{WARN} Only {len(gt)} cases — proposal targets 100.")
    return ok


def main() -> None:
    print("=== Preflight: is the project ready for a real benchmark run? ===\n")
    checks = [
        _check_api_key(),
        _check_kb(),
        _check_index(),
        _check_images_and_ground_truth(),
    ]
    print()
    if all(checks):
        print("READY. Next: `python -m eval.judge_validation` then `python -m eval.run_benchmark`.")
        sys.exit(0)
    print("NOT READY — resolve the [FAIL] items above first.")
    sys.exit(1)


if __name__ == "__main__":
    main()
