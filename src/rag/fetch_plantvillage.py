"""Fetch a labelled image POOL from the DScomp380/plant_village HF dataset.

Parquet-backed and public — no Kaggle credentials needed. Streams the dataset and saves
up to `per_class` images for the classes we care about into data/images/, then writes a
POOL csv (data/ground_truth_pool.csv) to curate down into the real data/ground_truth.csv.

Focus: visually-ambiguous tomato/potato disease classes for the adversarial suite, plus a
couple of off-crop classes (corn, grape) as raw material for cross-domain (RQ3) cases.

Run:  python cli.py fetch-images --per-class 50
"""
from __future__ import annotations

import argparse
import csv

from config.config import cfg
from src.rag.prepare_ground_truth import _label_from_folder

_REPO = "DScomp380/plant_village"

# Exact label names as they appear in the dataset.
_AMBIGUOUS = [
    "Tomato___Early_blight", "Tomato___Septoria_leaf_spot", "Tomato___Late_blight",
    "Tomato___Bacterial_spot", "Tomato___Target_Spot", "Tomato___Leaf_Mold",
    "Potato___Early_blight", "Potato___Late_blight",
]
_OFF_CROP = ["Corn___Common_rust", "Grape___Black_rot"]  # material for cross-domain rows
_TARGET = _AMBIGUOUS + _OFF_CROP


def fetch(per_class: int = 50) -> None:
    from datasets import load_dataset

    ds = load_dataset(_REPO, split="train", streaming=True)
    names = ds.features["label"].names
    want = [n for n in _TARGET if n in names]
    missing = [n for n in _TARGET if n not in names]
    if missing:
        print(f"note: not found in dataset, skipping: {missing}")
    label_to_name = {names.index(n): n for n in want}
    counts = {n: 0 for n in want}

    cfg.images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    print(f"Streaming {_REPO}; collecting up to {per_class} images for {len(want)} classes...")
    for ex in ds:
        cls = label_to_name.get(ex["label"])
        if cls is None or counts[cls] >= per_class:
            continue
        crop, label = _label_from_folder(cls)
        fname = f"{cls.lower().replace(' ', '_')}_{counts[cls]:03d}.jpg"
        ex["image"].convert("RGB").save(cfg.images_dir / fname, "JPEG", quality=90)
        case_type = "ambiguous" if cls in _AMBIGUOUS else "off_crop"
        rows.append({"image_filename": fname, "true_label": label, "crop": crop,
                     "case_type": case_type, "notes": f"pool from {cls}"})
        counts[cls] += 1
        if all(c >= per_class for c in counts.values()):
            break

    pool = cfg.root / "data" / "ground_truth_pool.csv"
    with pool.open("w", newline="") as f:
        f.write("# RAW POOL — curate into data/ground_truth.csv. Keep ambiguous pairs;\n")
        f.write("# turn some off_crop rows into cross_domain (true_label=REJECT, off-crop query).\n")
        w = csv.DictWriter(f, fieldnames=["image_filename", "true_label", "crop",
                                          "case_type", "notes"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved {len(rows)} images to {cfg.images_dir}")
    for c in sorted(counts):
        print(f"  {counts[c]:>4}  {c}")
    print(f"\nPool CSV: {pool}")
    print("NEXT: curate ~100 rows into data/ground_truth.csv (keep ambiguous pairs; add")
    print("      cross_domain rows: off-crop image + true_label=REJECT). Then: python cli.py check")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a labelled PlantVillage image pool from HF.")
    ap.add_argument("--per-class", type=int, default=50, help="images per class (default 50)")
    ap.add_argument("--repo", default=_REPO, help="override the HF dataset id")
    args = ap.parse_args()
    if args.repo != _REPO:
        globals()["_REPO"] = args.repo
    fetch(args.per_class)


if __name__ == "__main__":
    main()
