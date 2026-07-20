"""Data-prep helper: scaffold data/ground_truth.csv from a class-folder image dataset.

PlantVillage and the Tomato Leaf Dataset ship as one sub-folder per disease class:

    <dataset>/Tomato___Early_blight/xxxx.jpg
    <dataset>/Tomato___Septoria_leaf_spot/yyyy.jpg

This script copies a balanced sample of images into data/images/ and writes a
ground_truth.csv scaffold with a human-readable label derived from the folder name.
You then EDIT the csv by hand to (1) keep only visually-ambiguous cases, and
(2) add cross-domain rows (case_type=cross_domain, true_label=REJECT).

Run:  python cli.py prepare-data /path/to/dataset --per-class 20
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from config.config import cfg

_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _label_from_folder(folder: str) -> tuple[str, str]:
    """'Tomato___Early_blight' -> (crop='tomato', label='Early Blight')."""
    raw = folder.replace("___", "__").replace("_", " ").strip()
    parts = folder.split("___")
    if len(parts) == 2:
        crop = parts[0].replace("_", " ").strip().lower()
        label = parts[1].replace("_", " ").strip().title()
    else:
        crop = ""
        label = raw.title()
    return crop, label


def scaffold(dataset: Path, per_class: int) -> None:
    class_dirs = sorted(d for d in dataset.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class sub-folders under {dataset}.")

    cfg.images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for cdir in class_dirs:
        crop, label = _label_from_folder(cdir.name)
        images = sorted(p for p in cdir.iterdir() if p.suffix.lower() in _IMG_EXT)[:per_class]
        for i, src in enumerate(images):
            # Namespaced filename avoids collisions across classes.
            dest_name = f"{cdir.name.lower()}_{i:03d}{src.suffix.lower()}"
            shutil.copy2(src, cfg.images_dir / dest_name)
            rows.append({
                "image_filename": dest_name,
                "true_label": label,
                "crop": crop,
                "case_type": "ambiguous",
                "notes": f"auto from {cdir.name}",
            })

    out = cfg.ground_truth_csv
    with out.open("w", newline="") as f:
        f.write("# case_type: 'ambiguous' (visually similar diseases) or "
                "'cross_domain' (out-of-scope, RQ3)\n")
        f.write("# For cross_domain rows, set true_label to REJECT.\n")
        w = csv.DictWriter(f, fieldnames=["image_filename", "true_label", "crop",
                                          "case_type", "notes"])
        w.writeheader()
        w.writerows(rows)

    print(f"Copied {len(rows)} images into {cfg.images_dir}")
    print(f"Wrote scaffold {out} with {len(rows)} rows across {len(class_dirs)} classes.")
    print("\nNEXT (by hand):")
    print("  1. Keep the visually-ambiguous cases (e.g. Early Blight vs Septoria).")
    print(f"  2. Add cross-domain rows (~{cfg.adversarial_fraction:.0%}) with true_label=REJECT.")
    print("  3. Trim to ~100 total, then run `python cli.py check`.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaffold ground_truth.csv from a class-folder dataset.")
    ap.add_argument("dataset", type=Path, help="Path to dataset root (one sub-folder per class).")
    ap.add_argument("--per-class", type=int, default=20, help="Max images to sample per class.")
    args = ap.parse_args()
    scaffold(args.dataset, args.per_class)


if __name__ == "__main__":
    main()
