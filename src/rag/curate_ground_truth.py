"""Curate data/ground_truth.csv (100 cases) from data/ground_truth_pool.csv (500 rows).

Deterministic (fixed seed) so the benchmark case set is reproducible for the thesis.

Design (proposal §3 sampling frame):

  80 x case_type=ambiguous   — visually confusable foliar diseases
  20 x case_type=cross_domain — image of crop X + query about crop Y, true_label=REJECT
                                => 20% adversarial fraction, matching cfg.adversarial_fraction

AMBIGUOUS allocation weights the confusion the thesis is actually about. The core
tomato triad (Early Blight / Septoria Leaf Spot / Late Blight) takes 40 of the 80 slots
because those three are the canonical misdiagnosis set: all three produce dark brown
foliar lesions on lower leaves, and they are separated only by fine-grained cues
(concentric "target" rings + chlorotic halo for Early Blight; small circular tan
lesions with dark margins and black pycnidia for Septoria; irregular greasy
water-soaked patches with pale sporulating margins for Late Blight). Target Spot and
Bacterial Spot add 20 more small-lesion confusables. Leaf Mold (6) is a distinct
upper/lower-surface pattern that is still confused with early-stage spotting. Potato
Early/Late Blight (14) replicate the same Early-vs-Late confusion on a second crop,
which lets us check that any effect is not tomato-specific.

CROSS-DOMAIN uses two directions on purpose, because they trap different failure modes:

  Direction A (15 cases) — off-crop IMAGE (corn / grape) + query naming an IN-KB crop
    (tomato / potato). This is the harder, more informative trap: the retriever WILL
    return confident, on-topic tomato/potato passages (the KB is tomato/potato only),
    so the textual modality actively encourages a plausible-sounding diagnosis that
    the image cannot support. This is precisely the modality-misalignment
    hallucination the Critic is designed to catch.

  Direction B (5 cases) — in-KB IMAGE (tomato / potato) + query naming a crop ABSENT
    from the KB (corn / grape / wheat / rice). Here retrieval is unhelpful, so this
    tests plain out-of-scope refusal rather than modality conflict.

Every cross_domain query names its crop explicitly, because actor_system.txt rule 4
triggers REJECT on "image shows a different crop than the QUERY asks about" — an
implicit query would make the gate untestable.

NOTE: eval/run_benchmark.py::_default_query and eval/make_validation_subset.py::_query_for
both use the `notes` column VERBATIM as the query for cross_domain rows. So for those
rows `notes` must be the farmer's question itself, not a description of it. For
ambiguous rows `notes` is documentation only (the query is generated from `crop`).

Run:  python -m src.rag.curate_ground_truth
"""
from __future__ import annotations

import csv
import random

from config.config import cfg

SEED = 20260920

# --- 80 ambiguous cases: (pool true_label, pool crop, n, confusion rationale) ---
_AMBIGUOUS_PLAN: list[tuple[str, str, int, str]] = [
    ("Early Blight", "tomato", 14,
     "core triad: concentric target rings + chlorotic halo; confused with Septoria/Late Blight"),
    ("Septoria Leaf Spot", "tomato", 14,
     "core triad: small circular tan lesions, dark margins, black pycnidia; confused with Early Blight"),
    ("Late Blight", "tomato", 12,
     "core triad: irregular greasy water-soaked lesions, pale sporulating margin; confused with Early Blight"),
    ("Target Spot", "tomato", 10,
     "small lesions with concentric zonation; easily read as Early Blight"),
    ("Bacterial Spot", "tomato", 10,
     "small dark water-soaked spots; easily read as Septoria Leaf Spot"),
    ("Leaf Mold", "tomato", 6,
     "pale upper-surface patches + olive lower-surface sporulation; confused with early-stage spotting"),
    ("Early Blight", "potato", 7,
     "cross-crop replication of the Early-vs-Late Blight confusion"),
    ("Late Blight", "potato", 7,
     "cross-crop replication of the Early-vs-Late Blight confusion"),
]

# --- Direction A: off-crop image + query about an IN-KB crop (retrieval is a trap) ---
_CROSS_A: list[tuple[str, str, int, str]] = [
    ("Common Rust", "corn", 4,
     "My tomato plants have these lesions spreading on the lower leaves. "
     "Is this tomato early blight or septoria leaf spot, and what should I spray?"),
    ("Common Rust", "corn", 4,
     "What disease affects this potato crop? I am worried it is potato late blight "
     "and I need to decide about fungicide today."),
    ("Black Rot", "grape", 4,
     "Please diagnose the disease on my tomato plant and tell me the treatment for it."),
    ("Black Rot", "grape", 3,
     "My potatoes look diseased. Is this potato early blight? What fungicide do you recommend?"),
]

# --- Direction B: in-KB image + query about a crop ABSENT from the KB ---
_CROSS_B: list[tuple[str, str, int, str]] = [
    ("Early Blight", "tomato", 2,
     "My corn has these pustules on the leaves. Is this corn common rust, and how do I treat it?"),
    ("Septoria Leaf Spot", "tomato", 1,
     "What disease is affecting my wheat crop here, and what is the recommended fungicide?"),
    ("Late Blight", "potato", 1,
     "Is this grape black rot on my grapevine? Tell me how to manage it."),
    ("Early Blight", "potato", 1,
     "My rice paddy looks damaged. What rice disease is this and what should I apply?"),
]


def _load_pool() -> list[dict]:
    with (cfg.root / "data" / "ground_truth_pool.csv").open() as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]
    return rows


def _bucket(pool: list[dict], label: str, crop: str) -> list[dict]:
    return sorted((r for r in pool if r["true_label"] == label and r["crop"] == crop),
                  key=lambda r: r["image_filename"])


def curate() -> None:
    pool = _load_pool()
    rng = random.Random(SEED)
    used: set[str] = set()
    out: list[dict] = []

    def take(label: str, crop: str, n: int) -> list[dict]:
        avail = [r for r in _bucket(pool, label, crop) if r["image_filename"] not in used]
        if len(avail) < n:
            raise SystemExit(f"pool exhausted for {crop}/{label}: need {n}, have {len(avail)}")
        picked = rng.sample(avail, n)
        used.update(r["image_filename"] for r in picked)
        return picked

    # 80 ambiguous
    for label, crop, n, rationale in _AMBIGUOUS_PLAN:
        for r in take(label, crop, n):
            out.append({"image_filename": r["image_filename"], "true_label": label,
                        "crop": crop, "case_type": "ambiguous", "notes": rationale})

    # 20 cross-domain (notes == the literal query sent to the systems)
    for plan in (_CROSS_A, _CROSS_B):
        for label, crop, n, query in plan:
            for r in take(label, crop, n):
                out.append({"image_filename": r["image_filename"], "true_label": "REJECT",
                            "crop": crop, "case_type": "cross_domain", "notes": query})

    n_amb = sum(1 for r in out if r["case_type"] == "ambiguous")
    n_x = sum(1 for r in out if r["case_type"] == "cross_domain")
    assert len(out) == 100, len(out)
    assert n_amb == 80 and n_x == 20, (n_amb, n_x)

    with cfg.ground_truth_csv.open("w", newline="") as f:
        f.write("# Curated benchmark case set — 100 cases. Generated deterministically by\n")
        f.write(f"# src/rag/curate_ground_truth.py (SEED={SEED}). Do not hand-edit; re-run instead.\n")
        f.write("# case_type=ambiguous (80): visually confusable foliar diseases.\n")
        f.write("# case_type=cross_domain (20): image of crop X + query about crop Y,\n")
        f.write("#   true_label=REJECT. For THESE rows `notes` is the literal query text\n")
        f.write("#   (run_benchmark._default_query uses it verbatim). 20/100 = 20% adversarial.\n")
        w = csv.DictWriter(f, fieldnames=["image_filename", "true_label", "crop",
                                          "case_type", "notes"])
        w.writeheader()
        w.writerows(out)

    print(f"Wrote {len(out)} cases to {cfg.ground_truth_csv}")
    print(f"  ambiguous:    {n_amb}")
    print(f"  cross_domain: {n_x}  ({n_x/len(out):.0%} adversarial)")


if __name__ == "__main__":
    curate()
