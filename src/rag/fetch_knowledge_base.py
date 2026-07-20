"""Download a starter knowledge base of authoritative crop-disease PDFs.

Sources are public university-extension and government fact sheets covering the diseases
in the benchmark — with emphasis on the core ambiguous pair (Early Blight vs. Septoria).
Each file is verified to be a real PDF (starts with %PDF) before being kept.

Run:  python cli.py fetch-kb
"""
from __future__ import annotations

from config.config import cfg

# (filename, url). Curated authoritative extension / government sources.
_SOURCES: list[tuple[str, str]] = [
    # --- Core ambiguous pair: Early Blight vs. Septoria Leaf Spot ---
    ("cornell_early_blight_vs_septoria_tomato.pdf",
     "https://www.maine.gov/dacf/php/gotpests/diseases/factsheets/early-blight-and-septoria-cornell.pdf"),
    ("kstate_tomato_early_blight_septoria.pdf",
     "https://hnr.k-state.edu/extension/horticulture-resource-center/common-pest-problems/documents/Tomato%20-%20Early%20Blight%20and%20Septoria%20Leaf%20Spot.pdf"),
    ("wisc_tomato_early_blight_septoria.pdf",
     "https://barron.extension.wisc.edu/files/2023/02/Tomato-Disorder-Early-Blight-and-Septoria-Leaf-Spot.pdf"),
    # --- Broad tomato disease guides ---
    ("msu_e3170_tomato_diseases.pdf",
     "https://www.canr.msu.edu/home_gardening/uploads/files/E3170_Tomato_Diseases_June-2025_AA.pdf"),
    ("ut_sp277w_foliar_diseases_of_tomato.pdf",
     "https://utia.tennessee.edu/publications/wp-content/uploads/sites/269/2023/10/SP277-W.pdf"),
    # --- Late blight (tomato & potato) ---
    ("wisc_tomato_late_blight.pdf",
     "https://barron.extension.wisc.edu/files/2023/02/Tomato-Late-Blight.pdf"),
    ("umass_late_blight_tomato_potato.pdf",
     "https://www.umass.edu/agriculture-food-environment/sites/ag.umass.edu/files/fact-sheets/pdf/late_blight_management.pdf"),
    ("purdue_bp80w_late_blight_tomato_potato.pdf",
     "https://www.extension.purdue.edu/extmedia/bp/bp-80-w.pdf"),
    # --- Potato ---
    ("wisc_potato_late_blight.pdf",
     "https://barron.extension.wisc.edu/files/2023/02/Potato-Late-Blight.pdf"),
    ("ndsu_potato_diseases_home_garden.pdf",
     "https://www.ag.ndsu.edu/potatoextension/Mgmtofpotatodiseasesinhomegarden.pdf"),
]

_UA = "Mozilla/5.0 (research knowledge-base fetcher; crop-reflection-rag)"


def fetch() -> None:
    import requests

    cfg.kb_dir.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    for fname, url in _SOURCES:
        dest = cfg.kb_dir / fname
        try:
            r = requests.get(url, headers={"User-Agent": _UA}, timeout=60)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError(f"not a PDF (starts with {r.content[:8]!r})")
            dest.write_bytes(r.content)
            ok.append((fname, len(r.content)))
            print(f"  [ok]   {fname}  ({len(r.content)//1024} KB)")
        except Exception as exc:  # noqa: BLE001
            failed.append((fname, str(exc)))
            print(f"  [FAIL] {fname}: {exc}")

    print(f"\n{len(ok)}/{len(_SOURCES)} PDFs saved to {cfg.kb_dir}")
    if failed:
        print("Failed (skip or replace the URL in src/rag/fetch_knowledge_base.py):")
        for fname, err in failed:
            print(f"  - {fname}: {err}")
    if ok:
        print("\nNEXT: python cli.py ingest   (build the Chroma index)")


if __name__ == "__main__":
    fetch()
