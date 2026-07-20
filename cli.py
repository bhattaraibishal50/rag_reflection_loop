"""Single entry point for the whole project — run every pipeline step from here.

    python cli.py <command> [args]

Commands (in the order you'd normally run them):

    fetch-images             download a labelled PlantVillage image pool from Hugging Face
    fetch-kb                 download authoritative crop-disease PDFs into the knowledge base
    prepare-data <dataset>   scaffold data/ground_truth.csv from a local class-folder dataset
    ingest                   build the Chroma index from data/knowledge_base/*.pdf
    retrieve "<query>"       GATE: inspect retrieved passages
    check                    GATE: verify key + KB + index + images are ready
    diagnose <image>         run one diagnosis (--system a|b)
    make-validation-subset   run a subset + judge labels for judge validation
    validate-judge           GATE: Cohen's kappa of the LLM judge vs humans (>= 0.6)
    benchmark                run all cases through A and B; print stats
    plots                    generate Chapter 4 figures from the results
    demo                     launch the Streamlit comparison app

Imports are lazy per-command, so `python cli.py --help` and offline commands never
require an API key or a built index.
"""
from __future__ import annotations

import argparse
import sys


def _cmd_fetch_images(args: argparse.Namespace) -> None:
    from src.rag.fetch_plantvillage import fetch
    fetch(args.per_class)


def _cmd_fetch_kb(args: argparse.Namespace) -> None:
    from src.rag.fetch_knowledge_base import fetch
    fetch()


def _cmd_prepare_data(args: argparse.Namespace) -> None:
    from src.rag.prepare_ground_truth import scaffold
    from pathlib import Path
    scaffold(Path(args.dataset), args.per_class)


def _cmd_ingest(args: argparse.Namespace) -> None:
    from src.rag.ingest import main
    main()


def _cmd_retrieve(args: argparse.Namespace) -> None:
    from src.rag.retriever import Retriever
    for i, hit in enumerate(Retriever().search(args.query), 1):
        print(f"\n--- result {i} (dist={hit['distance']:.3f}, {hit['source']}) ---")
        print(hit["text"][:400])


def _cmd_check(args: argparse.Namespace) -> None:
    from eval.preflight import main
    main()  # calls sys.exit with the readiness code


def _cmd_diagnose(args: argparse.Namespace) -> None:
    if args.system == "a":
        from src.systems import baseline as system
    else:
        from src.systems.reflection import graph as system
    result = system.diagnose(args.image, args.query)
    print(result["diagnosis"])
    tail = f"latency: {result['latency_s']:.2f}s"
    if "iterations" in result:
        tail = f"iterations: {result['iterations']} · " + tail
    print(f"\n[{tail}]")


def _cmd_make_validation_subset(args: argparse.Namespace) -> None:
    from eval.make_validation_subset import make_subset
    make_subset(args.cases)


def _cmd_validate_judge(args: argparse.Namespace) -> None:
    from eval.judge_validation import main
    main()


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from eval.run_benchmark import run, summarize
    summarize(run())


def _cmd_plots(args: argparse.Namespace) -> None:
    from eval.plots import make_plots
    make_plots()


def _cmd_demo(args: argparse.Namespace) -> None:
    import subprocess
    subprocess.run(["streamlit", "run", "src/app/app.py"], check=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli.py", description="Crop Reflection RAG pipeline.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("fetch-images", help="download a labelled PlantVillage image pool from HF")
    sp.add_argument("--per-class", type=int, default=50, help="images per class (default 50)")
    sp.set_defaults(func=_cmd_fetch_images)

    sp = sub.add_parser("fetch-kb", help="download authoritative crop-disease PDFs into the KB")
    sp.set_defaults(func=_cmd_fetch_kb)

    sp = sub.add_parser("prepare-data", help="scaffold ground_truth.csv from a local dataset folder")
    sp.add_argument("dataset", help="path to dataset root (one sub-folder per class)")
    sp.add_argument("--per-class", type=int, default=20, help="max images sampled per class")
    sp.set_defaults(func=_cmd_prepare_data)

    sp = sub.add_parser("ingest", help="build the Chroma index from KB PDFs")
    sp.set_defaults(func=_cmd_ingest)

    sp = sub.add_parser("retrieve", help="GATE: inspect retrieved passages for a query")
    sp.add_argument("query", help="search query")
    sp.set_defaults(func=_cmd_retrieve)

    sp = sub.add_parser("check", help="GATE: verify the project is ready for a run")
    sp.set_defaults(func=_cmd_check)

    sp = sub.add_parser("diagnose", help="run one diagnosis on an image")
    sp.add_argument("image", help="path to a leaf image")
    sp.add_argument("--system", choices=["a", "b"], default="b",
                    help="a = single-pass baseline, b = reflection loop (default)")
    sp.add_argument("--query", default="What disease does this plant have?")
    sp.set_defaults(func=_cmd_diagnose)

    sp = sub.add_parser("make-validation-subset",
                        help="generate the subset + judge labels for judge validation")
    sp.add_argument("cases", nargs="?", type=int, default=13,
                    help="number of cases to sample (outputs = 2x this; default 13)")
    sp.set_defaults(func=_cmd_make_validation_subset)

    sp = sub.add_parser("validate-judge", help="GATE: Cohen's kappa of judge vs humans")
    sp.set_defaults(func=_cmd_validate_judge)

    sp = sub.add_parser("benchmark", help="run the full A-vs-B benchmark + stats")
    sp.set_defaults(func=_cmd_benchmark)

    sp = sub.add_parser("plots", help="generate Chapter 4 figures from benchmark results")
    sp.set_defaults(func=_cmd_plots)

    sp = sub.add_parser("demo", help="launch the Streamlit comparison app")
    sp.set_defaults(func=_cmd_demo)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
