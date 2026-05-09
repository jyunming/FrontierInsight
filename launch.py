"""Frontier Insight — pipeline entry point.

This is a stub launcher for the end-to-end automated research pipeline.
The actual pipeline orchestration will be implemented in :mod:`core` and the
specialized agents in :mod:`agents` as the project matures.

Example
-------
::

    python launch.py --topic "High-NA EUV stochastic effects in photoresist modeling" \
        --output ./outputs/my-first-run
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="frontier-insight",
        description="End-to-end automated research pipeline for scientific discovery.",
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Research topic or question to drive the pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./outputs/run"),
        help="Directory where artifacts (logs, figures, manuscripts) will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # TODO: wire up the multi-agent pipeline:
    #   1. Idea generation
    #   2. Literature & data cross-referencing
    #   3. Experiment design & execution
    #   4. Analysis & insight synthesis
    #   5. Paper / poster / presentation generation
    #   6. Follow-up research suggestions
    print(f"[frontier-insight] Topic: {args.topic}")
    print(f"[frontier-insight] Output directory: {args.output.resolve()}")
    print("[frontier-insight] Pipeline scaffolding only — implementation coming soon.")


if __name__ == "__main__":
    main()
