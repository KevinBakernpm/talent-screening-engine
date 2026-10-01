"""Command-line entry point: train every model on synthetic data and report.

Usage::

    python run_pipeline.py [--seed 2026]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from talent_screening.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate all screening models.")
    parser.add_argument("--seed", type=int, default=2026, help="synthetic data seed")
    args = parser.parse_args()
    run(seed=args.seed)


if __name__ == "__main__":
    main()
