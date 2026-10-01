"""Shared fixtures: put ``src`` on the path and build one synthetic dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talent_screening.pipeline import load_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def tables():
    return load_synthetic(seed=7)
