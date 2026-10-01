"""Shared text normalisation helpers for CV and transcript processing."""

from __future__ import annotations

import re
from typing import Iterable, List

_WHITESPACE = re.compile(r"\s+")
_NON_TEXT = re.compile(r"[^a-z0-9/+#.\-\s']")
_TOKEN = re.compile(r"[a-z][a-z0-9/+#'\-]*")
_SENTENCE = re.compile(r"[.!?]+")


def normalise(text: str) -> str:
    """Lower-case, strip unsupported characters and collapse whitespace."""
    text = _NON_TEXT.sub(" ", str(text).lower())
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    """Split normalised text into word tokens."""
    return _TOKEN.findall(normalise(text))


def sentences(text: str) -> List[str]:
    """Split text into non-empty sentences."""
    return [s.strip() for s in _SENTENCE.split(str(text)) if s.strip()]


def count_phrases(text: str, phrases: Iterable[str]) -> int:
    """Count occurrences of any of ``phrases`` (whole-word, case-insensitive)."""
    norm = f" {normalise(text)} "
    return sum(norm.count(f" {p} ") for p in phrases)
