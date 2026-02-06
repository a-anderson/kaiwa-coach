"""Invariant checks for Japanese text preservation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Tuple


_JP_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]+")


@dataclass(frozen=True)
class InvariantResult:
    ok: bool
    mismatches: List[Tuple[str, int]]


def extract_japanese_spans(text: str) -> List[str]:
    """Extract contiguous Japanese substrings.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list[str]
        Japanese substrings in order of appearance.
    """
    return _JP_RE.findall(text)


def check_japanese_invariant(original: str, candidate: str) -> InvariantResult:
    """Verify Japanese substrings are preserved byte-identical.

    Parameters
    ----------
    original : str
        Original text.
    candidate : str
        Candidate text after normalization.

    Returns
    -------
    InvariantResult
        Result with mismatch details.
    """
    original_spans = extract_japanese_spans(original)
    candidate_spans = extract_japanese_spans(candidate)

    mismatches: List[Tuple[str, int]] = []
    for idx, span in enumerate(original_spans):
        if idx >= len(candidate_spans):
            mismatches.append((span, idx))
            continue
        if span != candidate_spans[idx]:
            mismatches.append((span, idx))

    ok = len(mismatches) == 0
    return InvariantResult(ok=ok, mismatches=mismatches)


def enforce_japanese_invariant(
    original: str,
    candidate: str,
    logger: Callable[[str], None] | None = None,
) -> Tuple[str, InvariantResult]:
    """Enforce the Japanese invariant with fallback and logging.

    Parameters
    ----------
    original : str
        Original text.
    candidate : str
        Candidate text after normalization.
    logger : Callable[[str], None] | None
        Optional logger callable. If provided, will be called on violation.

    Returns
    -------
    tuple[str, InvariantResult]
        Output text (candidate or original) and invariant result.
    """
    result = check_japanese_invariant(original, candidate)
    if result.ok:
        return candidate, result

    message = (
        f"Japanese invariant violated; falling back to original. "
        f"Mismatches: {result.mismatches}"
    )
    if logger is not None:
        logger(message)
    return original, result
