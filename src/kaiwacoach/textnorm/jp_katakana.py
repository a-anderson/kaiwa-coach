"""Katakana conversion for non-Japanese spans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from kaiwacoach.textnorm.protected_spans import mask_protected_spans, restore_protected_spans


_JP_CHAR_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class KatakanaResult:
    text: str
    meta: Dict[str, object]


def normalise_katakana(
    text: str,
    llm_rewrite_fn: Callable[[str], str],
) -> KatakanaResult:
    """Rewrite non-Japanese spans into katakana using an LLM.

    Parameters
    ----------
    text : str
        Input text to normalize.
    llm_rewrite_fn : Callable[[str], str]
        LLM rewrite function that accepts text and returns rewritten text.

    Returns
    -------
    KatakanaResult
        Normalized text and metadata.
    """
    masked = mask_protected_spans(text)
    rewritten = llm_rewrite_fn(masked.text)
    restored = restore_protected_spans(rewritten, masked.spans)
    meta = {
        "masked_spans": len(masked.spans),
    }
    return KatakanaResult(text=restored, meta=meta)


def contains_japanese(text: str) -> bool:
    """Return True if the text contains Japanese characters."""
    return _JP_CHAR_RE.search(text) is not None


def contains_latin(text: str) -> bool:
    """Return True if the text contains Latin characters."""
    return _LATIN_RE.search(text) is not None
