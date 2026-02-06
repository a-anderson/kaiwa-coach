"""Punctuation and pause normalization for TTS."""

from __future__ import annotations

import re


_JP_SENTENCE_END = re.compile(r"([。！？])")
_MULTI_PUNCT = re.compile(r"([。！？!?.])\1+")


def normalize_sentence_breaks(text: str) -> str:
    """Insert pauses after Japanese sentence endings.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Text with pauses inserted.
    """
    return _JP_SENTENCE_END.sub(r"\1 ", text)


def normalize_repeated_punctuation(text: str) -> str:
    """Collapse repeated punctuation.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Normalized text.
    """
    return _MULTI_PUNCT.sub(r"\1", text)


def normalize_for_tts(text: str) -> str:
    """Normalize punctuation for TTS stability.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Normalized text.
    """
    out = normalize_repeated_punctuation(text)
    out = normalize_sentence_breaks(out)
    return out.strip()
