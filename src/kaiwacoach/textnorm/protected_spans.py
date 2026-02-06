"""Mask and restore protected spans in text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


_URL_RE = re.compile(r"https?://[^\\s)]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")
_CODE_BLOCK_RE = re.compile(r"```[\\s\\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_MARKDOWN_LINK_RE = re.compile(r"\\[[^\\]]+\\]\\([^\\)]+\\)")
_FILE_PATH_RE = re.compile(
    r"(?:/[^\\s]+)+|(?:[A-Za-z]:\\\\[^\\s]+)|(?:~/(?:[^\\s]+)*)"
)


@dataclass(frozen=True)
class MaskResult:
    text: str
    spans: Dict[str, str]


def mask_protected_spans(text: str) -> MaskResult:
    """Mask URLs, file paths, emails, and code spans in text.

    Parameters
    ----------
    text : str
        Input text to mask.

    Returns
    -------
    MaskResult
        Masked text and the mapping of placeholders to original spans.
    """
    spans: Dict[str, str] = {}
    masked = text
    patterns = [
        ("code_block", _CODE_BLOCK_RE),
        ("inline_code", _INLINE_CODE_RE),
        ("markdown_link", _MARKDOWN_LINK_RE),
        ("url", _URL_RE),
        ("email", _EMAIL_RE),
        ("file_path", _FILE_PATH_RE),
    ]

    counter = 0
    for label, regex in patterns:
        for match in list(regex.finditer(masked)):
            placeholder = f"__{label.upper()}_{counter}__"
            spans[placeholder] = match.group(0)
            masked = masked.replace(match.group(0), placeholder, 1)
            counter += 1

    return MaskResult(text=masked, spans=spans)


def restore_protected_spans(text: str, spans: Dict[str, str]) -> str:
    """Restore masked spans in text.

    Parameters
    ----------
    text : str
        Masked text containing placeholders.
    spans : dict[str, str]
        Mapping of placeholders to original spans.

    Returns
    -------
    str
        Restored text.
    """
    restored = text
    for placeholder, original in spans.items():
        restored = restored.replace(placeholder, original)
    return restored
