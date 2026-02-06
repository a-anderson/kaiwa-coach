"""Tests for protected span masking and restoration."""

from __future__ import annotations

from kaiwacoach.textnorm.protected_spans import mask_protected_spans, restore_protected_spans


def test_mask_and_restore_roundtrip() -> None:
    text = (
        "Email me at test@example.com and see https://example.com. "
        "Here is `code` and a file path /Users/test/file.txt."
    )
    masked = mask_protected_spans(text)
    restored = restore_protected_spans(masked.text, masked.spans)

    assert restored == text


def test_markdown_link_masking() -> None:
    text = "Check this [link](https://example.com) now."
    masked = mask_protected_spans(text)
    assert "https://example.com" not in masked.text
    restored = restore_protected_spans(masked.text, masked.spans)
    assert restored == text
