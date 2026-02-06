"""Prompt loading and rendering utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PromptRenderResult:
    text: str
    sha256: str
    path: Path


class PromptLoader:
    """Load markdown prompts and render with variables."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir)

    def render(self, name: str, variables: Mapping[str, Any]) -> PromptRenderResult:
        """Render a prompt by filename with variables.

        Parameters
        ----------
        name : str
            Prompt filename (e.g., "conversation.md").
        variables : Mapping[str, Any]
            Variables to substitute using `{var}` placeholders.

        Returns
        -------
        PromptRenderResult
            Rendered text, SHA256 hash, and source path.
        """
        path = self._root_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")

        template = path.read_text(encoding="utf-8")
        try:
            rendered = template.format_map(_StrictDict(variables))
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(f"Missing prompt variable: {missing}") from exc

        sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        return PromptRenderResult(text=rendered, sha256=sha256, path=path)


class _StrictDict(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover - raised in format_map
        raise KeyError(key)
