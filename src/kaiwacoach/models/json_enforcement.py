"""JSON extraction and schema enforcement for LLM outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Type

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Gemma 4 26B-A4B (MLX) generates thought blocks in this format before the answer.
# Requires the closing <channel|> tag — truncated output (no closing tag) will not be
# stripped and will cause raw_decode to fail. For Ollama, use suppress_thinking=True
# on OllamaBackend instead; this regex handles the MLX path only.
_GEMMA_CHANNEL_RE = re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL)

from pydantic import BaseModel, Field, StrictStr, ValidationError, conlist


class ConversationReply(BaseModel):
    reply: StrictStr


class JpTtsNormalisation(BaseModel):
    text: StrictStr


class DetectAndCorrect(BaseModel):
    errors: conlist(StrictStr, min_length=0)
    corrected: StrictStr


class ExplainAndNative(BaseModel):
    explanation: StrictStr
    native: StrictStr


class MonologueSummary(BaseModel):
    improvement_areas: conlist(StrictStr, min_length=1)
    overall_assessment: StrictStr


ROLE_SCHEMAS: dict[str, Type[BaseModel]] = {
    "conversation": ConversationReply,
    "jp_tts_normalisation": JpTtsNormalisation,
    "detect_and_correct": DetectAndCorrect,
    "explain_and_native": ExplainAndNative,
    "monologue_summary": MonologueSummary,
}


@dataclass(frozen=True)
class ParseResult:
    model: BaseModel | None
    raw_json: dict[str, Any] | None
    error: str | None
    repaired: bool


def extract_first_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a string.

    Parameters
    ----------
    text : str
        Raw LLM output that may contain extra content.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.

    Raises
    ------
    json.JSONDecodeError
        If no valid JSON object can be found.
    ValueError
        If the parsed value is not a JSON object.
    """
    cleaned = _THINK_TAG_RE.sub("", text)
    cleaned = _GEMMA_CHANNEL_RE.sub("", cleaned).strip()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(cleaned)
    if not isinstance(obj, dict):
        raise ValueError("First JSON value is not an object.")
    return obj


def parse_with_schema(
    role: str,
    text: str,
    repair_fn: Callable[[str], str] | None = None,
) -> ParseResult:
    """Parse and validate LLM JSON output with one optional repair attempt.

    Parameters
    ----------
    role : str
        Role name for schema selection.
    text : str
        Raw LLM output.
    repair_fn : Callable[[str], str] | None
        Optional repair function that receives the raw output and returns a new string.

    Returns
    -------
    ParseResult
        Parsed model or error details.
    """
    schema = ROLE_SCHEMAS.get(role)
    if schema is None:
        return ParseResult(model=None, raw_json=None, error=f"Unknown role: {role}", repaired=False)

    try:
        raw_json = extract_first_json_object(text)
        model = schema.model_validate(raw_json)
        return ParseResult(model=model, raw_json=raw_json, error=None, repaired=False)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        if repair_fn is None:
            return ParseResult(model=None, raw_json=None, error=str(exc), repaired=False)

    repaired_text = repair_fn(text)
    try:
        raw_json = extract_first_json_object(repaired_text)
        model = schema.model_validate(raw_json)
        return ParseResult(model=model, raw_json=raw_json, error=None, repaired=True)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        return ParseResult(model=None, raw_json=None, error=str(exc), repaired=True)
