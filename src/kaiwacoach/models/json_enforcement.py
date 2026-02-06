"""JSON extraction and schema enforcement for LLM outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type

from pydantic import BaseModel, Field, StrictStr, ValidationError, conlist


class ConversationReply(BaseModel):
    reply: StrictStr


class ErrorDetection(BaseModel):
    errors: conlist(StrictStr, min_length=0)


class CorrectedSentence(BaseModel):
    corrected: StrictStr


class NativeReformulation(BaseModel):
    native: StrictStr


class Explanation(BaseModel):
    explanation: StrictStr


class JpTtsNormalisation(BaseModel):
    text: StrictStr


ROLE_SCHEMAS: dict[str, Type[BaseModel]] = {
    "conversation": ConversationReply,
    "error_detection": ErrorDetection,
    "correction": CorrectedSentence,
    "native_reformulation": NativeReformulation,
    "explanation": Explanation,
    "jp_tts_normalisation": JpTtsNormalisation,
}


@dataclass(frozen=True)
class ParseResult:
    model: Optional[BaseModel]
    raw_json: Optional[Dict[str, Any]]
    error: Optional[str]
    repaired: bool


def extract_first_json_object(text: str) -> Dict[str, Any]:
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
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text)
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
        model = schema.parse_obj(raw_json)
        return ParseResult(model=model, raw_json=raw_json, error=None, repaired=False)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        if repair_fn is None:
            return ParseResult(model=None, raw_json=None, error=str(exc), repaired=False)

    repaired_text = repair_fn(text)
    try:
        raw_json = extract_first_json_object(repaired_text)
        model = schema.parse_obj(raw_json)
        return ParseResult(model=model, raw_json=raw_json, error=None, repaired=True)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        return ParseResult(model=None, raw_json=None, error=str(exc), repaired=True)
