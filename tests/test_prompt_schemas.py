"""Schema validation tests for LLM role output schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaiwacoach.models.json_enforcement import (
    ConversationSummaryResult,
    MonologueSummary,
    ROLE_SCHEMAS,
)


# ── ConversationSummaryResult ─────────────────────────────────────────────────


def test_conversation_summary_result_validates_correct_payload() -> None:
    model = ConversationSummaryResult.model_validate({
        "top_error_patterns": ["Particle errors", "Verb conjugation"],
        "priority_areas": ["Particles", "Verb forms"],
        "overall_notes": "Good progress overall.",
    })
    assert model.top_error_patterns == ["Particle errors", "Verb conjugation"]
    assert model.priority_areas == ["Particles", "Verb forms"]
    assert model.overall_notes == "Good progress overall."


def test_conversation_summary_result_rejects_empty_top_error_patterns() -> None:
    with pytest.raises(ValidationError):
        ConversationSummaryResult.model_validate({
            "top_error_patterns": [],
            "priority_areas": ["Particles"],
            "overall_notes": "Notes.",
        })


def test_conversation_summary_result_rejects_empty_priority_areas() -> None:
    with pytest.raises(ValidationError):
        ConversationSummaryResult.model_validate({
            "top_error_patterns": ["Error"],
            "priority_areas": [],
            "overall_notes": "Notes.",
        })


def test_conversation_summary_result_rejects_missing_overall_notes() -> None:
    with pytest.raises(ValidationError):
        ConversationSummaryResult.model_validate({
            "top_error_patterns": ["Error"],
            "priority_areas": ["Area"],
        })


def test_conversation_summary_result_registered_in_role_schemas() -> None:
    assert "summarise_conversation" in ROLE_SCHEMAS
    assert ROLE_SCHEMAS["summarise_conversation"] is ConversationSummaryResult


# ── MonologueSummary ──────────────────────────────────────────────────────────


def test_monologue_summary_validates_correct_payload() -> None:
    model = MonologueSummary.model_validate({
        "improvement_areas": ["Particle usage"],
        "overall_assessment": "Keep it up.",
    })
    assert model.improvement_areas == ["Particle usage"]
    assert model.overall_assessment == "Keep it up."


def test_monologue_summary_rejects_empty_improvement_areas() -> None:
    with pytest.raises(ValidationError):
        MonologueSummary.model_validate({
            "improvement_areas": [],
            "overall_assessment": "Good.",
        })


def test_monologue_summary_registered_in_role_schemas() -> None:
    assert "monologue_summary" in ROLE_SCHEMAS
    assert ROLE_SCHEMAS["monologue_summary"] is MonologueSummary
