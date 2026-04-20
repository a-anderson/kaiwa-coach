"""Tests for the /api/settings/profile routes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from kaiwacoach.api.server import create_app
from kaiwacoach.storage.blobs import SessionAudioCache


@pytest.fixture
def mock_orchestrator():
    orc = MagicMock()
    orc.language = "ja"
    orc.get_user_profile.return_value = {
        "user_name": None,
        "language_proficiency": {},
    }
    return orc


@pytest.fixture
def mock_audio_cache(tmp_path: Path):
    cache = MagicMock(spec=SessionAudioCache)
    cache.root_dir = tmp_path / "audio"
    cache.root_dir.mkdir()
    return cache


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.close = MagicMock()
    return db


@pytest.fixture
def client(mock_orchestrator, mock_audio_cache, mock_db):
    app = create_app(
        orchestrator=mock_orchestrator,
        audio_cache=mock_audio_cache,
        db=mock_db,
    )
    with TestClient(app) as c:
        yield c


def test_get_profile_returns_defaults(client, mock_orchestrator) -> None:
    res = client.get("/api/settings/profile")
    assert res.status_code == 200
    body = res.json()
    assert body["user_name"] is None
    assert body["language_proficiency"] == {}


def test_get_profile_reflects_set_value(client, mock_orchestrator) -> None:
    mock_orchestrator.get_user_profile.return_value = {
        "user_name": "Ashley",
        "language_proficiency": {"ja": "N3", "ja_kanji": "N1"},
    }
    res = client.get("/api/settings/profile")
    assert res.status_code == 200
    body = res.json()
    assert body["user_name"] == "Ashley"
    assert body["language_proficiency"]["ja"] == "N3"
    assert body["language_proficiency"]["ja_kanji"] == "N1"


def test_post_profile_calls_set_user_profile(client, mock_orchestrator) -> None:
    res = client.post(
        "/api/settings/profile",
        json={"user_name": "Ashley", "language_proficiency": {"ja": "N3", "fr": "B1"}},
    )
    assert res.status_code == 204
    mock_orchestrator.set_user_profile.assert_called_once_with(
        user_name="Ashley",
        language_proficiency={"ja": "N3", "fr": "B1"},
    )


def test_post_profile_updates_ja_and_ja_kanji_independently(client, mock_orchestrator) -> None:
    res = client.post(
        "/api/settings/profile",
        json={"language_proficiency": {"ja": "N2", "ja_kanji": "Native"}},
    )
    assert res.status_code == 204
    mock_orchestrator.set_user_profile.assert_called_once_with(
        user_name=None,
        language_proficiency={"ja": "N2", "ja_kanji": "Native"},
    )


def test_post_profile_invalid_level_returns_422(client, mock_orchestrator) -> None:
    res = client.post(
        "/api/settings/profile",
        json={"language_proficiency": {"ja": "fluent"}},
    )
    assert res.status_code == 422
    mock_orchestrator.set_user_profile.assert_not_called()


def test_post_profile_unknown_language_returns_422(client, mock_orchestrator) -> None:
    res = client.post(
        "/api/settings/profile",
        json={"language_proficiency": {"zh": "B1"}},
    )
    assert res.status_code == 422
    mock_orchestrator.set_user_profile.assert_not_called()


def test_post_profile_ja_kanji_native_succeeds(client, mock_orchestrator) -> None:
    res = client.post(
        "/api/settings/profile",
        json={"language_proficiency": {"ja_kanji": "Native"}},
    )
    assert res.status_code == 204


def test_post_profile_cefr_native_succeeds(client, mock_orchestrator) -> None:
    res = client.post(
        "/api/settings/profile",
        json={"language_proficiency": {"fr": "Native"}},
    )
    assert res.status_code == 204
