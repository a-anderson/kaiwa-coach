#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Ensuring required Python deps are installed via Poetry..."
poetry install

echo "[setup] Ensuring phonemizer-fork is used (misaki expects it)..."
poetry run pip uninstall -y phonemizer || true
poetry run pip install -U phonemizer-fork

echo "[setup] Installing Japanese tokeniser dependencies..."
poetry run pip install -U "misaki[ja]" fugashi unidic
poetry run python -m unidic download

echo "[setup] Done."
