#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Ensuring required Python deps are installed via Poetry..."
poetry install --sync

echo "[setup] Installing Japanese tokeniser dependencies..."
poetry run python -m unidic download

echo "[setup] Prefetching ML models (these will take some time)..."
poetry run python scripts/prefetch_models.py

echo "[setup] Done."
