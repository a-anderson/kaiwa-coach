#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [--llm-backend mlx|ollama] [--model MODEL_ID]

Options:
  --llm-backend   LLM execution backend: 'mlx' (default) or 'ollama'
  --model         LLM model ID to prefetch (backend-specific)

Run 'poetry run python scripts/prefetch_models.py --help' for available model IDs.
EOF
}

die_usage() {
    echo "Error: $1" >&2
    usage >&2
    exit 1
}

LLM_BACKEND="mlx"
MODEL_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm-backend)
            LLM_BACKEND="${2:?'--llm-backend requires a value'}"
            shift 2
            ;;
        --model)
            MODEL_ARG="${2:?'--model requires a value'}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die_usage "Unknown argument: $1"
            ;;
    esac
done

if [[ "$LLM_BACKEND" != "mlx" && "$LLM_BACKEND" != "ollama" ]]; then
    die_usage "--llm-backend must be 'mlx' or 'ollama'"
fi

echo "[setup] Installing Python dependencies..."
poetry install --sync

echo "[setup] Installing Japanese tokeniser dependencies..."
poetry run python -m unidic download

PREFETCH_ARGS=("--backend" "$LLM_BACKEND")
if [[ -n "$MODEL_ARG" ]]; then
    PREFETCH_ARGS+=("--model" "$MODEL_ARG")
fi

echo "[setup] Prefetching models (this may take some time)..."
poetry run python scripts/prefetch_models.py "${PREFETCH_ARGS[@]}"

echo "[setup] Done."
