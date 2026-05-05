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
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

if [[ "$LLM_BACKEND" != "mlx" && "$LLM_BACKEND" != "ollama" ]]; then
    echo "Error: --llm-backend must be 'mlx' or 'ollama'" >&2
    exit 1
fi

echo "[setup] Installing Python dependencies..."
poetry install

echo "[setup] Installing Japanese tokeniser dependencies..."
poetry run python -m unidic download

PREFETCH_ARGS=("--backend" "$LLM_BACKEND")
if [[ -n "$MODEL_ARG" ]]; then
    PREFETCH_ARGS+=("--model" "$MODEL_ARG")
fi

echo "[setup] Prefetching models (this may take some time)..."
poetry run python scripts/prefetch_models.py "${PREFETCH_ARGS[@]}"

echo "[setup] Done."
