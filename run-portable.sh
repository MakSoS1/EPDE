#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT="$ROOT/projects/pic"
BUNDLE="$ROOT/experiment.zip"
if [[ ! -f "$BUNDLE" ]]; then echo "Copy the downloaded experiment ZIP to $BUNDLE first." >&2; exit 1; fi
if command -v uv >/dev/null 2>&1; then UV="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then UV="$HOME/.local/bin/uv"
else
  echo "Installing official uv package manager..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="$HOME/.local/bin/uv"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
"$UV" run --project "$PROJECT" --python 3.12 --locked --extra cpu --extra app \
  python "$PROJECT/portable_experiment.py" --bundle "$BUNDLE"
echo "Upload $ROOT/experiment-results.zip on the website's Results page."
