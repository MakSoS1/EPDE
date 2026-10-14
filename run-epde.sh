#!/usr/bin/env bash
# Run the full original EPDE Streamlit UI on this computer (macOS / Linux / WSL).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT="$ROOT/projects/pic"
if [[ ! -f "$PROJECT/uv.lock" || ! -f "$PROJECT/app/Home.py" ]]; then
  echo "EPDE project files are missing; launch this script from a complete checkout." >&2
  exit 1
fi
if [[ "${1:-}" == "--check" ]]; then
  echo "EPDE launcher paths OK: $PROJECT"
  exit 0
fi
if ! command -v uv >/dev/null 2>&1; then
  for possible in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [[ -x "$possible" ]]; then UV="$possible"; break; fi
  done
else
  UV="$(command -v uv)"
fi
if [[ -z "${UV:-}" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and retry." >&2
    exit 1
  fi
  echo "uv missing. Installing the official Python environment manager (first run only)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="$HOME/.local/bin/uv"
  if [[ ! -x "$UV" ]]; then
    echo "uv install did not produce $UV. See https://docs.astral.sh/uv/" >&2
    exit 1
  fi
fi
echo "Starting the FULL EPDE Explorer at http://127.0.0.1:8501"
echo "First run downloads Python 3.12, PyTorch, data dependencies; allow time and disk space."
echo "Heavy data are loaded only when requested. Press Ctrl+C to stop."
export EPDE_LOCAL_FULL_RUN=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec "$UV" run --project "$PROJECT" --python 3.12 --locked --extra cpu --extra app \
  streamlit run "$PROJECT/app/Home.py" --server.address=127.0.0.1 --server.port=8501 "$@"
