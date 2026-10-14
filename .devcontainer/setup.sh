#!/usr/bin/env bash
# Codespaces first setup: locked CPU scientific environment from this checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
cd "$ROOT"
uv sync --project projects/pic --python 3.12 --locked --extra cpu --extra app
echo "EPDE Explorer dependencies installed. Streamlit starts on port 8501."
