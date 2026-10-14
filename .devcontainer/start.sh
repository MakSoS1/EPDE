#!/usr/bin/env bash
# Codespaces port forwarding is private; heavy scientific jobs run in Codespaces.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PATH="$HOME/.local/bin:$PATH"
export EPDE_LOCAL_FULL_RUN=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
if [[ ! -x "$ROOT/projects/pic/.venv/bin/streamlit" ]]; then
  echo "EPDE environment not installed. Run bash .devcontainer/setup.sh" >&2
  exit 1
fi
# postStartCommand must finish quickly; serve UI in the background.
nohup "$ROOT/projects/pic/.venv/bin/streamlit" run "$ROOT/projects/pic/app/Home.py" \
  --server.headless=true --server.address=0.0.0.0 --server.port=8501 \
  --browser.gatherUsageStats=false > /tmp/epde-codespaces.log 2>&1 < /dev/null &
echo "EPDE Explorer starting on private forwarded port 8501 (PID $!)."
echo "Logs: /tmp/epde-codespaces.log"
