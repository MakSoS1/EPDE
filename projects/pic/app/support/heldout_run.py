"""Run chronological discovery in a process isolated from Streamlit sessions."""
import sys
from pathlib import Path

PIC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PIC), str(PIC.parent.parent)]

from epde_bench import env

env.pin_blas_threads(1)

if __name__ == '__main__':
    from epde_bench.heldout import run_heldout
    record = run_heldout('ode', train_fraction=.7, seed=0, output_dir=sys.argv[1])
    raise SystemExit(0 if record['status'] == 'ok' else 1)
