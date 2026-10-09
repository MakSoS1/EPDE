"""Execute the unchanged notebook demo in a process dedicated to one UI request."""
import json
from pathlib import Path
import sys

PIC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PIC), str(PIC.parent.parent)]

from epde_bench import env

env.pin_blas_threads(1)

if __name__ == '__main__':
    from epde_bench.optimizer_demo import run_optimizer_demo
    result = run_optimizer_demo()
    with Path(sys.argv[1]).open('w', encoding='utf-8') as stream:
        json.dump(result, stream, allow_nan=False)
