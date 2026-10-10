"""Profile the real EPDE evolution; diagnostic measurements, not accuracy.

Every call uses full PIC 16x5 budget. Runs on fresh GitHub Actions VMs;
profiles and the raw EPDE result are archived, not interpreted as a win.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import platform
import sys
import time
from pathlib import Path


def profile_search(*, dataset: str, variant: str, seed: int,
                   output_directory: Path) -> dict:
    from .epde_adapter import run_nir1_epde

    output_directory.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    try:
        record = run_nir1_epde(dataset, variant, seed)
    finally:
        profiler.disable()
    elapsed = time.perf_counter() - start
    stats = pstats.Stats(profiler)
    profiler.dump_stats(str(output_directory / "profile.pstats"))
    functions = []
    for (filename, lineno, func_name), (primitive_calls, total_calls,
                                        inner_seconds, cumulative_seconds, callers) in stats.stats.items():
        functions.append({
            "source": str(filename), "line": int(lineno),
            "name": str(func_name), "primitive_calls": int(primitive_calls),
            "total_calls": int(total_calls),
            "self_seconds": float(inner_seconds),
            "cumulative_seconds": float(cumulative_seconds),
        })
    by_cumulative = sorted(functions, key=lambda item: item["cumulative_seconds"],
                           reverse=True)[:80]
    by_self = sorted(functions, key=lambda item: item["self_seconds"],
                     reverse=True)[:80]
    meta = {"dataset": dataset, "variant": variant, "optimizer_seed": seed,
            "status": record.get("status"), "search_seconds": record.get("fit_seconds"),
            "profiled_wall_seconds": elapsed,
            "python": platform.python_version(), "total_function_count": len(functions),
            "top_cumulative": by_cumulative, "top_self": by_self,
            "note": "cProfile changes wall time; use separate unprofiled matched pairs for speed claims"}
    for file_name, body in (("profile.json", meta), ("raw-epde.json", record)):
        out = output_directory / file_name
        temp = out.with_suffix(out.suffix + ".part")
        temp.write_text(json.dumps(body, indent=2, sort_keys=True,
                                   allow_nan=False, ensure_ascii=False)+"\n",
                        encoding="utf-8")
        temp.replace(out)
    if record.get("status") != "ok":
        raise RuntimeError(f"EPDE profile completed with non-ok status: {record.get('status')}")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = profile_search(dataset=args.dataset, variant=args.variant,
                            seed=args.seed, output_directory=args.output)
    print(json.dumps({"status": report["status"], "dataset": report["dataset"],
                      "variant": report["variant"], "top_self": [
                          (x["name"], x["self_seconds"])
                          for x in report["top_self"][:8]]}))


if __name__ == "__main__":
    main()
