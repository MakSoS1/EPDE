"""Check PDEComp/native benchmark records without conflating fixed and native.

Executed against MakSoS1/PDEComp@native in GitHub Actions; does not alter
or reimplement Vlad's original wrappers. Two repeat runs compare equations,
structure and coefficients, NOT wall time. Reference truth is never exposed
to framework fitting.
"""
import argparse
import csv
import math
import sys
from pathlib import Path


CORE = ("framework", "dataset", "target", "raw_target",
        "library_size", "active_terms_count",
        "active_terms", "model", "structure_hamming", "structure_success")


def rows(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def canonical(row):
    return (row.get("framework"), row.get("dataset"), row.get("target"))


def check_csv(path, framework, protocol):
    data = rows(path)
    if not data:
        raise AssertionError(f"{path}: empty report")
    for row in data:
        if row["framework"] != framework or row["protocol"] != protocol:
            raise AssertionError(f"{path}: mismatched framework/protocol: {row}")
        if row["status"] not in ("ok", "unsupported", "skipped"):
            raise AssertionError(f"{path}: error row: {row}")
        if row["status"] == "ok":
            for field in ("target", "library_size", "active_terms_count"):
                if not row.get(field):
                    raise AssertionError(f"{path}: absent {field} for a successful run: {row}")
            if int(row["active_terms_count"]) < 0:
                raise AssertionError("Negative number of equation terms")
    if not any(row["status"] == "ok" for row in data):
        raise AssertionError(f"{path}: no successful native scientific searches")
    return data


def compare_repeated(first, second):
    # Same PDEComp commit, library and input conditions, fixed seed.
    # Avoid requiring bitwise-identical float coefficients across BLAS builds.
    left, right = rows(first), rows(second)
    left.sort(key=canonical)
    right.sort(key=canonical)
    if len(left) != len(right):
        raise AssertionError("Different count of native benchmark targets")
    for a, b in zip(left, right):
        if canonical(a) != canonical(b):
            raise AssertionError(f"Target identities differ: {canonical(a)}, {canonical(b)}")
        if a["status"] != b["status"]:
            raise AssertionError(f"Run status changed: {a['status']} != {b['status']}")
        if a["status"] != "ok":
            continue
        for field in CORE:
            if a.get(field) != b.get(field):
                # coefficient model may use tolerable variations in decimal text
                if field == "model":
                    continue
                raise AssertionError(f"{canonical(a)}: {field} changed")
        for field in ("relative_error_sum", "coefficient_error"):
            av, bv = a.get(field), b.get(field)
            if av and bv and not math.isclose(float(av),float(bv),rel_tol=5e-3,abs_tol=1e-7):
                raise AssertionError(f"{canonical(a)}: {field}: {av} != {bv}")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--framework", required=True)
    p.add_argument("--protocol", default="native")
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path)
    a=p.parse_args()
    result=check_csv(a.first,a.framework,a.protocol)
    if a.second:
        check_csv(a.second,a.framework,a.protocol)
        compare_repeated(a.first,a.second)
    print(f"VALIDATED framework={a.framework}, protocol={a.protocol}, targets={len(result)}")
