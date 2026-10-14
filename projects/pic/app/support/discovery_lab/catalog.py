"""Complete registry-backed catalogue for the EPDE Explorer.

Never load numerical arrays merely to list entries: a full experimental
database includes large 2D/3D records and externally distributed files.
"""
from __future__ import annotations

import ast
import inspect

from epde_bench.datasets import REGISTRY
from epde_bench.identity import problem_metadata

KIND_LABELS = {
    "ode": "Ordinary differential equations",
    "ode_system": "ODE systems",
    "pde_1d": "PDE · 1 spatial dimension",
    "pde_2d": "PDE · 2 spatial dimensions",
    "pde_3d": "PDE · 3 spatial dimensions",
}
SUITE_LABELS = {
    "core": "Core benchmark",
    "extended": "Extended benchmark",
    "real": "Measured data",
    "other": "Other data",
}


def full_catalog():
    """Return one row per registered record, with no expensive data reads."""
    rows = []
    for name, spec in REGISTRY.items():
        meta = problem_metadata(name)
        rows.append({
            "id": name,
            "title": spec.title,
            "kind": spec.kind,
            "kind_label": KIND_LABELS.get(spec.kind, spec.kind),
            "suite": spec.suite,
            "suite_label": SUITE_LABELS.get(spec.suite, spec.suite),
            "source": spec.source,
            "files": spec.files,
            "law_known": bool(meta["truth_known"]),
        })
    return rows


def grouped_catalog():
    """All registered datasets, automatically including new additions."""
    rows = full_catalog()
    groups = {"All registered datasets": [row["id"] for row in rows]}
    for kind, label in KIND_LABELS.items():
        subset = [row["id"] for row in rows if row["kind"] == kind]
        if subset:
            groups[label] = subset
    for suite, label in SUITE_LABELS.items():
        subset = [row["id"] for row in rows if row["suite"] == suite]
        if subset:
            groups[label] = subset
    return groups


def declared_equations(name):
    """Only literal laws declared by the native loader; never infer a law.

    Nonliteral expressions or data-dependent truths are *not* reconstructed
    from symbolic guesses. Those still appear in the full catalog with a link
    to the source loader and may be inspected after an explicit data load.
    """
    if name not in REGISTRY:
        raise KeyError(name)
    tree = ast.parse(inspect.getsource(REGISTRY[name].loader))
    for call in ast.walk(tree):
        if isinstance(call, ast.Call):
            for keyword in call.keywords:
                if keyword.arg == "truth":
                    try:
                        declared = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                        return []
                    if isinstance(declared, str):
                        return [declared]
                    if isinstance(declared, (list, tuple)):
                        return [eq for eq in declared if isinstance(eq, str)]
    return []


def source_file_status(name):
    """Check stat metadata of literal source paths; never read numeric tensors.

    An incomplete/dynamic source declaration produces 'unknown', not a
    fabricated claim that every external research archive is installed.
    """
    from epde_bench.paths import DATA_DIR

    if name not in REGISTRY:
        raise KeyError(name)
    tree = ast.parse(inspect.getsource(REGISTRY[name].loader))
    paths = []
    dynamic = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in ("data_path", "_loadmat"):
            continue
        if node.args and all(isinstance(a, ast.Constant) and isinstance(a.value, str)
                             for a in node.args):
            paths.append(DATA_DIR.joinpath(*(a.value for a in node.args)))
        else:
            dynamic = True
    unique = list(dict.fromkeys(paths))
    def has_data(path):
        if path.is_file():
            return True
        if path.is_dir():
            # E.g. the sea-surface-temperature loader expects a directory
            # containing daily NetCDF files, not a regular file.
            if path.name == "sst_l4_files":
                return any(path.glob("*.nc"))
            return any(item.is_file() for item in path.iterdir())
        return False

    missing = [str(p.relative_to(DATA_DIR)) for p in unique if not has_data(p)]
    if dynamic or not unique:
        status = "unknown"
    elif missing:
        status = "missing"
    else:
        status = "present"
    return {"status": status, "checked": len(unique), "missing": missing}
