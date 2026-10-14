"""Bounded, numeric-only input files for the Discovery Lab."""
from __future__ import annotations
import io
import zipfile
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024
MAX_CELLS = 300_000


def parse_upload(raw: bytes, filename: str):
    import numpy as np
    from ..custom_run import validate_arrays
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("File must be nonempty and at most 8 MiB")
    ext = Path(filename).suffix.lower()
    if ext == ".npz":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if len(z.infolist()) > 15 or any(i.file_size > MAX_BYTES for i in z.infolist()):
                raise ValueError("Expanded archive entries are too large")
            if sum(i.file_size for i in z.infolist()) > 2 * MAX_BYTES:
                raise ValueError("Expanded archive too large")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if len(archive.files) > 15:
                raise ValueError("Too many arrays")
            arrays = {k: archive[k] for k in archive.files}
    elif ext == ".csv":
        import pandas as pd
        table = pd.read_csv(io.BytesIO(raw), nrows=MAX_CELLS + 1)
        if not 2 <= len(table.columns) <= 10 or len(table) < 10:
            raise ValueError("CSV needs time and 1-9 variables with at least 10 samples")
        if len(table) * len(table.columns) > MAX_CELLS:
            raise ValueError("CSV exceeds 300,000 numerical values")
        arrays = {"axis_0": table.iloc[:,0].to_numpy(dtype=float),
                  "axis_names": np.array(["t"])}
        for col in table.columns[1:]:
            arrays[f"var_{col}"] = table[col].to_numpy(dtype=float)
    elif ext == ".npy":
        field = np.load(io.BytesIO(raw), allow_pickle=False)
        if field.ndim != 2:
            raise ValueError("NPY field must be a 2-D time-by-space array")
        arrays = {"axis_0": np.linspace(0., 1., field.shape[0]),
                  "axis_1": np.linspace(0., 1., field.shape[1]),
                  "axis_names": np.array(["t", "x"]), "var_u": field}
    else:
        raise ValueError("Upload only CSV, NPY or NPZ")
    if sum(v.size for v in arrays.values() if hasattr(v, "size")) > MAX_CELLS:
        raise ValueError("Arrays exceed 300,000 values")
    validate_arrays(arrays)
    return arrays
