"""Tamper-evident cache checksums for exact weighted sufficient statistics."""

from __future__ import annotations

import hashlib
import struct

import numpy as np

from .gram import GramBlocks


def gram_checksum(gram: GramBlocks) -> str:
    h = hashlib.sha256()
    for arr in (gram.G, gram.b):
        value = np.ascontiguousarray(arr)
        h.update(str(value.shape).encode())
        h.update(str(value.dtype).encode())
        h.update(value.tobytes())
    h.update(struct.pack("!dd", gram.yy, gram.weight_sum))
    for name, (g, b, yy) in sorted(gram.by_environment.items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(g).tobytes())
        h.update(np.ascontiguousarray(b).tobytes())
        h.update(struct.pack("!d", yy))
    return h.hexdigest()
