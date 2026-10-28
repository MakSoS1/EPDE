"""Catalog navigation starts a fresh experiment without touching saved runs.

This module has no Streamlit/scientific imports. The caller is responsible
for selecting a registered catalog ID before invoking this transition.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

EXPERIMENT_STATE_PREFIXES = ("lab_", "portable_", "heavy_", "native_advanced_")


def begin_catalog_experiment(state: MutableMapping[str, Any], record_name: str) -> None:
    """Discard the old wizard before the page switch, not only on stage one.

    Reset the saved widget mirror as well as the rendered widget keys so
    back-navigation cannot resurrect an approved config for a different input.
    Jobs already stored in the queue are not cancelled, deleted or resubmitted.
    Session identity, theme, catalog filters and Results/Compare are preserved.
    """
    if not isinstance(record_name, str) or not record_name.strip():
        raise ValueError("Choose a registered catalog record")
    for key in list(state):
        if isinstance(key, str) and key.startswith(EXPERIMENT_STATE_PREFIXES):
            state.pop(key, None)
    state["lab_stage"] = 1
    state["lab_input_source"] = "Built-in equation"
    state["catalog_requested_record"] = record_name
