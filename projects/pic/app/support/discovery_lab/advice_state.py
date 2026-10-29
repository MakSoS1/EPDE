"""Small wizard state transitions for editable starting suggestions."""
from __future__ import annotations

from copy import deepcopy

_SETTINGS_PREFIXES = (
    "lab_variant_", "lab_budget_", "lab_pop_", "lab_epochs_", "lab_terms_",
    "lab_diff_", "lab_order_", "lab_poly_", "lab_boundary_", "lab_data_power_",
    "lab_deriv_power_", "lab_products_", "lab_inverse_", "lab_obj_",
    "lab_sparsity_", "lab_cache_", "lab_yaml_", "lab_export_", "lab_pdecomp_",
    "lab_deepmod_", "lab_discover_", "portable_", "heavy_", "native_advanced_",
)


def _selector(key):
    return (key.startswith("lab_variant_") or key.endswith(("_protocol", "_protocols", "_target"))
            or key == "lab_pdecomp_target_order" or key.startswith("portable_pdecomp_protocol_"))


def _reset_settings(state, *, keep_selectors):
    saved = dict(state.get("lab_widgets_saved", {}))
    for key in set(state) | set(saved):
        if key.startswith(("heavy_noise_", "heavy_seed_")):
            continue  # Stage-one input controls, not method settings.
        if key.startswith(_SETTINGS_PREFIXES) and not (keep_selectors and _selector(key)):
            state.pop(key, None)
            saved.pop(key, None)
    state.pop("lab_draft_config", None)
    state.pop("lab_started_jobs", None)
    state["lab_confirm_science"] = False
    saved["lab_confirm_science"] = False
    state["lab_widgets_saved"] = saved
    state["lab_advice_reports"] = {}
    state["lab_advice_generation"] = int(state.get("lab_advice_generation", 0)) + 1


def prepare_context(state, context_id):
    """Reset stale settings only when the numerical input/context changes."""
    if state.get("lab_advice_context") == context_id:
        return False
    _reset_settings(state, keep_selectors=False)
    state["lab_advice_context"] = context_id
    return True


def request_suggestions(state):
    """Explicit reapply keeps the selected method profiles and target semantics."""
    _reset_settings(state, keep_selectors=True)


def save_report(state, scope, report):
    reports = dict(state.get("lab_advice_reports", {}))
    reports[scope] = deepcopy(report)
    state["lab_advice_reports"] = reports
