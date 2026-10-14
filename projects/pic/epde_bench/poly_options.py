"""Scientific preflight for polynomial differentiation in EPDE Explorer.

The native Chebyshev fitter uses poly_order (or max derivative + 1 when None).
Polynomial order d cannot supply a nonzero derivative of order > d.
It also uses a centred odd window; the implementation falls back to FD near
edges. Validate before starting a possibly expensive evolutionary search.
"""
from __future__ import annotations


def validate_polynomial_options(search: dict, shape=None) -> None:
    prep = search.get("preprocessing", {})
    if prep.get("default_preprocessor_type") != "poly":
        return
    kw = prep.get("preprocessor_kwargs") or {}
    raw = prep.get("max_deriv_order", [2])
    orders = [int(x) for x in raw] if isinstance(raw, (list, tuple)) else [int(raw)]
    if not orders or any(k < 0 for k in orders):
        raise ValueError("Polynomial derivative orders must be nonnegative")
    window = kw.get("polynomial_window", 9)
    if isinstance(window, bool) or not isinstance(window, int) or window < 5 or window % 2 != 1:
        raise ValueError("Polynomial window must be an odd window length (integer >= 5)")
    degree = kw.get("poly_order")
    if degree is None:
        degree = max(orders) + 1
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("Polynomial degree must be a positive integer")
    if degree < max(orders):
        raise ValueError(
            f"Polynomial degree {degree} cannot compute derivative order "
            f"{max(orders)}. Increase degree or lower the requested derivative order."
        )
    if degree >= window - 1:
        raise ValueError(
            f"Polynomial degree {degree} needs at least {degree + 2} "
            f"grid points in the odd window; selected window is {window}."
        )
    if shape is not None and any(int(size) < window + 2 for size in shape):
        raise ValueError(
            f"Polynomial window {window} needs at least {window + 2} "
            f"samples on every axis, found {tuple(shape)}. "
            "Use a smaller window or finite differences."
        )


def polynomial_kwargs(window: int, degree: int | None, orders, shape=None) -> dict:
    """Validated EPDE-native keyword arguments, keeping the CPU pool bounded."""
    kws = {"mp_poolsize": 1, "polynomial_window": int(window),
           "poly_order": None if degree is None else int(degree)}
    validate_polynomial_options({"preprocessing": {
        "default_preprocessor_type": "poly",
        "max_deriv_order": list(orders),
        "preprocessor_kwargs": kws}}, shape=shape)
    return kws
