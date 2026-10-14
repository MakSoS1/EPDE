"""Safe YAML import/export for EPDE Explorer experiments (no Python evaluation)."""
from __future__ import annotations

import yaml

# Only configuration data are accepted, never import paths or user Python code.
GROUPS = frozenset(("domain", "preprocessing", "search_space",
                    "objectives", "solver", "evolution", "runtime"))
SAFE_FAMILIES = frozenset(("grid", "constant", "trigonometric",
                           "fixed_trigonometric", "inverse_space_coordinate",
                           "data_polynomials", "phased_sine_1d"))
MAX_CONFIG_BYTES = 65536


def parse_search_yaml(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("YAML must be a file of at most 64 KiB")
    try:
        obj = yaml.safe_load(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("Invalid UTF-8 YAML settings") from exc
    if not isinstance(obj, dict) or set(obj) != {"search"}:
        raise ValueError("Import only YAML with one top-level 'search' mapping")
    cfg = obj["search"]
    if not isinstance(cfg, dict) or any(k not in GROUPS for k in cfg):
        raise ValueError("Unknown search configuration group")
    for key, value in cfg.items():
        if not isinstance(value, dict):
            raise ValueError(f"search.{key} must be a mapping")
    space = cfg.get("search_space", {})
    for token in space.get("tokens", []) or []:
        if not isinstance(token, dict) or token.get("family") not in SAFE_FAMILIES:
            raise ValueError("Imported tokens must use an approved declarative family")
    evo = cfg.get("evolution", {})
    if not 1 <= int(evo.get("population_size", 16)) <= 96:
        raise ValueError("Imported population_size must be within 1..96")
    if not 1 <= int(evo.get("training_epochs", 5)) <= 50:
        raise ValueError("Imported training_epochs must be within 1..50")
    if int(space.get("equation_terms_max_number", 6)) > 15:
        raise ValueError("Imported equation_terms_max_number must not exceed 15")
    return obj


def dump_search_yaml(overrides: dict) -> str:
    """Export settings as a transportable, declarative override."""
    if not isinstance(overrides, dict) or set(overrides) != {"search"}:
        raise ValueError("Expected a search-only mapping")
    return yaml.safe_dump(overrides, sort_keys=False, allow_unicode=True)
