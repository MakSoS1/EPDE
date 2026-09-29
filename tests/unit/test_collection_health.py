#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Every module under ``tests/unit`` must import cleanly under any pytest flag.

``adapter_test.py`` parsed ``sys.argv[2:]`` with ``getopt`` at module scope, so
a plain ``-q`` reached it as an unknown short option and raised during
COLLECTION. A collection error is not a failed test: pytest aborts the whole
session and exits 2 having run nothing. That is why the suite had to be invoked
as ``--ignore=tests/unit/adapter_test.py`` -- one dead 2022 sketch (no working
test in it: ``TF_Pool``/``Define_Derivatives`` had been renamed away, ``x`` was
undefined, and its one test body was a bare ``a == b == c`` asserting nothing)
made the documented command for the whole suite conditional on remembering a
flag.

This pins the class rather than that one file. It is pure AST -- nothing is
imported and nothing is executed -- so it cannot itself break collection, and
it costs milliseconds.
"""

import ast
from pathlib import Path

import pytest

UNIT_DIR = Path(__file__).parent


def _module_level_calls(tree):
    """Every call made at module scope, as dotted-name strings."""
    names = []
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.FunctionDef) or isinstance(sub, ast.ClassDef):
                continue
            if not isinstance(sub, ast.Call):
                continue
            target = sub.func
            parts = []
            while isinstance(target, ast.Attribute):
                parts.append(target.attr)
                target = target.value
            if isinstance(target, ast.Name):
                parts.append(target.id)
            if parts:
                names.append('.'.join(reversed(parts)))
    return names


#: Reading argv at import time makes collection depend on the flags the suite
#: was invoked with. A test that genuinely needs an option should take it from
#: a fixture or ``pytest_addoption``, never from ``sys.argv``.
FORBIDDEN_AT_IMPORT = ('getopt.getopt', 'argparse.ArgumentParser')


@pytest.mark.parametrize('path', sorted(UNIT_DIR.glob('*.py')),
                         ids=lambda p: p.name)
def test_no_module_parses_argv_while_being_collected(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    called = _module_level_calls(tree)
    offenders = [name for name in called if name in FORBIDDEN_AT_IMPORT]
    assert not offenders, (
        f'{path.name} calls {offenders} at module scope; any pytest flag then '
        'reaches it and the whole session dies during collection')
