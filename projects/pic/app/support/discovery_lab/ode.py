"""Restricted mathematical-expression ODE generator.

Supports a scalar du/dt=f(t,u). User input is parsed with an allowlist.
"""
from __future__ import annotations
import ast
import math
from typing import Callable

_ALLOWED_FUNCS = {"sin": math.sin, "cos": math.cos, "tanh": math.tanh}
_MAX_NODES = 60


def compile_rhs(expression: str) -> Callable[[float, float], float]:
    if not isinstance(expression, str) or not 1 <= len(expression) <= 160:
        raise ValueError("RHS must be 1-160 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid mathematical expression") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_NODES:
        raise ValueError("Expression is too complex")

    def parse(node):
        if isinstance(node, ast.Expression):
            return parse(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
            if not math.isfinite(value) or abs(value) > 1e4:
                raise ValueError("Constant out of range")
            return lambda t, u, v=value: v
        if isinstance(node, ast.Name) and node.id in ("t", "u"):
            return (lambda t, u: t) if node.id == "t" else (lambda t, u: u)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            f = parse(node.operand)
            return (lambda t, u: f(t, u)) if isinstance(node.op, ast.UAdd) else (lambda t, u: -f(t, u))
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = parse(node.left), parse(node.right)
            if isinstance(node.op, ast.Add):
                return lambda t, u: a(t,u) + b(t,u)
            if isinstance(node.op, ast.Sub):
                return lambda t, u: a(t,u) - b(t,u)
            if isinstance(node.op, ast.Mult):
                return lambda t, u: a(t,u) * b(t,u)
            if isinstance(node.op, ast.Div):
                return lambda t, u: a(t,u) / b(t,u)
            if not isinstance(node.right, ast.Constant) or type(node.right.value) not in (int, float):
                raise ValueError("Exponent must be a constant")
            power = float(node.right.value)
            if not -3 <= power <= 4 or power != int(power):
                raise ValueError("Exponent must be an integer from -3 to 4")
            return lambda t, u: a(t,u) ** power
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _ALLOWED_FUNCS and len(node.args) == 1
                and not node.keywords):
            f, arg = _ALLOWED_FUNCS[node.func.id], parse(node.args[0])
            return lambda t, u: f(arg(t,u))
        raise ValueError("Use only t, u, + - * / ** and sin/cos/tanh")

    fn = parse(tree)
    def checked(t, u):
        try:
            result = float(fn(float(t), float(u)))
        except (ArithmeticError, ValueError, OverflowError) as exc:
            raise ValueError("RHS undefined on the integration trajectory") from exc
        if not math.isfinite(result) or abs(result) > 1e8:
            raise ValueError("ODE trajectory diverged")
        return result
    return checked


def generate(expression: str, t_end: float, u0: float, points: int):
    if not 1 <= float(t_end) <= 30 or not -10 <= float(u0) <= 10:
        raise ValueError("Invalid initial state or horizon")
    if not 60 <= int(points) <= 3000:
        raise ValueError("Allowed sample count: 60-3000")
    import numpy as np
    from scipy.integrate import solve_ivp
    rhs = compile_rhs(expression)
    ts = np.linspace(0.0, float(t_end), int(points))
    evaluations = 0
    def bounded_rhs(t, y):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 25000:
            raise ValueError("Integration exceeded the 25,000-evaluation budget")
        return [rhs(t, y[0])]
    sol = solve_ivp(bounded_rhs, (0.0, float(t_end)), [float(u0)],
                    t_eval=ts, rtol=1e-7, atol=1e-9, max_step=.05)
    if not sol.success or len(sol.t) != len(ts) or not np.isfinite(sol.y).all():
        raise ValueError("Integration did not produce a finite trajectory")
    return {"axis_0": ts, "axis_names": np.array(["t"]), "var_u": sol.y[0]}
