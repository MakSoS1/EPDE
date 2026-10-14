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


def uses_time(expression: str) -> bool:
    """Whether the right-hand side depends on t explicitly (non-autonomous)."""
    compile_rhs(expression)
    return any(isinstance(node, ast.Name) and node.id == "t"
               for node in ast.walk(ast.parse(expression, mode="eval")))


def _num(value: float) -> str:
    return repr(float(value))


def reference_law(expression: str):
    """The generated ODE as an EPDE equation, so a search on it can be scored.

    Returns ``(equation_text, frequency, note)``. ``equation_text`` is ``None`` when
    a term cannot be written with the tokens the Lab offers (u up to the cube,
    t up to the square, one sine/cosine frequency of t, at most two factors per
    term); ``frequency`` is the single sin/cos frequency used (or ``None``).
    The expression is validated by :func:`compile_rhs` first; sympy only parses
    an already allowlisted syntax.
    """
    import sympy as sp
    compile_rhs(expression)
    t, u = sp.symbols('t u')
    expr = sp.expand(sp.sympify(expression, locals={'t': t, 'u': u, 'sin': sp.sin,
                                                    'cos': sp.cos, 'tanh': sp.tanh}))
    terms, constant, freqs = [], 0.0, set()
    for term in sp.Add.make_args(expr):
        coef, rest = term.as_coeff_Mul()
        if rest == 1:
            constant += float(coef)
            continue
        factors = []
        for factor in sp.Mul.make_args(rest):
            base, power = factor.as_base_exp()
            if not power.is_Integer or power < 1:
                return None, None, f'{factor} is not a positive integer power'
            power = int(power)
            if base == u and power <= 3:
                factors.append(f'u{{power: {float(power)}}}')
            elif base == t and power <= 2:
                factors.append(f'x{{power: {float(power)}, dim: 0.0}}')
            elif isinstance(base, (sp.sin, sp.cos)) and power == 1:
                k = sp.simplify(base.args[0] / t)
                if not k.is_number or k.free_symbols or float(k) <= 0:
                    return None, None, f'{base} is not sin/cos of a positive multiple of t'
                freqs.add(round(float(k), 9))
                name = 'sin' if isinstance(base, sp.sin) else 'cos'
                factors.append(f'{name}{{power: 1.0, freq: {float(k)}, dim: 0.0}}')
            else:
                return None, None, f'{factor} has no token in the search space'
        if len(factors) > 2:
            return None, None, f'{rest} has more than two factors'
        terms.append(f'{_num(coef)} * ' + ' * '.join(factors))
    if len(freqs) > 1:
        return None, None, 'only one sine/cosine frequency can be searched at a time'
    if constant:
        terms.append(_num(constant))
    if not terms:
        return None, None, 'the right-hand side is zero'
    equation = ' + '.join(terms) + ' = du/dx0{power: 1.0}'
    return equation, (next(iter(freqs)) if freqs else None), None


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
