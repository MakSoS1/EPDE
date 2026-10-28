"""EPDE text form -> LaTeX, for readable equations in the app.

``-0.98 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + 0.0 = d^2u/dx0^2{power: 1.0}``
becomes ``u_{tt} = -0.98\\, u_{t}\\,\\sin(2 t)``.
"""

import ast
import re

_FACTOR = re.compile(r'([A-Za-z0-9_\^/\(\)]+)\s*\{([^}]*)\}')
_DERIV = re.compile(r'^d(?:\^(\d+))?([A-Za-z_][A-Za-z0-9_]*)/dx(\d+)(?:\^\d+)?$')


def _params(text):
    out = {}
    for item in text.split(','):
        if ':' in item:
            key, value = item.split(':', 1)
            try:
                out[key.strip()] = float(value)
            except ValueError:
                out[key.strip()] = value.strip()
    return out


def _number(value):
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    return f'{value:.4g}'


def _power(base, power):
    return base if power == 1 else f'{{{base}}}^{{{_number(power)}}}'


def _variable(name):
    if len(name) == 1:
        return name
    if '(' in name:                        # product tokens such as cos(t)sin(x)
        return re.sub(r'(sin|cos)\(([a-z])\)', r'\\\1 \2\\,', name).rstrip('\\,')
    return rf'\mathrm{{{name}}}'.replace('_', r'\_')


def factor_latex(name, params, axes):
    power = params.get('power', 1.0)
    m = _DERIV.match(name)
    if m:
        order, var, axis = int(m.group(1) or 1), m.group(2), int(m.group(3))
        label = axes[axis] if axis < len(axes) else f'x_{axis}'
        return _power(f'{_variable(var)}_{{{label * order}}}', power)
    if name in ('sin', 'cos'):
        axis = int(params.get('dim', 0))
        label = axes[axis] if axis < len(axes) else f'x_{axis}'
        freq = params.get('freq', 1.0)
        arg = f'{_number(freq)} {label}' if freq != 1 else label
        return _power(rf'\{name}({arg})', power)
    if name == 'x' or re.match(r'^x_\d+$', name):
        axis = int(params.get('dim', 0))
        return _power(axes[axis] if axis < len(axes) else f'x_{axis}', power)
    return _power(_variable(name), power)


def term_latex(text, axes):
    coef, factors = 1.0, []
    for piece in (p.strip() for p in text.split('*')):
        m = _FACTOR.search(piece) if piece else None
        if m:
            factors.append(factor_latex(m.group(1), _params(m.group(2)), axes))
        elif piece:
            try:
                coef *= float(piece)
            except ValueError:
                factors.append(rf'\mathrm{{{piece}}}')
    return coef, r'\,'.join(factors)


def equation_latex(eq_text, axes=('t', 'x', 'y', 'z'), *, wrap=False):
    """LaTeX of one EPDE equation; ``axes`` names x0, x1, ... in order."""
    axes = list(axes)
    lhs_text, rhs_text = eq_text.strip().lstrip('/|\\ ').split('=', 1)
    target_coef, target = term_latex(rhs_text, axes)
    parts = []
    for chunk in re.split(r'\s\+\s', lhs_text):
        coef, body = term_latex(chunk, axes)
        if coef == 0:
            continue
        sign = '-' if coef < 0 else '+'
        value = abs(coef) / (target_coef or 1.0)
        number = '' if body and abs(value - 1) < 1e-9 else _number(value)
        if body and number:
            text = rf'{number}\,{body}'
        else:
            text = body or number
        parts.append((sign, text))
    if not parts:
        rhs = '0'
    else:
        rhs = ''.join((s if i or s == '-' else '') + (' ' if i else '') + p + ' '
                      for i, (s, p) in enumerate(parts)).strip()
    if wrap and len(parts) > 2:
        lines = [(('' if i == 0 and sign == '+' else sign + ' ') + text)
                 for i, (sign, text) in enumerate(parts)]
        return r'\begin{aligned}' + target + ' &= ' + lines[0] + ''.join(
            r' \\ &\quad ' + line for line in lines[1:]) + r'\end{aligned}'
    return f'{target} = {rhs}'


def native_equation_latex(eq_text, axes=('t', 'x'), variables=('u',)):
    """Render DISCOVER's expression grammar without evaluating its source text."""
    coordinate = axes[1] if len(axes) > 1 else axes[0]

    def render(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return _number(node.value)
        if isinstance(node, ast.Name):
            if node.id in ('u1', 'u'):
                return _variable(variables[0])
            if node.id == 'x1':
                return _variable(coordinate)
            target = re.fullmatch(r'u_([txyz]+)', node.id)
            if target:
                return _variable(variables[0]) + '_{' + target.group(1) + '}'
            return _variable(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return ('-' if isinstance(node.op, ast.USub) else '') + render(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = render(node.left), render(node.right)
            if isinstance(node.op, ast.Add):
                return left + ' + ' + right
            if isinstance(node.op, ast.Sub):
                return left + ' - ' + right
            if isinstance(node.op, ast.Mult):
                return atom(node.left) + r'\,' + atom(node.right)
            if isinstance(node.op, ast.Div):
                return r'\frac{' + left + '}{' + right + '}'
            if isinstance(node.op, ast.Pow):
                return '{' + atom(node.left) + '}^{' + right + '}'
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name, args = node.func.id, [render(a) for a in node.args]
            if name in ('sin', 'cos', 'exp', 'log') and len(args) == 1:
                return '\\' + name + r'\left(' + args[0] + r'\right)'
            if name in ('n2', 'n3') and len(args) == 1:
                return '{' + args[0] + '}^{' + name[1:] + '}'
            if name == 'mul' and len(args) == 2:
                return atom(node.args[0]) + r'\,' + atom(node.args[1])
            if name == 'add' and len(args) == 2:
                return r'\left(' + args[0] + ' + ' + args[1] + r'\right)'
            if name == 'div' and len(args) == 2:
                return r'\frac{' + args[0] + '}{' + args[1] + '}'
            if name in ('diff', 'diff2', 'diff3', 'diff4') and len(args) == 2:
                order = name[4:] or '1'
                return r'\operatorname{diff}_{' + order + r'}\!\left(' + args[0] + '; ' + args[1] + r'\right)'
        raise ValueError('Unsupported display expression')

    def atom(node):
        text = render(node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return r'\left(' + text + r'\right)'
        return text

    def parse(text):
        text = re.sub(r'(?<=[\w)])\s+(?=[\w(])', '*', text.strip().replace('^', '**'))
        return render(ast.parse(text, mode='eval').body)

    lhs, rhs = eq_text.split('=', 1)
    return parse(rhs) + ' = ' + parse(lhs)


def system_latex(system, axes=('t', 'x', 'y', 'z')):
    lines = [equation_latex(eq, axes) for eq in system]
    if len(lines) == 1:
        return lines[0]
    return r'\begin{cases}' + r' \\ '.join(lines) + r'\end{cases}'


_PLAIN = re.compile(r"([A-Za-z0-9_\^/\(\)]+)\s*\{([^}]*)\}")


def term_plain(text, axes=('t', 'x', 'y', 'z')):
    """Readable plain-text term: ``du/dx0{'power': 1.0} * sin{...freq 2...}`` -> ``u_t·sin(2t)``."""
    out = []
    for name, params in _PLAIN.findall(text):
        p = _params(params.replace("'", ''))
        power = p.get('power', 1.0)
        m = _DERIV.match(name)
        if m:
            axis = int(m.group(3))
            label = axes[axis] if axis < len(axes) else f'x{axis}'
            base = f"{m.group(2)}_{label * int(m.group(1) or 1)}"
        elif name in ('sin', 'cos'):
            axis = int(p.get('dim', 0))
            freq = p.get('freq', 1.0)
            base = f"{name}({_number(freq) if freq != 1 else ''}{axes[axis] if axis < len(axes) else axis})"
        elif name == 'x' or re.match(r'^x_\d+$', name):
            axis = int(p.get('dim', 0))
            base = axes[axis] if axis < len(axes) else f'x{axis}'
        else:
            base = name
        out.append(base if power == 1 else f'{base}^{_number(power)}')
    return '·'.join(out) or text
