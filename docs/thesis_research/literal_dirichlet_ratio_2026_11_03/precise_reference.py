"""Independent reference for the Dirichlet ratio: Decimal arithmetic, Spouge's Gamma approximation; no module import.

Two results per case:
- "pipeline": the double values when every source operation is rounded once to the nearest double (as the module
  claims to do), with lgamma from Spouge's formula (a = 70, 130 digits) instead of the module's Stirling series and
  row sums as exact sums of the doubles rounded once;
- "exact": the same quantities without any intermediate rounding (60 significant digits as strings), used for the
  closed-form hand values and to show the size of the source's rounding.
IEEE specials follow the source: x/0 is Inf or NaN, 0 * Inf is NaN, NaN becomes 0 in line 1273.
"""
import math
from decimal import Decimal, localcontext
from fractions import Fraction

A, PREC = 70, 130


def _ctx():
    return localcontext(prec=PREC, Emax=10**7, Emin=-10**7)


def _pi():
    """pi by the Gauss-Legendre iteration (a different method from the module)."""
    with _ctx():
        a, b, t, p = Decimal(1), Decimal(1) / Decimal(2).sqrt(), Decimal(1) / 4, Decimal(1)
        for _ in range(9):
            a, b, t, p = (a + b) / 2, (a * b).sqrt(), t - p * ((a - b) / 2) ** 2, 2 * p
        return (a + b) ** 2 / (4 * t)


PI = _pi()


def _spouge_coefficients():
    with _ctx():
        c = [(2 * PI).sqrt()]
        fact = Decimal(1)
        for k in range(1, A):
            if k > 1:
                fact *= k - 1
            c.append((-1) ** (k - 1) / fact * Decimal(A - k) ** (Decimal(k) - Decimal('0.5')) * Decimal(A - k).exp())
        return c


C = _spouge_coefficients()


def lgamma_exact(x):
    """ln Gamma(x) for Decimal x > 0: ln Gamma(x+1) by Spouge, minus ln x."""
    with _ctx():
        if x in (1, 2):
            return Decimal(0)
        z = x
        s = C[0] + sum(C[k] / (z + k) for k in range(1, A))
        return (z + Decimal('0.5')) * (z + A).ln() - (z + A) + s.ln() - x.ln()


def _rd(v):
    """Round a Decimal once to the nearest double (Inf/0 on over/underflow)."""
    return float(v)


def _sum_rounded(values):
    return float(sum((Fraction(v) for v in values), Fraction(0)))


def density_pipeline(x, alpha):
    with _ctx():
        s_alpha = _sum_rounded(alpha)
        t1 = _rd(lgamma_exact(Decimal(s_alpha)))
        t2 = _sum_rounded([_rd(lgamma_exact(Decimal(a))) for a in alpha])
        t3 = _sum_rounded([(a - 1.0) * _rd(Decimal(v).ln()) for a, v in zip(alpha, x)])
        dens = (t1 - t2) + t3
        if dens > 1000:
            return math.inf
        if dens < -1000:
            return 0.0
        return _rd(Decimal(dens).exp())


def density_exact(x, alpha):
    with _ctx():
        xs, al = [Decimal(v) for v in x], [Decimal(a) for a in alpha]
        log_d = lgamma_exact(sum(al)) - sum(lgamma_exact(a) for a in al) + sum((a - 1) * v.ln()
                                                                              for a, v in zip(al, xs))
        return log_d.exp()


def _div(a, b):
    if b == 0:
        return math.nan if a == 0 or math.isnan(a) else math.inf
    return a / b


def ratio(old, new, num_taxa):
    half = [0.5] * num_taxa
    wn, wd = density_pipeline(new, half), density_pipeline(old, half)
    pn, pd = density_pipeline(old, new), density_pipeline(new, old)
    w, p = _div(wn, wd), _div(pn, pd)
    raw = p * w
    with _ctx():
        ewn, ewd = density_exact(new, half), density_exact(old, half)
        epn, epd = density_exact(old, new), density_exact(new, old)
        exact = {'weights_ratio': ewn / ewd, 'proposal_ratio': epn / epd}
        exact['parameter_ratio'] = exact['weights_ratio'] * exact['proposal_ratio']
    return {'densities': [wn, wd, pn, pd], 'weights_ratio': w, 'proposal_ratio': p,
            'parameter_ratio_before_na': raw, 'parameter_ratio': 0.0 if math.isnan(raw) else raw,
            'exact': {k: format(v, '.60g') for k, v in exact.items()}}


def closed_form(text):
    """Evaluate the fixture's closed forms: integers, pi, sqrt(), / and * only."""
    import ast

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return Decimal(node.value)
        if isinstance(node, ast.Name) and node.id == 'pi':
            return PI
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'sqrt':
            return ev(node.args[0]).sqrt()
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Mult)):
            a, b = ev(node.left), ev(node.right)
            return a / b if isinstance(node.op, ast.Div) else a * b
        raise ValueError(f'unsupported closed form {text!r}')
    with _ctx():
        return ev(ast.parse(text, mode='eval'))
