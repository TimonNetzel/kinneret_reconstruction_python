"""Exact scalar arithmetic for artificial TF fixtures; no NumPy/module imports."""
from fractions import Fraction


def exact_preparation(raw_rows, keep):
    normalized = []
    masked = []
    for row in raw_rows:
        values = [Fraction(str(v)) for v in row]
        total = sum(values, Fraction(0))
        norm = [v / total for v in values]
        normalized.append(norm)
        masked.append([v if keep[k] else Fraction(0) for k, v in enumerate(norm)])
    return {
        "normalized": [[str(v) for v in row] for row in normalized],
        "masked": [[str(v) for v in row] for row in masked],
        "row_masses_after_mask": [str(sum(row, Fraction(0))) for row in masked],
    }
