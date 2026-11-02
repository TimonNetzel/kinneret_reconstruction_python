"""Exact arithmetic oracle for this synthetic fixture only; no NumPy/module imports.

Fractions and explicit scalar sums independently evaluate the documented
formulas. It is not an alternative historical runner and reads no originals.
"""
from fractions import Fraction as F


def matrix(rows):
    return [[F(str(x)) for x in row] for row in rows]


def dot(left, right):
    return [[sum((left[i][j] * right[j][k] for j in range(len(right))), F(0))
             for k in range(len(right[0]))] for i in range(len(left))]


def strings(rows):
    return [[str(x) for x in row] for row in rows]


def exact_expectations(f):
    position = f["resolution_id_R_1_based"] - 1
    rho = matrix(f["rho_grids"][position])
    site_age = [F(str(v)) for v in f["age_grids"][position]]
    seen = set()
    age = []
    for a in site_age:
        age.append(a + F(1, 2) if a in seen else a)
        seen.add(a)
    step = age[1] - age[0]
    masks = {name: [lo <= a <= hi and (a - lo) % step == 0 for a in age]
             for name, lo, hi in (("hol", -50, 11450), ("lg", 11500, 115000), ("eem", 115500, 125000))}
    # This explicitly models only the preserved reset IF the source lines are reached.
    initial = masks["hol"] if f["taxa_merge_control"] else [True] * len(age)
    effective = [True] * len(age)
    depths = [F(str(d)) for d in f["depths"]]
    indices = [int(d - min(depths)) for d in depths]
    w = []
    for row in rho:
        selected = [row[i] for i in indices]
        mass = sum(selected)
        w.append([v / mass if v != 0 else F(1, 10**9) for v in selected])
    p = matrix(f["pollen_depth_by_taxon"])
    p = [[v if v != 0 else F(1, 10**6) for v in row] for row in p]
    pollen = dot(w, p)
    maxima = [max(row[j] for row in pollen) for j in range(len(pollen[0]))]
    scaled = [[v / maxima[j] for j, v in enumerate(row)] for row in pollen]
    ids, known = [], set()
    for i, energy in enumerate(f["energy"], 1):
        e = F(str(energy))
        if e not in known:
            ids.append(i)
            known.add(e)
    tf = matrix(f["transfer_functions_taxon_by_cell"])
    draws = {}
    densities = []
    for i in ids:
        weights = [F(str(v)) for v in f["rw_weights_by_original_draw_id"][str(i)]]
        coefficients = []
        for row in scaled:
            weighted = [v * weights[j] for j, v in enumerate(row)]
            mass = sum(weighted)
            coefficients.append([v / mass for v in weighted])
        density = dot([row for row, keep in zip(coefficients, effective) if keep], tf)
        cumulative = [[sum(row[:k+1]) for k in range(len(row))] for row in density]
        scores = [sum(row) / f["current_dims"] for row in cumulative]
        densities.append(density)
        draws[str(i)] = {"coefficients": strings(coefficients), "raw_density": strings(density),
                         "raw_cumulative": strings(cumulative), "zero_reference_scores": [str(v) for v in scores]}
    raw_sum = [[sum((d[a][k] for d in densities), F(0)) for k in range(len(tf[0]))] for a in range(len(age))]
    pdf = [[v / sum(row) for v in row] for row in raw_sum]
    return {"site_age": [str(v) for v in site_age], "age": [str(v) for v in age], "period_masks": masks,
            "initial_reconst_mask": list(initial), "effective_reconst_mask": effective,
            "depth_weights": strings(w), "pollen_at_ages": strings(pollen),
            "column_max_scaled": strings(scaled), "draw_ids_R_1_based": ids, "draws": draws,
            "raw_sum": strings(raw_sum), "pdf": strings(pdf)}
