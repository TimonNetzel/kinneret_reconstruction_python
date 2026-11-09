"""Bind ratios_tfs_no_ages (mcmc_functions_v1.6.R:1485-1525) and its reader, and record which run steps reach it. No draw, no compilation.

Reads the frozen originals, the CRAN tarballs of Rcpp and the R 3.5.0 tarball in memory. Only hashes, versions and line numbers are stored; no GPL text enters this repository.
A version counts when it was the current one at any time inside the window. Writes source_binding.json. Usage: bind_source.py <cache dir of Rcpp tarballs> <R-3.5.0.tar.gz>
"""
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TARBALL_SHA256 = 'fd1725535e21797d3d9fea8963d99be0ba4c3aecadcf081b43e261458b416870'
F, P, C = 'src/mcmc_functions_v1.6.R', 'src/mcmc_prepare.R', 'src/mcmc_conduction.R'
SOURCES = {F: 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c', P: 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e',
           C: '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d'}
PINS = {
    F: [(1437, 'List ratios_tfs_no_ages(int num_taxa, NumericVector new_tann,'), (1483, '// [[Rcpp::depends(RcppDist)]]'), (1484, '// [[Rcpp::export]]'), (1485, 'List ratios_tfs_no_ages(int num_taxa, NumericVector tann_range,NumericVector pann_range,'),
        (1503, 'for(int i=0; i<num_taxa; i++) {'), (1504, 'NumericVector tann_dists_i = tann_dists(i, _);'), (1507, 'propose_tann = tann_dists_i(which_min(abs(tann_range - new_tann[i])));'), (1508, 'start_tann = tann_dists_i(which_min(abs(tann_range - old_tann[i])));'),
        (1509, 'tann_tf_ratio[i] = propose_tann/start_tann;'), (1511, 'propose_pann = pann_dists_i(which_min(abs(pann_range - new_pann[i])));'), (1512, 'start_pann = pann_dists_i(which_min(abs(pann_range - old_pann[i])));'), (1513, 'pann_tf_ratio[i] = propose_pann/start_pann;'),
        (1516, 'propose_dists_tann = d_truncnorm(old_tann[i],new_tann[i],'), (1517, 'start_dists_tann = d_truncnorm(new_tann[i],old_tann[i],'), (1518, 'tann_dists_tf_ratio[i] = propose_dists_tann/start_dists_tann;'),
        (1520, 'propose_dists_pann = d_truncnorm(old_pann[i],new_pann[i],'), (1521, 'start_dists_pann = d_truncnorm(new_pann[i],old_pann[i],'), (1522, 'pann_dists_tf_ratio[i] = propose_dists_pann/start_dists_pann;'),
        (1524, 'return List::create(tann_tf_ratio,pann_tf_ratio,tann_dists_tf_ratio,pann_dists_tf_ratio);'), (1529, '# ratios_tfs_no_ages <- function(num_taxa, tann_range, pann_range,'), (1533, '#         propose_tann <- tann_dists[i,which.min(abs(tann_range - new_tann[i]))]'), (1552, '# }')],
    P: [(1890, 'taxa_tann_dens <- taxa_pann_dens <- array(NA, dim = c(num_taxa,1000))'), (1934, 'tann_range <- seq( min(x1range.bck), max(x1range.bck), length.out = 1000)'), (1935, 'pann_range <- seq( min(x2range.bck), max(x2range.bck), length.out = 1000)')],
    C: [(138, 'for(i in 2:sample_length){'), (228, 'ratios_tfs <- ratios_tfs_no_ages(num_taxa, tann_range = tann_range,pann_range = pann_range,'), (230, 'posterior$transfer_ratio[i] <- prod(unlist(ratios_tfs))'),
        (302, 'posterior$acc_prob[i] <- posterior$climate_ratio[i] * posterior$variance_ratio[i]  * posterior$parameter_ratio[i] * posterior$transfer_ratio[i] * posterior$variance_ratio_hi[i]')],
}
R_LOCATIONS = {'src/main/summary.c': [r'^static Rboolean rprod\(', r'LDOUBLE s = 1\.0;', r's \*= x\[k\];', r'if\(s > DBL_MAX\) \*value = R_PosInf;', r'else \*value = \(double\) s;'],
               'src/nmath/dnorm.c': [r'^double dnorm4\('], 'src/nmath/pnorm.c': [r'^double pnorm5\(']}


def bind_rcpp(cache):
    rows, which, math_lines = [], set(), set()
    for path in sorted(cache.glob('Rcpp_*.tar.gz')):
        with tarfile.open(path, 'r:gz') as archive:
            names = [m.name for m in archive.getmembers() if m.isfile()]
            raw = archive.extractfile(next(n for n in names if n.endswith('sugar/functions/which_min.h'))).read()
            math_text = archive.extractfile(next(n for n in names if n.endswith('sugar/functions/math.h'))).read().decode('utf-8', 'replace').split('\n')
        digest = hashlib.sha256(raw).hexdigest()
        line = next(i + 1 for i, text in enumerate(math_text) if re.match(r'VECTORIZED_MATH_1\(abs,\s*::fabs\)', text))
        which.add(digest)
        math_lines.add(line)
        rows.append({'file': path.name, 'tarball_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'which_min_h_sha256': digest, 'abs_is_fabs_at_line_of_math_h': line})
    return {'window': ['2018-04-23', '2023-07-07'], 'versions': rows, 'versions_read': len(rows), 'which_min_h_identical_in_all': len(which) == 1, 'abs_lines': sorted(math_lines),
            'rule_read_from_the_source': 'which_min compares with a strict <, so the first minimum wins; a missing value gives NA_INTEGER',
            'not_bound': 'the installed Rcpp of the historical run is unknown; a non-CRAN installation is not excluded and not checked'}


def main():
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    for rel, expected in SOURCES.items():
        raw = (base / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'{rel} differs from the frozen binding')
        lines = raw.decode('utf-8', 'replace').split('\n')
        for number, start in PINS[rel]:
            if not lines[number - 1].strip().startswith(start):
                raise ValueError(f'{rel}:{number} differs: {lines[number - 1]!r}')
    tar_path = Path(sys.argv[2])
    if hashlib.sha256(tar_path.read_bytes()).hexdigest() != TARBALL_SHA256:
        raise ValueError('R-3.5.0.tar.gz differs from the frozen binding')
    found = {}
    with tarfile.open(tar_path, 'r:gz') as archive:
        for name, marks in R_LOCATIONS.items():
            raw = archive.extractfile('R-3.5.0/' + name).read()
            text = raw.decode('latin-1').split('\n')
            where, start = [], 0  # the marks of a file follow one another
            for mark in marks:
                start = next(i for i in range(start, len(text)) if re.search(mark, text[i])) + 1
                where.append(start)
            found[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'lines': where}
    report = {'kind': 'source_binding_no_draw_no_compilation', 'date': '2026-11-09', 'sources_sha256': SOURCES, 'pinned_lines': {rel: [n for n, _ in pins] for rel, pins in PINS.items()},
              'active_definition': f'{F}:1485', 'why_active': 'the named arguments of mcmc_conduction.R:228 (tann_range, pann_range, tann_dists, pann_dists) exist only in the definition at line 1485',
              'd_truncnorm_provider': 'RcppDist; see literal_rtruncnorm_binding_2026_11_09 (0.1.1 in the window, header lines 142-152)', 'rcpp': bind_rcpp(Path(sys.argv[1]).resolve()), 'r_3_5_0': {'tarball_sha256': TARBALL_SHA256, 'locations': found},
              'reach': {'call': f'{C}:228 inside the loop of line 138, without a condition, so in every iteration and for Kinneret as well', 'reader': f'{C}:302, posterior$acc_prob'},
              'inputs_in_the_run': {'grid_length': 1000, 'matrix_rows': 'num_taxa', 'source': f'{P}:1890-1935; the numeric transfer functions are missing'}, 'random_state_used': False, 'draws_run': 0, 'compiled': False, 'historical_reproduction_ready': False}
    (OUT / 'source_binding.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(report['rcpp']['versions_read'], report['rcpp']['which_min_h_identical_in_all'], report['rcpp']['abs_lines'], found)


if __name__ == '__main__':
    main()
