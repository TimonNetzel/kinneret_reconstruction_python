"""Source and binding check of rtruncnorm_cpp (mcmc_functions_v1.6.R:1556-1570, 1812-1813): no draw, no compilation, no run of the sampler.

Reads the frozen originals, the CRAN tarballs of RcppDist and the R 3.5.0 tarball. Archives are read in memory, nothing is extracted or executed.
Only versions, dates, hashes, line numbers and counts of function names are stored; RcppDist and R are GPL, no text of theirs enters this repository.
A version counts when it was the current one at any time inside the window: every version published inside it plus the last one published before it.
Writes rcppdist_versions.json and source_binding.json. Usage: bind_rtruncnorm.py <cache dir of RcppDist tarballs> <R-3.5.0.tar.gz>
"""
import hashlib
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
WINDOW = ('2018-04-23', '2023-07-07')  # window of r_runtime_crosscheck_2026_11_03: R 3.5.0 to the last source change
BASE = 'https://cran.r-project.org/src/contrib/'
PACKAGE = 'RcppDist'
HEADER = 'RcppDist/inst/include/truncnorm.h'
LISTING = re.compile(r'href="([A-Za-z0-9.]+_[0-9.\-]+\.tar\.gz)".*?(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}')
TARBALL_SHA256 = 'fd1725535e21797d3d9fea8963d99be0ba4c3aecadcf081b43e261458b416870'
F, P, M, C = 'src/mcmc_functions_v1.6.R', 'src/mcmc_prepare.R', 'main_v1.2.R', 'src/mcmc_conduction.R'
SOURCES = {M: '82486f99bafff08af2b51475ed225501d26cec1a0322a286b97a248ebd2f0520',
           F: 'b887144370e328fa016d1834ceebcd55622816d4b36838d33a26197110ab453c',
           P: 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e',
           C: '47c59c508e15150f7d6fda775a4f6bf6c14f0148e10cefd0014fa1ce6cfe5a9d'}
PINS = {  # (line, start of the stripped line)
    M: [(29, 'library(Rcpp)'), (31, 'library(RcppZiggurat)'), (644, 'source("src/mcmc_functions_v1.6.R")'), (675, 'source("src/mcmc_prepare.R")'),
        (676, 'source("src/mcmc_conduction.R")'), (1055, 'set.seed(2019)')],
    F: [(1556, 'sourceCpp(code = "'), (1557, '#include <Rcpp.h>'), (1558, '#include <truncnorm.h>'), (1560, '// [[Rcpp::depends(RcppDist)]]'), (1561, '// [[Rcpp::export]]'),
        (1563, 'NumericVector rtruncnorm_cpp(int n,NumericVector mean, NumericVector sd, NumericVector a, NumericVector b) {'), (1564, 'NumericVector out(n);'),
        (1565, 'for(int i=0; i<n; i++) {'), (1566, 'out(i) = r_truncnorm(mean(i), sd(i), a(i), b(i));'), (1568, 'return out;'),
        (1812, 'posterior_temp$tf_sample[,1] <- rtruncnorm_cpp(num_taxa,posterior$tf_sample[[i-1]][,1],sds_tann,tf_quarts_tann[,1], tf_quarts_tann[,3])'),
        (1813, 'posterior_temp$tf_sample[,2] <- rtruncnorm_cpp(num_taxa,posterior$tf_sample[[i-1]][,2],sds_pann,tf_quarts_pann[,1], tf_quarts_pann[,3])'),
        (1354, '// [[Rcpp::depends(RcppDist)]]'), (1362, 'propose_tann(i,j) = r_truncnorm('), (1363, 'propose_pann(i,j) = r_truncnorm('),
        (1437, 'List ratios_tfs_no_ages(int num_taxa, NumericVector new_tann,'), (1483, '// [[Rcpp::depends(RcppDist)]]'),
        (1485, 'List ratios_tfs_no_ages(int num_taxa, NumericVector tann_range,NumericVector pann_range,'),
        (1516, 'propose_dists_tann = d_truncnorm(old_tann[i],new_tann[i], sds_tann_trunc[i], lower_tann[i], upper_tann[i]);'),
        (1517, 'start_dists_tann = d_truncnorm(new_tann[i],old_tann[i], sds_tann_trunc[i], lower_tann[i], upper_tann[i]);'),
        (1520, 'propose_dists_pann = d_truncnorm(old_pann[i],new_pann[i], sds_pann_trunc[i], lower_pann[i], upper_pann[i]);'),
        (1521, 'start_dists_pann = d_truncnorm(new_pann[i],old_pann[i], sds_pann_trunc[i], lower_pann[i], upper_pann[i]);')],
    P: [(1886, 'library(truncnorm)'), (1896, 'set.seed(2023)'), (1897, 'tf_tann_sample <- sample(x1range.bck,size=10000,prob = tf.x1.current[i,],replace = T)'), (1898, 'set.seed(2023)'),
        (1899, 'tf_pann_sample <- sample(x2range.bck,size=10000,prob = tf.x2.current[i,],replace = T)'), (1913, 'tf_quarts_tann[i,] <- quantile(tf_tann_sample,c(0.05,0.5,0.95))'),
        (1914, 'tf_quarts_pann[i,] <- c(0,quantile(tf_pann_sample,c(0.5,0.95)))'), (1922, 'sds_tann[i] <- sd(tf_tann_sample)*(2.38^2/(2*(num_taxa)))'),
        (1923, 'sds_pann[i] <- sd(tf_pann_sample)*(2.38^2/(2*(num_taxa)))'), (1945, 'posterior$tf_sample[[1]][,1] <- tf_quarts_tann[,2]'), (1946, 'posterior$tf_sample[[1]][,2] <- tf_quarts_pann[,2]'),
        (1957, 'posterior$tf_sample_tann[[1]][j,] <- rtruncnorm(1, tf_quarts_tann[j,1], tf_quarts_tann[j,3],tf_quarts_tann[j,2],sds_tann[j])'),
        (1958, 'posterior$tf_sample_pann[[1]][j,] <- rtruncnorm(1, tf_quarts_pann[j,1], tf_quarts_pann[j,3],tf_quarts_pann[j,2],sds_pann[j])'),
        (1979, 'post_tf_samples[,j-1] <- rnorm(10000,means[ct],sds[ct])'), (1980, 'post_tf_samples[,j] <- rgamma(10000,shape=shapes[ct],rate=rates[ct])')],
    C: [(138, 'for(i in 2:sample_length){'), (146, 'posterior_temp <- reconstruction(posterior_temp=posterior_temp,'), (228, 'ratios_tfs <- ratios_tfs_no_ages(num_taxa, tann_range = tann_range,pann_range = pann_range,')],
}
TOKENS_RANDOM = ('R::runif', 'R::rnorm', 'R::rexp', 'R::rgamma', 'unif_rand', 'norm_rand', 'exp_rand', 'Rcpp::runif', 'Rcpp::rnorm')
TOKENS_OTHER = ('R::pnorm', 'R::qnorm', 'R::dnorm', 'q_truncnorm(')
FUNCTIONS = ('d_truncnorm', 'q_truncnorm', 'r_truncnorm')
R_LOCATIONS = {'src/nmath/runif.c': [r'^double runif\(double a, double b\)', r'if \(a == b\)', r'unif_rand\(\)'],
               'src/main/RNG.c': [r'^static double fixup\(double x\)', r'if\(x <= 0\.0\)', r'if\(\(1\.0 - x\) <= 0\.0\)', r'return fixup\(MT_genrand\(\)\)', r'RNG_DEFAULT\s+MERSENNE_TWISTER', r'RNG_kind = RNG_DEFAULT'],
               'src/nmath/qnorm.c': [r'^double qnorm5\('], 'src/nmath/pnorm.c': [r'^double pnorm5\('], 'src/nmath/dnorm.c': [r'^double dnorm4\(']}


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'kinneret-reconstruction-source-check'})
    with urllib.request.urlopen(request, timeout=120) as reply:
        return reply.read()


def code(line):
    return line.split('#', 1)[0].strip() if not line.lstrip().startswith('#include') else line.strip()


def listing():
    rows = [(n, d, f'Archive/{PACKAGE}/') for n, d in LISTING.findall(fetch(f'{BASE}Archive/{PACKAGE}/').decode('utf-8', 'replace'))]
    rows += [(n, d, '') for n, d in LISTING.findall(fetch(BASE).decode('utf-8', 'replace')) if n.startswith(PACKAGE + '_')]
    return sorted(rows, key=lambda row: row[1])


def spans(lines):
    found = {}
    for name in FUNCTIONS:
        start = next(i for i, text in enumerate(lines) if re.match(rf'inline double {name}\(', text))
        end = next(i for i in range(start, len(lines)) if lines[i] == '}')
        body = '\n'.join(lines[start:end + 1])
        found[name] = {'first_line': start + 1, 'last_line': end + 1, 'calls': {t: body.count(t) for t in TOKENS_RANDOM + TOKENS_OTHER if body.count(t)}}
    return found


def inspect(path):
    with tarfile.open(path, 'r:gz') as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        raw = archive.extractfile(HEADER).read()
        description = archive.extractfile('RcppDist/DESCRIPTION').read().decode('utf-8', 'replace')
        carriers, defining = [], []
        for member in members:
            parts = member.name.split('/')
            if parts[-1].lower() == 'truncnorm.h':
                carriers.append(member.name)
            if len(parts) > 2 and parts[1] in ('inst', 'src') and parts[-1].lower().endswith(('.h', '.hpp', '.cpp', '.c')):
                if re.search(r'^\s*(inline\s+)?double\s+r_truncnorm\s*\(', archive.extractfile(member).read().decode('latin-1'), re.M):
                    defining.append(member.name)
    field = lambda key: re.search(rf'^{key}:\s*(.+)$', description, re.M).group(1).strip()
    text = raw.decode('utf-8').replace('\r\n', '\n').split('\n')
    return {'version': field('Version'), 'date_publication': field('Date/Publication'), 'license': field('License'), 'tarball_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'header_member': HEADER, 'header_sha256': hashlib.sha256(raw).hexdigest(), 'header_lines': len(text), 'members_named_truncnorm_h': carriers, 'members_defining_r_truncnorm': defining,
            'functions': spans(text)}


def bind_cran(cache):
    rows = listing()
    if not rows:
        raise ValueError('no versions listed')
    keep = [r for r in rows if WINDOW[0] <= r[1] <= WINDOW[1]]
    before = [r for r in rows if r[1] < WINDOW[0]]
    results, outside = [], []
    for name, date, prefix in rows:
        target = cache / name
        if not target.exists():
            target.write_bytes(fetch(BASE + prefix + name))
        entry = {'file': name, 'listing_date': date, **inspect(target)}
        (results if (name, date, prefix) in before[-1:] + keep else outside).append(entry)
    every = results + outside
    report = {'kind': 'cran_source_binding_of_r_truncnorm_no_text_stored', 'package': PACKAGE, 'window': WINDOW, 'versions_listed_on_cran': [r[0] for r in rows], 'versions_in_window': results,
              'versions_outside_window_for_comparison': outside, 'last_version_before_window': before[-1][0] if before else None,
              'first_publication_on_cran': rows[0][1], 'window_days_without_any_cran_version': rows[0][1] > WINDOW[0],
              'header_identical_in_all_cran_versions': len({v['header_sha256'] for v in every}) == 1,
              'source': 'CRAN source archives, read as archives only, nothing extracted or executed',
              'not_bound': 'the installed version of the historical run is unknown; a non-CRAN installation (for example from GitHub) is not excluded and not checked'}
    return report


def bind_diss():
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    base = Path(cfg['original_data_root'])
    lines = {}
    for rel, expected in SOURCES.items():
        raw = (base / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'{rel} differs from the frozen binding')
        lines[rel] = raw.decode('utf-8', 'replace').split('\n')
    for rel, pins in PINS.items():
        for number, start in pins:
            if not lines[rel][number - 1].strip().startswith(start):
                raise ValueError(f'{rel}:{number} differs: {lines[rel][number - 1]!r}')
    def search(rel, pattern):
        return [i + 1 for i, text in enumerate(lines[rel]) if re.search(pattern, code(text))]
    definitions = []
    seen = set()
    for path in base.rglob('*.R'):
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        for i, text in enumerate(path.read_bytes().decode('latin-1').split('\n')):
            if re.search(r'NumericVector\s+rtruncnorm_cpp\s*\(', code(text)):
                definitions.append(f'{path.relative_to(base).as_posix()}:{i + 1}')
    return {
        'sources_sha256': SOURCES, 'pinned_lines': {rel: [n for n, _ in pins] for rel, pins in PINS.items()}, 'r_files_scanned_for_definitions': len(seen),
        'definitions_of_rtruncnorm_cpp_in_source_tree': definitions,
        'calls': {'rtruncnorm_cpp': {f'{F}': [n for n in search(F, r'\brtruncnorm_cpp\s*\(') if n != 1563], C: search(C, r'\brtruncnorm_cpp\s*\('), P: search(P, r'\brtruncnorm_cpp\s*\(')},
                  'propose_cpp_uncommented': {F: [n for n in search(F, r'\bpropose_cpp\s*\(') if n != 1357], C: search(C, r'\bpropose_cpp\s*\(')},
                  'ratios_tfs_no_ages_uncommented': {C: search(C, r'\bratios_tfs_no_ages\s*\(')},
                  'r_package_truncnorm_rtruncnorm_uncommented': {P: search(P, r'\brtruncnorm\s*\(')}},
        'seeds_uncommented': {rel: search(rel, r'\bset\.seed\s*\(') for rel in SOURCES},
        'rngkind_or_rngversion_uncommented': {rel: search(rel, r'\b(RNGkind|RNGversion)\s*\(') for rel in SOURCES},
    }


def bind_r(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != TARBALL_SHA256:
        raise ValueError('R-3.5.0.tar.gz differs from the frozen binding')
    found = {}
    with tarfile.open(path, 'r:gz') as archive:
        for name, marks in R_LOCATIONS.items():
            raw = archive.extractfile('R-3.5.0/' + name).read()
            text = raw.decode('latin-1').split('\n')
            found[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'lines': [next(i + 1 for i, line in enumerate(text) if re.search(mark, line)) for mark in marks]}
    return {'tarball_sha256': TARBALL_SHA256, 'locations': found}


def main():
    cache = Path(sys.argv[1]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    cran = bind_cran(cache)
    (OUT / 'rcppdist_versions.json').write_text(json.dumps(cran, indent=1) + '\n', encoding='utf-8', newline='\n')
    binding = {'kind': 'source_binding_no_draw_no_compilation', 'date': '2026-11-09', 'window': WINDOW, **bind_diss(), 'r_3_5_0': bind_r(Path(sys.argv[2])),
               'random_state_used': False, 'draws_run': 0, 'compiled': False, 'historical_reproduction_ready': False}
    (OUT / 'source_binding.json').write_text(json.dumps(binding, indent=1) + '\n', encoding='utf-8', newline='\n')
    print([(v['file'], v['listing_date'], v['date_publication'], v['header_sha256'][:12]) for v in cran['versions_in_window']],
          [(v['file'], v['header_sha256'][:12]) for v in cran['versions_outside_window_for_comparison']], cran['header_identical_in_all_cran_versions'])
    print(json.dumps({k: binding[k] for k in ('definitions_of_rtruncnorm_cpp_in_source_tree', 'calls', 'seeds_uncommented', 'rngkind_or_rngversion_uncommented', 'r_files_scanned_for_definitions')}))
    print(json.dumps(cran['versions_in_window'][0]['functions']))
    print(json.dumps(binding['r_3_5_0']['locations']))


if __name__ == '__main__':
    main()
