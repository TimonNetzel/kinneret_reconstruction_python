"""Bind the weightedMedian calls of mcmc_prepare.R, the matrixStats kernel, R's sort routines and the installed binary.

Reads only. mcmc_prepare.R must match the SHA-256 of the earlier frozen binding. matrixStats and R sources are read from
the CRAN tarballs in memory, the binary zip is only hashed and its DESCRIPTION read. The history of R's sort files inside
the window comes from the GitHub mirror wch/r-source via ``gh api``; the file text at the end of the window is compared
with R 3.5.0. Writes source_binding.json, source_excerpt.json, determinism_audit.json, binary_package.json and
r_sort_source.json. Usage: bind_source.py <matrixStats tarball folder> <R-3.5.0.tar.gz> <matrixStats zip> <headers.txt>
"""
import base64
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASIS = 'cfd0ad8f1c0cb3f06909f50f21cc1e9a65592a68'
SOURCE = ('src/mcmc_prepare.R', 'b897e84f8f2d72399de729592066341585974d521bf8164b185f20377ed5cc7e')
LITERAL = [
    (1412, 'tann_recent[count] <- weightedMedian(x1range.bck,tann.recon.age)'),
    (1413, 'pann_recent[count] <- weightedMedian(x2range.bck,pann.recon.age)'),
    (1462, 'taxa_change <- FALSE'),
    (1623, 'tf.x1.current.mean <- (tf.x1.current %*% x1range.bck)[,1] # expectation value'),
    (1624, 'tf.x2.current.mean <- (tf.x2.current %*% x2range.bck)[,1] # expectation value'),
    (1628, 'tf.x1.current.mean <- tf.x1.current.mean_change <- c()'),
    (1629, 'for(j in 1:nrow(tf.x1.current) ){'),
    (1631, 'tf.x1.current.mean[j] <- weightedMedian(x1range.bck,tf.x1.current[j,] )'),
    (1632, 'if(taxa_change) tf.x1.current.mean_change[j] <- weightedMedian(x1range.bck,tf.x1.current_change[j,] )'),
    (1640, '}'),
    (1648, 'tf.x2.current.mean <- tf.x2.current.mean_change <- c()'),
    (1649, 'for(j in 1:nrow(tf.x2.current) ){'),
    (1651, 'tf.x2.current.mean[j] <- weightedMedian(x2range.bck,tf.x2.current[j,] )'),
    (1652, 'if(taxa_change) tf.x2.current.mean_change[j] <- weightedMedian(x2range.bck,tf.x2.current_change[j,] )'),
    (1660, '}'),
    (1664, 'which.tf.missing <- which(is.na(tf.x1.current.mean))'),
]
KERNEL_MUST = [
    'if (ISNAN(weight)) {', 'if (!narm) {', '} else if (weight <= 0) {', '} else if (isinf(weight)) {',
    'if (isinf(weight)) {', 'wtmp[jj] = 1;', 'if (X_ISNAN(value)) {', 'if (nxt == 0) {', 'wtotal += wtmp[jj];',
    'if (nx == 1) {', 'half = (nx+1) / 2;', 'X_PSORT(x, nx, half);', 'res = (double)x[half-1];',
    'X_PSORT(x, half, half-1);', 'X_QSORT_I(x, idxs_int, 1, nx);', 'half = nx+1; /* Default is last */',
    'tmp_d = w[idxs_int[ii]] / wtotal;', 'tmp_d2 += tmp_d;', 'wcum[ii] = tmp_d2 - (tmp_d/2);', 'if (wcum[ii] >= 0.5) {',
    'if (half == 0 || half == nx) {', 'dx = (double)(x[half] - x[half-1]);', 'Dy = wcum[half] - wcum[half-1];',
    'dy = 0.5 - wcum[half];', 'dx = (dy/Dy) * dx;', 'res = dx + x[half];',
]
AUDIT = {
    'kind': 'determinism_audit_from_source_text_and_R_cross_check',
    'question': 'Is the NA mask of which.tf.missing deterministic once weightedMedian is bound?',
    'candidates': [
        {'id': 'weighted_median_1631_1651', 'lines': 'mcmc_prepare.R:1628-1640,1648-1660,1664', 'random_state': False,
         'package_semantics_needed': True,
         'package_semantics_bound': 'matrixStats::weightedMedian, first on the search path within 2018-04-23..'
                                    '2023-07-07; one normalised C kernel in all 13 versions (helper_comparison.json)',
         'functions': ['weightedMedian (matrixStats C kernel)', 'R_qsort_I and rPsort (R)', 'is.na', 'which'],
         'inputs': 'x1range.bck, x2range.bck and tf.x1/x2.current as EXPLICIT inputs; their numeric values are not '
                   'recovered (the transfer-function container is missing)',
         'readers': 'tf.x1.current.mean tested in 1664; both means carried by 1669/1670 and 1693/1694',
         'decision': 'implemented as tools/historical_weighted_median.py; inputs only synthetic'},
        {'id': 'expectation_1623_1624', 'lines': 'mcmc_prepare.R:1623-1624', 'random_state': False,
         'evidence': 'overwritten by c() in 1628 and 1648 before any reader', 'decision': 'no reader, not implemented'},
        {'id': 'spline_expansion_1627_1647', 'lines': 'mcmc_prepare.R:1627,1647', 'random_state': False,
         'evidence': 'tann_expand and pann_expand are only read in comments', 'decision': 'no reader, not implemented'},
        {'id': 'other_calls_1412_1413', 'lines': 'mcmc_prepare.R:1412-1413', 'random_state': False,
         'evidence': 'same function, different inputs, not part of the mask', 'decision': 'not implemented here'},
    ],
    'selected_for_implementation': 'weighted_median_1631_1651',
    'final_variant_selected': False, 'R_execution_for_binding': False,
}


def gh(*args):
    return subprocess.run(['gh', 'api', *args], check=True, capture_output=True, stdin=subprocess.DEVNULL).stdout


def history(path):
    data = json.loads(gh(f'repos/wch/r-source/commits?path={path}&since=2018-04-23T00:00:00Z&until=2023-07-07T23:59:59Z'
                         '&per_page=100'))
    return [{'sha': c['sha'], 'date': c['commit']['committer']['date'], 'subject': c['commit']['message'].split('\n')[0]}
            for c in data]


def at_commit(path, sha):
    return base64.b64decode(json.loads(gh(f'repos/wch/r-source/contents/{path}?ref={sha}'))['content']).decode('utf-8')


def psort_text(sort_c):
    start = sort_c.index('#define psort_body')
    return sort_c[start:sort_c.index('static void iPsort2')]


def rcmp_text(sort_c):
    return re.search(r'static int rcmp\(double x, double y, Rboolean nalast\).*?\n}\n', sort_c, re.S).group(0)


def main():
    tarballs, r_tar, zip_path, headers = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    spec = importlib.util.spec_from_file_location('ws_bind', OUT.parent / 'literal_weights_stage_2026_11_03/bind_source.py')
    binder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binder)
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    raw = (Path(cfg['original_data_root']) / SOURCE[0]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE[1]:
        raise ValueError('mcmc_prepare.R differs from the frozen binding')
    lines = raw.decode('utf-8-sig').splitlines()
    for n, text in LITERAL:
        if lines[n - 1].strip() != text:
            raise ValueError(f'mcmc_prepare.R:{n} no longer reads {text!r}')
    code = binder._code(lines)
    depth = binder._depths(code)
    expect_depth = {1628: 0, 1629: 0, 1631: 1, 1648: 0, 1649: 0, 1651: 1, 1664: 0}
    if {n: depth[n - 1] for n in expect_depth} != expect_depth:
        raise ValueError(f'brace depths differ: { {n: depth[n - 1] for n in expect_depth} }')
    expand_readers = [n for n, line in enumerate(code, 1) if re.search(r'\b(tann|pann)_expand\b', line)]
    if expand_readers != [1627, 1647]:
        raise ValueError(f'spline results now have readers: {expand_readers}')
    mean_writers = [n for n, line in enumerate(code, 1) if re.search(r'\btf\.x[12]\.current\.mean\s*(\[[^]]*\])?\s*<-', line)]
    if mean_writers != [1623, 1624, 1628, 1631, 1648, 1651, 1669, 1670]:
        raise ValueError(f'writers of the means differ: {mean_writers}')
    other_taxa_change = [n for n, line in enumerate(code, 1) if re.search(r'\btaxa_change\s*<-', line)]
    if other_taxa_change != [1462, 1501]:
        raise ValueError(f'taxa_change writers differ: {other_taxa_change}')

    provider = json.loads((OUT.parent / 'r_runtime_crosscheck_2026_11_03/provider_analysis.json').read_text(encoding='utf-8'))
    wm_provider = provider['providers_bound_within_window']['weightedMedian']
    if wm_provider != 'matrixStats' or provider['window'] != ['2018-04-23', '2023-07-07']:
        raise ValueError('the provider of weightedMedian is no longer matrixStats within the window')
    helper = json.loads((OUT / 'helper_comparison.json').read_text(encoding='utf-8'))
    if len(helper['normalised_kernel_groups']) != 1 or len(helper['versions']) != 13:
        raise ValueError('the normalised kernel is no longer one text across the 13 versions')

    with tarfile.open(tarballs / 'matrixStats_0.56.0.tar.gz', 'r:gz') as tar:
        kernel = tar.extractfile('matrixStats/src/weightedMedian_lowlevel_template.h').read().decode('utf-8')
    missing = [m for m in KERNEL_MUST if m not in kernel]
    if missing:
        raise ValueError(f'kernel statements differ: {missing}')

    data = r_tar.read_bytes()
    r_files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        for name in ('qsort-body.c', 'qsort.c', 'sort.c'):
            r_files[name] = tar.extractfile(f'R-3.5.0/src/main/{name}').read().decode('utf-8')
    must_r = ['double R = 0.375;', 'if (R < 0.5898437) R += 0.0390625; else R -= 0.21875;',
              'ij = (INTt)(i + (INTt)((j - i)*R));', 'if (j - i > 10)\tgoto L20;', 'if (i == ii)\tgoto L10;']
    if any(m not in r_files['qsort-body.c'] for m in must_r):
        raise ValueError('R_qsort_I statements differ')
    hist = {name: history(f'src/main/{name}') for name in ('qsort-body.c', 'qsort.c', 'sort.c')}
    end_sort = at_commit('src/main/sort.c', hist['sort.c'][0]['sha'])
    end_body = at_commit('src/main/qsort-body.c', hist['qsort-body.c'][0]['sha'])
    psort_same = psort_text(end_sort) == psort_text(r_files['sort.c'])
    rcmp_same = rcmp_text(end_sort) == rcmp_text(r_files['sort.c'])
    a, b = r_files['qsort-body.c'].split('\n'), end_body.split('\n')
    body_diff = [(x, y) for x, y in zip(a, b) if x != y] if len(a) == len(b) else None
    if not (psort_same and rcmp_same and body_diff == [('    --I;', '    if(I) --I; // precaution for use in package bnlearn')]):
        raise ValueError(f'R sort code changed inside the window: psort {psort_same}, rcmp {rcmp_same}, body {body_diff}')
    (OUT / 'r_sort_source.json').write_text(json.dumps({
        'kind': 'R_sort_routines_source_binding_not_runtime_binding',
        'R_tarball': {'url': 'https://cran.r-project.org/src/base/R-3/R-3.5.0.tar.gz', 'size': len(data),
                      'sha256': hashlib.sha256(data).hexdigest(), 'md5': hashlib.md5(data).hexdigest()},
        'files_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in r_files.items()},
        'history_in_window_wch_r_source': hist,
        'psort_body_identical_at_window_end': psort_same, 'rcmp_identical_at_window_end': rcmp_same,
        'qsort_body_only_change': body_diff[0], 'qsort_body_change_affects_this_call': False,
        'note': 'R_qsort_I(x, idxs_int, 1, nx) passes a non-NULL index, so the only change in the window is not reached '
                'by another code path; qsort.c changed only in comments and attribute placement according to the '
                'commit subjects.',
    }, indent=1) + '\n', encoding='utf-8', newline='\n')

    zipped = zip_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(zipped)) as z:
        description = z.read('matrixStats/DESCRIPTION').decode('utf-8').replace('\r\n', '\n')
    fields = dict(re.findall(r'^(Version|Packaged|Built|Date/Publication|Archs): (.*)$', description, re.M))
    head = dict(re.findall(r'^(Last-Modified|Content-Length): (.*?)\s*$', headers.read_text(encoding='latin-1'), re.M))
    if fields['Version'] != '0.56.0' or '0.56.0' not in helper['versions']:
        raise ValueError('binary package is not a window version')
    (OUT / 'binary_package.json').write_text(json.dumps({
        'kind': 'matrixStats_windows_binary_recorded_before_installation',
        'url': 'https://cran-archive.r-project.org/bin/windows/contrib/3.5/matrixStats_0.56.0.zip',
        'size': len(zipped), 'sha256': hashlib.sha256(zipped).hexdigest(), 'md5': hashlib.md5(zipped).hexdigest(),
        'server_last_modified': head['Last-Modified'], 'description': fields,
        'source_group_in_window': next(k for k, v in json.loads((OUT.parent / 'r_runtime_crosscheck_2026_11_03/'
                                       'weighted_median_versions.json').read_text(encoding='utf-8'))['groups'].items()
                                       if '0.56.0' in v),
        'install_folder': '.local/R/library_ms', 'installer_report': "unpacked with matching MD5 sums",
        'note': 'R 3.5 has exactly one binary in the CRAN archive; built under R 3.5.3, loaded in R 3.5.0 with a warning',
    }, indent=1) + '\n', encoding='utf-8', newline='\n')

    spans = [
        {'source': 'original:src/mcmc_prepare.R', 'first_line': 1623, 'last_line': 1664,
         'role': 'the two weightedMedian loops and the mask', 'text': '\n'.join(lines[1622:1664])},
        {'source': 'CRAN:matrixStats 0.56.0 src/weightedMedian_lowlevel_template.h (Artistic-2.0)', 'first_line': None,
         'last_line': None, 'role': 'statements that decide NA and the value, not the whole file', 'text': '\n'.join(KERNEL_MUST)},
    ]
    (OUT / 'source_excerpt.json').write_text(json.dumps({'status': 'read_only_spans', 'spans': spans}, indent=2,
                                                        ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'source_binding.json').write_text(json.dumps({
        'status': 'bounded_historical_source_fragment_not_case_reproduction', 'basis_commit': BASIS,
        'source_sha256': {SOURCE[0]: SOURCE[1]},
        'provider_evidence': 'r_runtime_crosscheck_2026_11_03/provider_analysis.json', 'provider_record': wm_provider,
        'versions_compared': 13, 'normalised_kernel_groups': 1,
        'implemented_span': [SOURCE[0], [1628, 1640], [1648, 1660], [1664, 1664]],
        'brace_depth_at_line_start': {str(k): v for k, v in expect_depth.items()},
        'mean_writers': mean_writers, 'spline_only_written': expand_readers, 'taxa_change_writers': other_taxa_change,
        'only_x1_tested_in_1664': True, 'portable_excerpt': 'source_excerpt.json',
        'portable_excerpt_sha256': hashlib.sha256((OUT / 'source_excerpt.json').read_bytes()).hexdigest(),
        'supported_domain': 'double x and w of equal length, NA as None, NaN and +-Inf allowed; idxs = NULL, na.rm = '
                            'FALSE, interpolate = TRUE, ties = NULL; an overflowing total weight is rejected',
        'x_type_note': 'x1range.bck comes from the missing transfer-function container; double is assumed from the call '
                       'and from all_in_all.R (seq and matrix means), an integer x would use the integer kernel',
        'actual_original_inputs_used_for_numerics': False, 'run_binding': None, 'historical_reproduction_ready': False,
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'determinism_audit.json').write_text(json.dumps(AUDIT, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('bound; depth', {n: depth[n - 1] for n in expect_depth}, 'mean writers', mean_writers, 'provider', wm_provider)
    print('R sort unchanged in window:', psort_same, rcmp_same, body_diff)


if __name__ == '__main__':
    main()
