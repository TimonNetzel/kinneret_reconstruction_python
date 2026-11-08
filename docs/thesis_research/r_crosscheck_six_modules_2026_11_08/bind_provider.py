"""Provider of rowCumsums (mcmc_functions_v1.6.R:485-486) and possible masks of base functions on the search path.

Reads only the CRAN cache written earlier (NAMESPACE copies of the mirror, tarballs in memory); nothing is downloaded,
built or executed. The search path is taken unchanged from r_runtime_crosscheck_2026_11_03/provider_analysis.json.
Compares R/rowCumsums and its C kernel across the 13 matrixStats versions of the window.
Writes provider_rowcumsums.json. Usage: bind_provider.py <cran cache folder> <cran_exports.json>
"""
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parent / 'r_runtime_crosscheck_2026_11_03/provider_analysis.json'
# every function the six literal spans call by name, besides rowCumsums
BASE_NAMES = ('rowSums', 'colMeans', 'sum', 'max', 'min', 'abs', 'seq', 'duplicated', 'which', 'matrix', 'array', 'rep',
              't', 'sort', 'length', 'ncol', 'nrow', 'diff', 'any', 'as.matrix', 'c', 'list', '%in%', '%*%')
NAMES = ('rowCumsums', *BASE_NAMES)


def exported(text):
    code = '\n'.join(line.split('#', 1)[0] for line in text.splitlines())
    explicit = set()
    for m in re.finditer(r'\bexport\s*\((.*?)\)', code, re.S):
        explicit.update(x.strip().strip('"\'`') for x in m.group(1).split(','))
    return explicit, re.findall(r'exportPattern\s*\(\s*["\']([^"\']+)["\']', code)


def tarball_names(path, pkg):
    """Top-level definitions of the checked names and the NAMESPACE, read in memory."""
    defined, ns = set(), ''
    with tarfile.open(path, 'r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if member.name == f'{pkg}/NAMESPACE':
                ns = tar.extractfile(member).read().decode('utf-8', 'replace')
            elif re.match(re.escape(pkg) + r'/R/[^/]+\.[Rr]$', member.name):
                text = tar.extractfile(member).read().decode('utf-8', 'replace')
                for n in NAMES:
                    # column 0 only: an indented `rep = function` is a list element or argument, not a definition
                    if re.search(r'^[`"\']?' + re.escape(n) + r'[`"\']?\s*(<-|=)\s*function', text, re.M):
                        defined.add(n)
    explicit, patterns = exported(ns)
    return {n for n in NAMES if n in explicit or (n in defined and any(
        re.search(p.replace('[[:alpha:]]', '[A-Za-z]'), n) for p in patterns))}, patterns


def normal(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    return '\n'.join(' '.join(line.split()) for line in text.replace('\r\n', '\n').split('\n') if line.strip())


def proxy_as_matrix(cache, exports):
    """How proxy, which exports as.matrix, defines it in each window version (tarballs read in memory)."""
    rows = []
    for v in exports['packages']['proxy']['window_versions']:
        with tarfile.open(cache / 'tarballs' / f"proxy_{v['version']}.tar.gz", 'r:gz') as tar:
            found = []
            for m in tar.getmembers():
                if m.isfile() and re.match(r'proxy/R/[^/]+\.[Rr]$', m.name):
                    text = tar.extractfile(m).read().decode('utf-8', 'replace')
                    lines = text.splitlines()
                    found += [' '.join(x.strip() for x in lines[i:i + 3]) for i, line in enumerate(lines)
                              if re.match(r'^as\.matrix\s*(<-|=)', line)]
            ns = tar.extractfile('proxy/NAMESPACE').read().decode('utf-8', 'replace')
        rows.append({'version': v['version'], 'definitions': found,
                     'import_from_base': bool(re.search(r'importFrom\s*\(\s*["\']?base', ns)),
                     'exports_as_matrix': '"as.matrix"' in ns or 'as.matrix,' in ns})
    return rows


def main():
    cache, exports = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    analysis = json.loads(ANALYSIS.read_text(encoding='utf-8'))
    path = analysis['search_path_front_to_back']
    hits, unresolved = {}, {}
    for pkg in path:
        for v in exports['packages'][pkg]['window_versions']:
            ns_file = cache / 'mirror' / pkg / v['version'] / 'NAMESPACE'
            tar_file = cache / 'tarballs' / f"{pkg}_{v['version']}.tar.gz"
            if v.get('mirror') and ns_file.exists():
                explicit, patterns = exported(ns_file.read_text(encoding='utf-8'))
                found = {n for n in NAMES if n in explicit}
                if patterns and tar_file.exists():
                    found, patterns = tarball_names(tar_file, pkg)
                elif patterns:
                    unresolved.setdefault(pkg, []).append(v['version'])
            elif tar_file.exists():
                found, _ = tarball_names(tar_file, pkg)
            else:
                unresolved.setdefault(pkg, []).append(v['version'])
                continue
            for n in found:
                hits.setdefault(n, {}).setdefault(pkg, []).append(v['version'])
    versions = [v['version'] for v in exports['packages']['matrixStats']['window_versions']]
    r_groups, c_groups, signatures = {}, {}, {}
    kernels = {}
    for version in versions:
        with tarfile.open(cache / 'tarballs' / f'matrixStats_{version}.tar.gz', 'r:gz') as tar:
            members = {m.name.split('/', 1)[1]: m for m in tar.getmembers() if m.isfile()}
            read = lambda name: tar.extractfile(members[name]).read().decode('utf-8', 'replace')
            r_text = read('R/rowCumsums.R') if 'R/rowCumsums.R' in members else read('R/rowCumsums.matrix.R')
            template = read('src/rowCumsums_lowlevel_template.h')
        signature = re.search(r'rowCumsums\s*<-\s*function\s*\((.*?)\)\s*\{', r_text, re.S)
        signatures.setdefault(' '.join(signature.group(1).split()), []).append(version)
        r_groups.setdefault(hashlib.sha256(normal(r_text).encode()).hexdigest()[:12], []).append(version)
        kernel = normal(template)
        c_groups.setdefault(hashlib.sha256(kernel.encode()).hexdigest()[:12], []).append(version)
        kernels[version] = kernel
    front = path[:path.index('matrixStats')]
    arithmetic = {v: [line for line in k.split('\n') if re.search(r'\bvalue\s*=|ans\[kk\]\s*=|^LDOUBLE', line)]
                  for v, k in kernels.items()}
    arith_groups = {}
    for v, lines in arithmetic.items():
        arith_groups.setdefault(hashlib.sha256('\n'.join(lines).encode()).hexdigest()[:12], []).append(v)
    result = {
        'kind': 'rowCumsums_provider_and_base_mask_analysis_not_runtime_binding',
        'window': analysis['window'], 'search_path_source': 'r_runtime_crosscheck_2026_11_03/provider_analysis.json',
        'rowCumsums_exported_by': hits.get('rowCumsums', {}),
        'exporters_in_front_of_matrixStats': sorted(p for p in hits.get('rowCumsums', {}) if p in front),
        'base_names_exported_by_attached_packages': {n: {p: len(v) for p, v in hits[n].items()}
                                                     for n in BASE_NAMES if n in hits},
        'unresolved_versions': unresolved,
        'matrixStats_versions': versions, 'signatures': signatures,
        'na_rm_is_a_formal': False, 'r_text_groups': r_groups, 'normalised_c_template_groups': c_groups,
        'c_template_uses_long_double': {v: 'LDOUBLE' in k for v, k in kernels.items()},
        'arithmetic_statement_groups': arith_groups,
        'proxy_as_matrix': proxy_as_matrix(cache, exports),
        'historical_reproduction_ready': False,
    }
    if any('na.rm' in s for s in signatures):
        result['na_rm_is_a_formal'] = True
    (OUT / 'provider_rowcumsums.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8', newline='\n')
    print({k: result[k] for k in ('exporters_in_front_of_matrixStats', 'base_names_exported_by_attached_packages',
                                  'unresolved_versions', 'signatures', 'na_rm_is_a_formal')})
    print('rowCumsums exported by', {k: len(v) for k, v in result['rowCumsums_exported_by'].items()})
    print('R groups', r_groups)
    print('C groups', c_groups)
    print('LDOUBLE', set(result['c_template_uses_long_double'].values()))
    print('arithmetic groups', arith_groups)
    print('proxy', result['proxy_as_matrix'])


if __name__ == '__main__':
    main()
