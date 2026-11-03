"""Which attached package provides rdirichlet, ddirichlet and weightedMedian on the search path of main_v1.2.R?

Input: the cached CRAN mirror files written by collect_cran_exports.py (NAMESPACE and DESCRIPTION per version available
between R 3.5.0, 2018-04-23, and the last source change, 2023-07-07). A package with exportPattern() is checked by
its R sources only if it matters (it is listed as unresolved otherwise). The search path follows the library() order
bound in literal_weights_stage_2026_11_03/source_binding.json, each package preceded by the Depends it attaches
(R attaches Depends before the package itself, so the package sits in front of them). Writes provider_analysis.json.
Usage: analyse_providers.py <cran cache folder> <cran_exports.json>
"""
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
NAMES = ('rdirichlet', 'ddirichlet', 'weightedMedian')
BASE = {'R', 'base', 'stats', 'graphics', 'grDevices', 'utils', 'datasets', 'methods', 'grid', 'parallel', 'splines',
        'stats4', 'tools', 'tcltk', 'compiler'}


def exported(ns_text):
    code = '\n'.join(line.split('#', 1)[0] for line in ns_text.splitlines())
    explicit = set()
    for m in re.finditer(r'\bexport\s*\((.*?)\)', code, re.S):
        explicit.update(x.strip().strip('"\'`') for x in m.group(1).split(','))
    return explicit, re.findall(r'exportPattern\s*\(\s*["\']([^"\']+)["\']', code)


def main():
    cache, data = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    packages = data['packages']
    calls = json.loads((ROOT / 'docs/thesis_research/literal_weights_stage_2026_11_03/source_binding.json')
                       .read_text(encoding='utf-8'))['library_calls_in_run_order']
    table = {}
    for pkg, info in packages.items():
        rows = []
        for v in info['window_versions']:
            ns_file = cache / 'mirror' / pkg / v['version'] / 'NAMESPACE'
            if not v.get('mirror'):
                rows.append({'version': v['version'], 'published': v['published'], 'checked': False})
                continue
            explicit, patterns = exported(ns_file.read_text(encoding='utf-8'))
            rows.append({'version': v['version'], 'published': v['published'], 'checked': True,
                         'export_pattern': patterns, **{n: n in explicit for n in NAMES}})
        table[pkg] = {'versions': rows, 'depends_union': sorted({d for v in info['window_versions']
                                                                 for d in v.get('depends', []) if d not in BASE})}

    # search path: later attach = earlier in the path; a package's Depends are attached right before it
    order, seen = [], set()

    def attach(pkg):
        if pkg in seen or pkg in BASE or pkg not in table:
            return
        seen.add(pkg)
        for dep in table[pkg]['depends_union']:
            attach(dep)
        order.append(pkg)

    for call in calls:
        attach(call['package'])
    search_path = list(reversed(order))
    providers = {}
    for name in NAMES:
        hits = []
        for pkg in search_path:
            rows = [r for r in table[pkg]['versions'] if r['checked']]
            if any(r.get(name) for r in rows):
                hits.append({'package': pkg, 'versions_exporting': [r['version'] for r in rows if r.get(name)],
                             'versions_not_exporting': [r['version'] for r in rows if not r.get(name)]})
        providers[name] = {'first_on_search_path': hits[0]['package'] if hits else None, 'all_exporting': hits}
    unresolved = {pkg: sorted({p for r in t['versions'] if r['checked'] for p in r['export_pattern']})
                  for pkg, t in table.items() if any(r['checked'] and r['export_pattern'] for r in t['versions'])}
    unchecked = {pkg: [r['version'] for r in t['versions'] if not r['checked']]
                 for pkg, t in table.items() if any(not r['checked'] for r in t['versions'])}
    result = {
        'kind': 'search_path_provider_analysis_from_CRAN_namespaces_not_runtime_binding',
        'window': data['window'], 'packages_checked': len(table),
        'search_path_front_to_back': search_path,
        'providers': providers,
        'export_pattern_packages_not_resolved_by_namespace': unresolved,
        'versions_missing_in_mirror': unchecked,
        'caveat': 'exact package versions of the historical run are unbound; a provider is only bound if every '
                  'version in the window agrees and no exportPattern package in front of it could export the name',
    }
    (OUT / 'provider_analysis.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8', newline='\n')
    for name, p in providers.items():
        print(name, '->', p['first_on_search_path'], [h['package'] for h in p['all_exporting']])
    print('exportPattern:', unresolved)
    print('missing in mirror:', {k: len(v) for k, v in unchecked.items()})


if __name__ == '__main__':
    main()
