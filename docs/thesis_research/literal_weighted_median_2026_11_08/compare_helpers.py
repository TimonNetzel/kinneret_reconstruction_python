"""Compare the shared matrixStats helper files behind weightedMedian across the 13 window versions.

Tarballs are read in memory only; nothing is built, unpacked or executed. Compared, file by file: every src/000.*
helper, the index validation files and the R wrapper. From the merged text of each version the definitions that the
call weightedMedian(x, w) (numeric x, idxs = NULL) actually expands are extracted and grouped. Writes
helper_comparison.json and helper_diffs.txt. Usage: compare_helpers.py <tarball folder>
"""
import difflib
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
VERSIONS = json.loads((OUT.parent / 'r_runtime_crosscheck_2026_11_03/weighted_median_versions.json').read_text(encoding='utf-8'))
FILES = re.compile(r'matrixStats/src/(000\.[^/]*|validateIndices[^/]*)$')
MACROS = ('R_INDEX_GET', 'R_INDEX_OP', 'IDX_INDEX', 'X_QSORT_I', 'X_PSORT', 'X_ISNAN', 'X_NA', 'X_C_TYPE', 'X_IN_C',
          'X_C_SIGNATURE', 'NA_R_XLEN_T', 'ISNAN', 'Calloc', 'Free', 'R_Calloc', 'R_Free')
FUNCTIONS = ('validateIndices', 'validateIndicesCheckNA', 'asLogicalNoNA', 'assertArgVector')
KERNEL_START = 'Check for missing, negative, and infite weights'


def load(version, tarballs):
    data = (tarballs / f'matrixStats_{version}.tar.gz').read_bytes()
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        helper = {}
        kernel = None
        for m in tar.getmembers():
            if not m.isfile():
                continue
            if FILES.search(m.name):
                helper[m.name.split('/src/')[1]] = tar.extractfile(m).read().decode('utf-8', 'replace').replace('\r\n', '\n')
            if m.name.endswith('src/weightedMedian_lowlevel_template.h'):
                kernel = tar.extractfile(m).read().decode('utf-8', 'replace').replace('\r\n', '\n')
    return hashlib.sha256(data).hexdigest(), helper, kernel


def defines(text, name):
    """Logical #define lines for name (continuation lines joined), in file order."""
    out = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        if re.match(rf'\s*#\s*define\s+{name}\b', lines[i]):
            s = lines[i]
            while s.endswith('\\') and i + 1 < len(lines):
                i += 1
                s = s[:-1].rstrip() + ' ' + lines[i].strip()
            out.append(' '.join(s.split()))
        i += 1
    return out


def normal_kernel(text):
    """The kernel body with the index expression written once, so only arithmetic and control flow can differ."""
    start = text.index('/* Check for missing')
    body = text[start:text.index('HISTORY')] if 'HISTORY' in text else text[start:]
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
    body = re.sub(r'//[^\n]*', '', body)
    body = body.replace('((idxs == NULL) ? (ii) : idxs[ii])', 'IDX(ii)').replace('((idxs == NULL) ? (jj) : idxs[jj])', 'IDX(jj)')
    body = body.replace('IDX_INDEX(cidxs,ii)', 'IDX(ii)').replace('IDX_INDEX(cidxs,jj)', 'IDX(jj)')
    body = re.sub(r'R_INDEX_GET\(\s*(\w+)\s*,\s*IDX\((\w+)\)\s*,\s*(\w+)(\s*,\s*\w+)?\s*\)', r'GET(\1,\2,\3)', body)
    return '\n'.join(' '.join(line.split()) for line in body.split('\n') if line.strip())


def main():
    tarballs = Path(sys.argv[1])
    rows, per_version, kernels = [], {}, {}
    for v in VERSIONS['versions']:
        sha, helper, kernel = load(v['version'], tarballs)
        if sha != v['tarball_sha256']:
            raise ValueError(f"{v['version']}: tarball hash differs from the recorded one")
        per_version[v['version']] = helper
        kernels[v['version']] = kernel
    names = sorted({n for h in per_version.values() for n in h})
    file_groups = {}
    for name in names:
        groups = {}
        for version, helper in per_version.items():
            text = helper.get(name)
            key = hashlib.sha256(text.encode()).hexdigest()[:12] if text is not None else 'absent'
            groups.setdefault(key, []).append(version)
        file_groups[name] = groups
    macro_values = {}
    for macro in MACROS:
        groups = {}
        for version, helper in per_version.items():
            found = sorted({d for text in helper.values() for d in defines(text, macro)})
            groups.setdefault(json.dumps(found, ensure_ascii=False), []).append(version)
        macro_values[macro] = [{'definitions': json.loads(k), 'versions': v} for k, v in groups.items()]
    kernel_groups = {}
    normal = {v: normal_kernel(k) for v, k in kernels.items()}
    for version, text in normal.items():
        kernel_groups.setdefault(hashlib.sha256(text.encode()).hexdigest()[:12], []).append(version)
    with (OUT / 'helper_diffs.txt').open('w', encoding='utf-8', newline='\n') as f:
        order = [v['version'] for v in VERSIONS['versions']]
        for a, b in zip(order, order[1:]):
            head = False
            for name in names:
                x, y = per_version[a].get(name), per_version[b].get(name)
                if x == y:
                    continue
                if not head:
                    f.write(f'===== {a} -> {b}\n')
                    head = True
                d = list(difflib.unified_diff((x or '').split('\n'), (y or '').split('\n'), f'{a}/{name}', f'{b}/{name}',
                                              n=1, lineterm=''))
                f.write('\n'.join(d[:150]) + '\n')
    result = {
        'kind': 'matrixStats_helper_comparison_source_text_only_not_runtime_binding',
        'window': VERSIONS['window'], 'versions': [v['version'] for v in VERSIONS['versions']],
        'tarball_hashes_equal_recorded': True, 'files': file_groups, 'macros': macro_values,
        'normalised_kernel_groups': kernel_groups,
        'normalised_kernel_note': 'comments removed, the index expression and R_INDEX_GET written once; a single group '
                                  'means identical arithmetic and control flow for idxs = NULL',
    }
    (OUT / 'helper_comparison.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n', encoding='utf-8',
                                                newline='\n')
    for name, groups in file_groups.items():
        print(name, len(groups), list(groups.values()) if len(groups) > 1 else '')
    for macro, v in macro_values.items():
        print(macro, json.dumps(v, ensure_ascii=False))
    print('kernel groups', kernel_groups)


if __name__ == '__main__':
    main()
