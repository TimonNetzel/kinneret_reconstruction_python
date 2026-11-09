"""Check on CRAN sources whether signal, pracma, WaveletComp or biwavelet mask stats::fft or stats::mvfft in the run window.

Tarballs go to <cache>/<package>/ and are only read as archives: nothing is extracted or executed. A version counts when it was the
current one at any time inside the window: every version published inside it plus the last one published before it starts.
The other packages attached by main_v1.2.R are not checked here. Usage: check_fft_masking.py <cache dir>
"""
import hashlib
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
WINDOW = ('2018-04-23', '2023-07-07')  # window of r_runtime_crosscheck_2026_11_03: R 3.5.0 to the last source change
PACKAGES = {'signal': 24, 'pracma': 47, 'WaveletComp': 51, 'biwavelet': 52}  # main_v1.2.R line of library()
BASE = 'https://cran.r-project.org/src/contrib/'
DEFINES = re.compile(r'^\s*[`"\']?(fft|mvfft)[`"\']?\s*(<-|=)\s*function', re.M)
EXPORTED = re.compile(r'export\s*\(([^)]*)\)', re.S)
LISTING = re.compile(r'href="([A-Za-z0-9.]+_[0-9.\-]+\.tar\.gz)".*?(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}')


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'kinneret-reconstruction-source-check'})
    with urllib.request.urlopen(request, timeout=120) as reply:
        return reply.read()


def versions(package, current_listing):
    found = []
    try:
        found += [(name, date, f'Archive/{package}/') for name, date in LISTING.findall(fetch(f'{BASE}Archive/{package}/').decode('utf-8', 'replace'))]
    except OSError:
        pass  # no archive directory: only one version was ever published
    found += [(name, date, '') for name, date in current_listing if name.startswith(package + '_')]
    return sorted(found, key=lambda row: row[1])


def inside_window(rows):
    keep = [row for row in rows if WINDOW[0] <= row[1] <= WINDOW[1]]
    before = [row for row in rows if row[1] < WINDOW[0]]
    return before[-1:] + keep


def inspect(path):
    report = {'namespace_exports_fft': False, 'namespace_exports_mvfft': False, 'export_pattern': False, 'defines_in_R': [], 'daughter_type': None}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = member.name.split('/')
            handle = None
            if len(parts) == 2 and parts[1] == 'NAMESPACE':
                handle = archive.extractfile(member)
                text = re.sub(r'#.*', '', handle.read().decode('utf-8', 'replace'))
                names = {n.strip().strip('"\'') for block in EXPORTED.findall(text) for n in block.split(',')}
                report['namespace_exports_fft'] = 'fft' in names
                report['namespace_exports_mvfft'] = 'mvfft' in names
                report['export_pattern'] = 'exportPattern' in text
            elif len(parts) == 3 and parts[1] == 'R' and parts[2].lower().endswith('.r'):
                handle = archive.extractfile(member)
                hits = DEFINES.findall(handle.read().decode('utf-8', 'replace'))
                report['defines_in_R'] += [f'{parts[2]}:{name}' for name, _ in hits]
            elif len(parts) == 3 and parts[1] == 'src' and 'morlet' in parts[2].lower() and parts[2].lower().endswith('.cpp'):
                handle = archive.extractfile(member)
                match = re.search(r'(\w+(?:::\w+)?(?:<[^>]*>)?)\s+daughter\b', handle.read().decode('utf-8', 'replace'))
                report['daughter_type'] = match.group(1) if match else None
    return report


def main():
    cache = Path(sys.argv[1]).resolve()
    current = LISTING.findall(fetch(BASE).decode('utf-8', 'replace'))
    results, masked = {}, []
    for package, line in PACKAGES.items():
        folder = cache / package
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for name, date, prefix in inside_window(versions(package, current)):
            target = folder / name
            if not target.exists():
                target.write_bytes(fetch(BASE + prefix + name))
            found = inspect(target)
            rows.append({'file': name, 'date': date, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), **found})
            if found['namespace_exports_fft'] or found['namespace_exports_mvfft'] or (found['export_pattern'] and found['defines_in_R']):
                masked.append(name)
        results[package] = {'library_line_main_v1_2': line, 'versions': rows}
    report = {'kind': 'cran_source_check_of_fft_masking', 'window': WINDOW, 'packages': results, 'versions_read': sum(len(p['versions']) for p in results.values()),
              'versions_masking_fft_or_mvfft': masked,
              'not_checked': 'the other packages attached by main_v1.2.R; the installed versions of the historical run are unknown',
              'source': 'CRAN source archives, read as archives only, nothing extracted or executed'}
    (OUT / 'fft_masking.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(report['versions_read'], 'versions read; masking:', masked)
    for package, data in results.items():
        print(package, [(v['file'], v['date'], v['export_pattern'], v['defines_in_R'], v['daughter_type']) for v in data['versions']])


if __name__ == '__main__':
    main()
