"""Retry only unresolved byte contents from an existing historical model audit.

Preserves the prior evidence, validates all source aliases by content, never
executes R, and never equates a parse success with validated predictions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
from audit_my_reconst_sources import sha


def emit(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def retry(config, previous, output, timeout=180):
    cfg = json.loads(config.read_text(encoding='utf-8-sig'))
    source = Path(cfg['additional_source_root']).resolve(strict=True)
    original = Path(cfg['original_data_root']).resolve(strict=True)
    output = output.resolve()
    for root in (source, original):
        if output == root or root in output.parents or output in root.parents:
            raise ValueError('Output overlaps an original source tree')
    prior_digest = sha(previous)
    prior = json.loads(previous.read_text(encoding='utf-8'))
    selected = [(i, r) for i, r in enumerate(prior['results'], 1)
                if r['parse_status'] != 'complete_structure_not_R_validation']
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for position, old in selected:
        paths = [(source / name).resolve(strict=True) for name in old['relative_paths']]
        if any(source not in p.parents for p in paths):
            raise ValueError('Source path escapes configured root')
        if not paths or not all(sha(p) == old['sha256'] for p in paths):
            raise ValueError('Source contents differ from prior evidence')
        started = time.monotonic()
        try:
            proc = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('audit_historical_provenance.py')),
                                   '--worker', str(paths[0])], capture_output=True, text=True,
                                  encoding='utf-8', timeout=timeout)
            if proc.returncode:
                result = {'parse_status': 'failed', 'error': proc.stderr or proc.stdout,
                          'R_prediction_validated': False}
            else:
                result = json.loads(proc.stdout)
        except subprocess.TimeoutExpired:
            result = {'parse_status': 'timeout_not_classified', 'timeout_seconds': timeout,
                      'R_prediction_validated': False}
        result.update(prior_position=position, prior_parse_status=old['parse_status'],
                      sha256=old['sha256'], relative_paths=old['relative_paths'],
                      elapsed_seconds=round(time.monotonic()-started, 3),
                      original_alias_hashes_unchanged=all(sha(p) == old['sha256'] for p in paths))
        emit(output / f'model_{position:03d}.json', result)
        results.append(result)
        print(json.dumps({k:result.get(k) for k in ('prior_position','parse_status','caret_fit_count','elapsed_seconds') }), flush=True)
    updated = {r['sha256']: r for r in results}
    combined = [updated.get(r['sha256'], r) for r in prior['results']]
    report = {'previous_evidence_sha256': prior_digest, 'previous_evidence_unchanged': sha(previous)==prior_digest,
              'retried_unique_contents': len(results), 'retried_file_aliases': sum(len(r['relative_paths']) for r in results),
              'retry_status_counts': dict(Counter(r['parse_status'] for r in results)),
              'combined_status_counts': dict(Counter(r['parse_status'] for r in combined)),
              'combined_caret_fits': sum(r.get('caret_fit_count', 0) for r in combined),
              'combined_binary_caret_fits': sum(r.get('binary_caret_fit_count', 0) for r in combined),
              'combined_multiclass_caret_fits': sum(r.get('multiclass_caret_fit_count', 0) for r in combined),
              'results': results, 'no_execution_no_training_no_prediction': True,
              'reader_sha256': sha(Path(__file__).with_name('local_rdata_metadata.py')),
              'worker_sha256': sha(Path(__file__).with_name('audit_historical_provenance.py'))}
    emit(output/'retry_summary.json', report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--previous', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--timeout', default=180, type=float)
    args=ap.parse_args()
    if args.timeout <= 0: ap.error('timeout must be positive')
    retry(args.config, args.previous, args.output, args.timeout)


if __name__ == '__main__': main()
