#!/usr/bin/env python3
"""Source correspondence audit; never changes ages or reconstructs pollen counts."""
import argparse
from collections import Counter, defaultdict, deque
import csv
from decimal import Decimal as D, ROUND_HALF_EVEN
import hashlib
import json
from pathlib import Path
import statistics

from import_data import read_pangaea

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT/'docs/thesis_research/chronology_crosswalk_2026_10_29/sources.json'
TAXA = [('Pin [#]', 'Pinus [%]', 4), ('Bet [#]', 'Betula [%]', 6),
        ('Que [#]', 'Quercus [%]', 7), ('Aln [#]', 'Alnus [%]', 8),
        ('Abi [#]', 'Abies [%]', 9), ('Fag [#]', 'Fagus [%]', 10),
        ('Art [#]', 'Artemisia [%]', 13), ('Cheae [#]', 'Chenopod [%]', 14)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_out(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def csv_out(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def noaa_table(path, columns):
    # Explicit source layout, no generic numeric-line guessing.
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(lines) >= 4, 'Missing NOAA headers')
    if columns == 2:
        require(lines[2].split('\t')[:2] == ['Depth (m)', 'Age (Lamination years BP)'], 'Changed age header')
    else:
        require(lines[1].startswith('Sample age\tPollen count\t'), 'Changed pollen header')
        require(lines[2].startswith('lamination years BP\tgrains\t'), 'Changed pollen units')
    rows = []
    for line in lines[3:]:
        if not line.strip():
            continue
        row = [s.strip() for s in line.split('\t')]
        require(len(row) >= columns, 'Truncated NOAA row')
        D(row[0]); D(row[1])
        rows.append(row)
    return rows


def percent_error(count, total, published):
    c, n, p = D(count), D(total), D(published)
    require(c >= 0 and c == c.to_integral_value() and n > 0, 'Invalid source count/denominator')
    require(0 <= p <= 100, 'Invalid published percentage')
    return 100*c/n-p


def candidate_rows(raw, source, tolerance, kind):
    hits = []
    for i, row in enumerate(source):
        total = row['Pollen tot [#]'] if kind == 'pangaea' else row[1]
        errors = [percent_error(raw[a], total, row[b] if kind == 'pangaea' else row[c]) for a, b, c in TAXA]
        if all(abs(e) <= tolerance for e in errors):
            hits.append(i)
    return hits


def age_archive_alignment(noaa, pangaea):
    # Preserve multiplicities: duplicate pairs are not independent observations.
    queues = defaultdict(deque)
    for i, row in enumerate(pangaea):
        queues[D(row['Depth sed [m]']), D(row['Age model [ka]'])].append(i)
    require(len(noaa) == len(pangaea), 'Age archive length differs')
    alignment = []
    for i, row in enumerate(noaa):
        depth, age = D(row[0]), D(row[1])
        key = depth.quantize(D('.001'), rounding=ROUND_HALF_EVEN), age
        require(bool(queues[key]), 'No multiplicity-preserving age correspondence')
        j = queues[key].popleft()
        alignment.append(dict(noaa_source_row=i+1, pangaea_source_row=j+1,
                              noaa_depth_m=str(depth), pangaea_depth_m=str(key[0]),
                              age_numeric_token=str(age), row_moved=int(i != j)))
    require(all(not q for q in queues.values()), 'Unmatched PANGAEA ages')
    return alignment


def run(raw, output, catalog_path=CATALOG):
    require(not output.exists(), 'Output exists')
    catalog = json.loads(catalog_path.read_text())
    for source in catalog['files']:
        path = raw/source['file']
        require(path.name == source['file'], 'Invalid source path')
        require(path.stat().st_size == source['byte_count'] and digest(path) == source['sha256'], 'Source checksum mismatch: '+path.name)
    pollen = read_pangaea(raw/'monticchio_739452.tab')[2]
    cut = [r for r in read_pangaea(raw/'epd_ages_cut_804596.tab')[2] if r['Event'] == 'LGM90D' and r['ID'] == '932']
    published = read_pangaea(raw/'pollen_707075.tab')[2]
    noaa = noaa_table(raw/'pollen.txt', 18)
    noaa_age = noaa_table(raw/'age.txt', 2)
    pangaea_age = read_pangaea(raw/'monticchio_707050.tab')[2]
    require(len(pollen) == 245 and len(cut) == 69 and len(published) == len(noaa) == 496, 'Source cohort changed')
    # Establish the two summary exports' correspondence before the fossil search.
    denominator_changes = []
    for i, (a, b) in enumerate(zip(noaa, published)):
        require(D(a[0]) == 1000*D(b['Age [ka BP]']), 'NOAA/PANGAEA summary age mismatch')
        require(D(a[1]).quantize(D('1'), rounding=ROUND_HALF_EVEN) == D(b['Pollen tot [#]']), 'Unexplained summary denominator difference')
        if D(a[1]) != D(b['Pollen tot [#]']):
            denominator_changes.append(dict(summary_source_row=i+1, age_bp=a[0],
                noaa_pollen_count=a[1], pangaea_pollen_count=b['Pollen tot [#]'],
                status='half_grain_value_rounded_in_pangaea', meaning_of_fractional_value='unresolved'))
        for _, col, j in TAXA:
            require(abs(D(a[j])-D(b[col])) <= D('.05'), 'Summary rounding exceeds expected precision')
    alignment = age_archive_alignment(noaa_age, pangaea_age)
    used = set()
    correspondences, discrepancies = [], []
    for age in cut:
        depth = D(age['Depth sed [m]'])
        links = [i for i, r in enumerate(pollen) if abs(D(r['Depth sed [m]'])-depth) <= D('.005')]
        require(len(links) == 1 and links[0] not in used, 'Nonunique raw/Cut link')
        index = links[0]; used.add(index); row = pollen[index]
        hits_p = candidate_rows(row, published, D('.005'), 'pangaea')
        hits_n = candidate_rows(row, noaa, D('.05'), 'noaa')
        require(hits_p == hits_n, 'Two source precisions imply different candidates')
        record = dict(sample_id=f'LGM90D:{index+1:06d}', source_row=index+1,
                      pollen_depth_m=row['Depth sed [m]'], chronology_depth_m=str(depth),
                      candidate_count=len(hits_p), matching_taxa=8,
                      status='unique_pollen_fingerprint_candidate' if len(hits_p) == 1 else 'unresolved',
                      summary_source_row='', original_summary_age_bp='', epd_cut_age_bp=str(1000*D(age['Age dated [ka BP]'])),
                      original_minus_epd_years='', max_abs_percent_error='', denominator_from_published_summary='')
        if len(hits_p) == 1:
            j = hits_p[0]; s = published[j]
            require(noaa[j][0] != '', 'Missing summary age')
            record.update(summary_source_row=j+1, original_summary_age_bp=str(1000*D(s['Age [ka BP]'])),
                          original_minus_epd_years=str(1000*(D(s['Age [ka BP]'])-D(age['Age dated [ka BP]']))),
                          max_abs_percent_error=str(max(abs(percent_error(row[a], s['Pollen tot [#]'], s[b])) for a, b, _ in TAXA)),
                          denominator_from_published_summary=s['Pollen tot [#]'])
        else:
            # Diagnostics only: never accept a relaxed match after seeing a failure.
            candidates = []
            for j, s in enumerate(published):
                errors = [abs(percent_error(row[a], s['Pollen tot [#]'], s[b])) for a, b, _ in TAXA]
                candidates.append((sum(errors), sum(e > D('.005') for e in errors), j))
            error, failed, j = min(candidates)
            s = published[j]
            for a, b, _ in TAXA:
                delta = percent_error(row[a], s['Pollen tot [#]'], s[b])
                if abs(delta) > D('.005'):
                    discrepancies.append(dict(sample_id=record['sample_id'], exploratory_closest_summary_row=j+1,
                        original_column=a, existing_raw_count=row[a], summary_denominator=s['Pollen tot [#]'],
                        published_percent=s[b], raw_count_percent=str(100*D(row[a])/D(s['Pollen tot [#]'])),
                        residual_percentage_points=str(delta), accepted_as_match=0))
        correspondences.append(record)
    matches = [r for r in correspondences if r['status'] != 'unresolved']
    require(len({r['summary_source_row'] for r in matches}) == len(matches), 'Summary sample reused')
    times = [D(r['original_summary_age_bp']) for r in matches]
    require(all(b > a for a, b in zip(times, times[1:])), 'Nonmonotone matched source order')
    errors = [abs(D(r['original_minus_epd_years'])) for r in matches]
    same_depth = lambda rows: sum(v > 1 for v in Counter(D(r[0]) for r in rows).values())
    summary = dict(schema_version=1, baseline_commit='f9acc16c58a47a51b1063690a69a2db01381c76a',
        nature='exploratory_source_correspondence_not_independent_chronology_validation',
        pollen_fingerprint=dict(features=[a for a, _, _ in TAXA], pangaea_tolerance_percentage_points='.005',
            noaa_tolerance_percentage_points='.05', age_used_for_matching=False, counts_reconstructed_from_percentages=False,
            published_denominator_used_only_for_forward_percent_check=True, retained_samples=69, unique_candidates=len(matches),
            unresolved_sample_ids=[r['sample_id'] for r in correspondences if r['status'] == 'unresolved'],
            ambiguous_candidates=sum(r['candidate_count'] > 1 for r in correspondences),
            same_candidates_in_both_summary_sources=True,
            median_abs_source_minus_epd_years=str(statistics.median(errors)), max_abs_source_minus_epd_years=str(max(errors)),
            direct_original_sample_identifiers_available=False, depth_crosswalk_to_MON16_established=False),
        summary_archives=dict(rows=496, all_ages_correspond=True, exact_denominator_matches=496-len(denominator_changes),
            fractional_noaa_denominators_rounded_in_pangaea=len(denominator_changes),
            all_denominators_match_after_half_even_rounding=True, all_eight_percent_columns_rounding_consistent=True),
        age_archives=dict(rows=1120, all_pairs_correspond_after_half_even_depth_rounding=True,
            rounding_depth_step_m='.001', source_row_positions_changed=sum(r['row_moved'] for r in alignment),
            noaa_depth_reversals=sum(D(b[0]) < D(a[0]) for a, b in zip(noaa_age, noaa_age[1:])),
            noaa_duplicate_depth_groups=same_depth(noaa_age),
            pangaea_duplicate_depth_groups=same_depth([(r['Depth sed [m]'], r['Age model [ka]']) for r in pangaea_age]),
            noaa_header_age_unit_ambiguous=True, numeric_age_scale_confirmed_by_pangaea_ka=True,
            age_scale_or_depth_order_changed_in_original_files=False),
        ages_changed=False, new_age_model=False, excluded_samples_released=False)
    output.mkdir(parents=True)
    csv_out(output/'sample_correspondence.csv', correspondences)
    csv_out(output/'unresolved_cells.csv', discrepancies)
    csv_out(output/'age_archive_alignment.csv', alignment)
    csv_out(output/'denominator_changes.csv', denominator_changes)
    json_out(output/'summary.json', summary)
    json_out(output/'manifest.json', dict(script_sha256=digest(Path(__file__)), parser_sha256=digest(ROOT/'tools/thesis_import/import_data.py'),
        catalog_sha256=digest(catalog_path), sources={r['file']:r['sha256'] for r in catalog['files']},
        outputs={p.name:dict(sha256=digest(p),byte_count=p.stat().st_size) for p in sorted(output.iterdir())}))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.raw_dir, args.output_dir)
