"""Compare actual vector marks in two TF PDFs; never recover probability arrays."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def mark_text(trace):
    return ''.join(chr(c[0]) for c in trace['chars'])


def numeric(text):
    text = text.replace('\u2212', '-')
    return float(text) if re.fullmatch(r'-?\d+(?:\.\d+)?', text) else None


def read_page(page):
    traces = page.get_texttrace()
    point_traces = [t for t in traces if t['font'] == 'ZapfDingbats']
    if not point_traces:
        raise ValueError('Missing observed point font')
    points = {kind: [tuple(c[2]) for t in point_traces if t['type'] == kind for c in t['chars']] for kind in (0, 1)}
    expected_color = (.18, .545, .341)
    colors_ok = all(len(t['color']) == 3 and max(abs(a-b) for a,b in zip(t['color'], expected_color)) < 1e-5 for t in point_traces)
    if any(t['type'] not in (0, 1) for t in point_traces):
        raise ValueError('Uninspected glyph painting mode')
    first = min(t['seqno'] for t in point_traces)
    last = max(t['seqno'] for t in point_traces)
    before = [t for t in traces if t['seqno'] < first]
    title = ''.join(mark_text(t) for t in traces if t['font'] == 'Helvetica-Bold').replace('\u2212', '-')
    drawings = page.get_drawings()
    axis_drawings = [d for d in drawings if d['seqno'] < first]
    contours = [d for d in drawings if d['seqno'] > last]
    # Match numeric tick labels to actual vector tick marks, not text-box centers.
    xticks = []; yticks = []
    for d in axis_drawings:
        if len(d['items']) != 1 or d['items'][0][0] != 'l':
            continue
        _, a, b = d['items'][0]
        if abs(a.x-b.x) < .001 and abs(abs(a.y-b.y)-7.2) < .02:
            xticks.append((a.x, a.y))
        if abs(a.y-b.y) < .001 and abs(abs(a.x-b.x)-7.2) < .02:
            yticks.append((a.x, a.y))
    xlabels = [numeric(mark_text(t)) for t in before if numeric(mark_text(t)) is not None and t['bbox'][1] > 530]
    ylabels = [numeric(mark_text(t)) for t in before if numeric(mark_text(t)) is not None and t['bbox'][2] < 60]
    if len(xticks) != len(xlabels) or len(yticks) != len(ylabels) or min(len(xticks),len(yticks)) < 2:
        raise ValueError('Ambiguous axis tick matching')
    def calibration(ticks, labels, axis):
        coords = [p[axis] for p in ticks]
        slope = (labels[-1]-labels[0])/(coords[-1]-coords[0])
        intercept = labels[0]-slope*coords[0]
        residual = max(abs(slope*x+intercept-y) for x,y in zip(coords,labels))
        return {'ticks_pdf_points':coords,'labels':labels,'units_per_pdf_point':slope,'intercept':intercept,'max_tick_rounding_residual':residual}
    axes = {'temperature':calibration(xticks,xlabels,0),'precipitation':calibration(yticks,ylabels,1),
            'text_before_points':[{'text':mark_text(t),'origins':[c[2] for c in t['chars']],'dir':t['dir']} for t in before],
            'vector_axes_sha256':digest([str(d['items']) for d in axis_drawings]),'page_rectangle':tuple(page.rect)}
    filled = points[0]
    return {'title':title,'filled_count':len(filled),'stroked_count':len(points[1]),'fill_stroke_positions_identical':filled == points[1],
            'all_point_colors_expected':colors_ok,'point_order_sha256':digest(filled),'point_multiset_sha256':digest(sorted(Counter(filled).items())),
            'pdf_point_origin_bounds':[min(p[0] for p in filled),min(p[1] for p in filled),max(p[0] for p in filled),max(p[1] for p in filled)],
            'axis_sha256':digest(axes),'axes':axes,'contour_path_count':len(contours),
            'contour_vector_sha256':digest([{'items':str(d['items']),'width':d['width'],'color':d['color'],'dashes':d['dashes']} for d in contours]),
            'contour_labels_sha256':digest([{'text':mark_text(t),'origins':[c[2] for c in t['chars']],'dir':t['dir']} for t in traces if last < t['seqno'] and t['font'] != 'Helvetica-Bold'])}


def run(repo, output):
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(repo/'.local/console_provenance_v1/pdf_runtime'))
    import pymupdf
    old = json.loads((repo/'.local/console_provenance_v1/emd_plot_pages.json').read_text())
    second = json.loads((repo/'.local/tf_plot_comparison_v1/second_plot_pages.json').read_text())
    emd_path = Path(old['pdf']); other_path = emd_path.with_name('all_trafos.pdf')
    files = {'emd':emd_path,'second':other_path}
    hashes = {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()}
    page_data = {}
    for kind, path in files.items():
        raw = old if kind == 'emd' else second
        doc = pymupdf.open(path); rows = []
        if len(doc) != len(raw['pages']):
            raise ValueError('Page count changed')
        for i, page in enumerate(doc):
            row = read_page(page); ref = raw['pages'][i]
            row.update(page=i+1,snapshot_positions=ref['same_title_and_presence_count_positions'])
            row['independent_raw_title_count_match'] = row['title'] == ref['title'] and row['filled_count'] == ref['green_point_glyph_count']
            rows.append(row)
            if (i+1)%50 == 0: print(f'{kind}: {i+1} pages',flush=True)
        page_data[kind] = rows
    by_position = {r['snapshot_positions'][0]:r for r in page_data['emd'] if len(r['snapshot_positions'])==1}
    pairs = []
    for r in page_data['second']:
        if len(r['snapshot_positions']) != 1: raise ValueError('Non-unique position')
        position=r['snapshot_positions'][0]; e=by_position[position]
        pairs.append({'position':position,'title':r['title'],'second_page':r['page'],'emd_page':e['page'],
                      'same_point_order':r['point_order_sha256']==e['point_order_sha256'],
                      'same_point_multiset':r['point_multiset_sha256']==e['point_multiset_sha256'],
                      'same_axes':r['axis_sha256']==e['axis_sha256'],
                      'same_contour_vectors':r['contour_vector_sha256']==e['contour_vector_sha256'],
                      'same_contour_labels':r['contour_labels_sha256']==e['contour_labels_sha256']})
    result={'method':'Unfiltered MuPDF texttrace, fill/stroke separately; vector ticks and contours read at stored PDF precision. No TF values digitized.',
            'pdf_hashes':hashes,'source_hashes_unchanged':all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in files.items()),
            'counts':{key:sum(p[key] for p in pairs) for key in ['same_point_order','same_point_multiset','same_axes','same_contour_vectors','same_contour_labels']},
            'independent_title_count_checks_pass':all(r['independent_raw_title_count_match'] for rows in page_data.values() for r in rows),
            'fill_stroke_checks_pass':all(r['fill_stroke_positions_identical'] and r['filled_count']==r['stroked_count'] for rows in page_data.values() for r in rows),
            'point_color_checks_pass':all(r['all_point_colors_expected'] for rows in page_data.values() for r in rows),
            'unique_axis_signatures':{k:len({r['axis_sha256'] for r in rows}) for k,rows in page_data.items()},
            'pairs':pairs,'pages':page_data,'no_numeric_transfer_functions_recovered':True,'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['pages','pairs','method']},indent=2),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    run(Path(__file__).resolve().parents[1],a.output)