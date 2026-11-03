"""Read PDF titles and green point glyph counts, without digitizing TF values."""
from __future__ import annotations
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path


def read_page_marks(operations, fonts):
    font=None;color=None;stack=[];titles=[];points=0
    for args,op in operations:
        if op==b'q':stack.append((font,color))
        elif op==b'Q':
            if not stack:raise ValueError('Unbalanced PDF graphics state')
            font,color=stack.pop()
        elif op==b'Tf':font=str(args[0])
        elif op in (b'rg',b'scn',b'g',b'k'):
            try:color=tuple(float(v) for v in args)
            except (TypeError,ValueError):color=None
        elif op==b'Do':raise ValueError('Uninspected PDF form/image XObject')
        elif op in (b'Tj',b'TJ'):
            values=[args[0]] if op==b'Tj' else args[0]
            text=''.join(v.decode('latin-1') if isinstance(v,bytes) else v for v in values if isinstance(v,(str,bytes)))
            name=fonts.get(font,'')
            if name=='/ZapfDingbats':
                if set(text)-{'l'}:raise ValueError('Unexpected point glyph')
                if color is None or len(color)!=3 or any(abs(a-b)>1e-6 for a,b in zip(color,(.18,.545,.341))):
                    raise ValueError('Point glyph is not in the expected observed green color')
                points+=len(text)
            elif name=='/Helvetica-Bold':titles.append(text)
    if stack:raise ValueError('Unclosed PDF graphics state')
    title=''.join(titles).strip()
    if not title:raise ValueError('No bold title in PDF page')
    return {'title':title,'green_point_glyph_count':points}


def run(pdf_path, cardinality, output):
    from pypdf import PdfReader
    from pypdf.generic import ContentStream
    if output.exists():raise FileExistsError('Refusing to overwrite PDF evidence')
    initial=hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    data=json.loads(cardinality.read_text(encoding='utf-8'))
    source_rows=data['rows'];reader=PdfReader(pdf_path);pages=[]
    for i,page in enumerate(reader.pages,1):
        fonts={str(k):str(v.get_object().get('/BaseFont')) for k,v in page['/Resources']['/Font'].items()}
        marks=read_page_marks(ContentStream(page.get_contents(),reader).operations,fonts)
        matches=[r for r in source_rows if r['snapshot_name_not_historical_taxon_assignment']==marks['title']]
        count_matches=[r['printed_position'] for r in matches if r['north']['presence']==marks['green_point_glyph_count']]
        pages.append({'page':i,**marks,'same_title_snapshot_positions':[r['printed_position'] for r in matches],
                      'same_title_and_presence_count_positions':count_matches})
        if i%50==0:print(f'PDF pages read: {i}',flush=True)
    titles=Counter(p['title'] for p in pages)
    expected=Counter(r['snapshot_name_not_historical_taxon_assignment'] for r in source_rows if r['north']['presence']!=1)
    result={'pdf':str(pdf_path),'sha256':initial,'page_count':len(pages),'metadata':dict(reader.metadata or {}),
            'pages':pages,'all_titles_known':all(p['same_title_snapshot_positions'] for p in pages),
            'titles_multiset_equals_snapshot_excluding_single_presence':titles==expected,
            'pages_with_matching_title_and_presence_count':sum(bool(p['same_title_and_presence_count_positions']) for p in pages),
            'ambiguous_pages':[p['page'] for p in pages if len(p['same_title_and_presence_count_positions'])>1],
            'unmatched_title_counts':dict(titles-expected),'missing_title_counts':dict(expected-titles),
            'source_hash_unchanged':hashlib.sha256(pdf_path.read_bytes()).hexdigest()==initial,
            'no_probability_values_or_original_TF_arrays_recovered':True,
            'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pages','pdf','metadata')},ensure_ascii=True),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pdf',type=Path,required=True);ap.add_argument('--cardinality',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();run(args.pdf,args.cardinality,args.output)
