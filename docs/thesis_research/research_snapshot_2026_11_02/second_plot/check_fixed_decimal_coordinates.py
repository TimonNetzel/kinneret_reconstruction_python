"""Independent fixed-decimal check of observed PDF point translations only."""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import hashlib,json,re,sys
repo=Path(__file__).resolve().parents[2];base=Path(__file__).parent
sys.path.insert(0,str(repo/'.local/console_provenance_v1/pdf_runtime'))
import pymupdf
raw=json.loads((base/'vector_comparison.json').read_text())
root=Path('<ORIGINAL_DATA_ROOT>/out/plots/trafo/2D')
docs={'second':pymupdf.open(root/'all_trafos.pdf'),'emd':pymupdf.open(root/'all_trafos_emd.pdf')}
pattern=re.compile(rb'/F1 1 Tf 2 Tr 4\.11 0 0 4\.11 (-?\d+\.\d{2}) (-?\d+\.\d{2}) Tm \(l\) Tj 0 Tr')
def coords(page):
 stream=page.read_contents()
 if stream.count(b'Tm (l) Tj')!=len(pattern.findall(stream)):
  raise ValueError('Point syntax outside explicit parser')
 return [(int(Decimal(x.decode())*100),int(Decimal(y.decode())*100)) for x,y in pattern.findall(stream)]
pages=[];total=Counter();all_checks=True
for pair in raw['pairs']:
 a=coords(docs['second'][pair['second_page']-1]);b=coords(docs['emd'][pair['emd_page']-1])
 refa=raw['pages']['second'][pair['second_page']-1];refb=raw['pages']['emd'][pair['emd_page']-1]
 count_ok=len(a)==len(b)==refa['filled_count']==refb['filled_count']
 if not count_ok:raise ValueError('Independent point count mismatch')
 diffs=Counter((x-u,y-v) for (x,y),(u,v) in zip(a,b));total.update(diffs)
 pages.append({'position':pair['position'],'second_page':pair['second_page'],'emd_page':pair['emd_page'],'point_count':len(a),
               'max_abs_dx_hundredth_pdf_point':max(abs(x) for x,y in diffs),'max_abs_dy_hundredth_pdf_point':max(abs(y) for x,y in diffs),
               'exactly_equal_points':diffs[(0,0)],'displacement_counts':[{'dx_hundredth_pdf_point':x,'dy_hundredth_pdf_point':y,'count':n} for (x,y),n in sorted(diffs.items())],
               'independent_counts_match':count_ok})
result={'method':'Exact Decimal parsing of all original F1/Tr=2/Tm/(l) point commands, integer hundredths of a PDF point. Independent of MuPDF text positions.',
 'all_556_page_point_counts_match_independent_trace':True,'point_pairs':sum(total.values()),
 'max_abs_dx_pdf_points':max(abs(x) for x,y in total)/100,'max_abs_dy_pdf_points':max(abs(y) for x,y in total)/100,
 'displacement_counts':[{'dx_hundredth_pdf_point':x,'dy_hundredth_pdf_point':y,'count':n} for (x,y),n in sorted(total.items())],
 'pages':pages,'no_historical_climate_values_or_TF_arrays_recovered':True,'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
target=base/'fixed_decimal_coordinates.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='pages'},indent=2))
# Complete representative pages, always use MuPDF to retain observed green glyphs.
for kind,page in [('second',1),('emd',1),('second',54),('emd',54),('second',76),('second',278)]:
 docs[kind][page-1].get_pixmap(matrix=pymupdf.Matrix(1.2,1.2),alpha=False).save(base/f'{kind}_page_{page}.png')