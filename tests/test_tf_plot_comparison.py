"""Targeted tests for scientific PDF coordinate comparison, using synthetic marks."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).resolve().parents[1]/'tools/audit_tf_plot_comparison.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def trace(text, seq, origin=(0,0), bbox=(0,0,10,10), font='Helvetica', kind=0, color=(0,0,0)):
    return {'font':font,'type':kind,'color':color,'seqno':seq,'bbox':bbox,'dir':(1,0),
            'chars':[(ord(c),0,(origin[0]+i,origin[1]),(0,0,1,1)) for i,c in enumerate(text)]}


class P:
    rect=(0,0,576,576)
    def __init__(self):
        self.traces=[trace('-10',1,bbox=(100,535,110,547)),trace('10',2,bbox=(300,535,310,547)),
                     trace('0',3,bbox=(30,400,40,412)),trace('1000',4,bbox=(30,200,50,212)),
                     trace('G',10,(150,250),font='ZapfDingbats',color=(.18,.545,.341)),
                     trace('G',11,(150,250),font='ZapfDingbats',kind=1,color=(.18,.545,.341)),
                     trace('G',12,(150,250),font='ZapfDingbats',color=(.18,.545,.341)),
                     trace('G',13,(150,250),font='ZapfDingbats',kind=1,color=(.18,.545,.341)),
                     trace('Taxon',20,font='Helvetica-Bold')]
        self.draws=[{'seqno':0,'items':[('l',Point(100,518.4),Point(100,525.6))]},
                    {'seqno':0,'items':[('l',Point(300,518.4),Point(300,525.6))]},
                    {'seqno':0,'items':[('l',Point(72,400),Point(64.8,400))]},
                    {'seqno':0,'items':[('l',Point(72,200),Point(64.8,200))]}]
    def get_texttrace(self):return self.traces
    def get_drawings(self):return self.draws


class Point:
    def __init__(self,x,y):self.x=x;self.y=y
    def __repr__(self):return f'Point({self.x},{self.y})'


def test_overlapping_glyphs_preserved_fill_and_stroke_not_doubled():
    r=m.read_page(P());assert r['filled_count']==r['stroked_count']==2;assert r['fill_stroke_positions_identical']


def test_tick_values_bound_to_vector_ticks_and_y_inversion():
    r=m.read_page(P());assert r['axes']['temperature']['units_per_pdf_point']==.1
    assert r['axes']['precipitation']['units_per_pdf_point']==-5
    assert r['axes']['temperature']['max_tick_rounding_residual']==0


def test_equal_counts_do_not_imply_equal_coordinates():
    a=P();b=P();b.traces[4]['chars'][0]=(71,0,(151,250),(0,0,1,1))
    ra=m.read_page(a);rb=m.read_page(b)
    assert ra['filled_count']==rb['filled_count'];assert ra['point_multiset_sha256']!=rb['point_multiset_sha256']
    assert not rb['fill_stroke_positions_identical']


def test_wrong_color_is_flagged():
    p=P();p.traces[4]['color']=(0,0,0);assert not m.read_page(p)['all_point_colors_expected']


def test_uninspected_point_mode_rejected():
    p=P();p.traces[4]['type']=3
    with pytest.raises(ValueError,match='painting mode'):m.read_page(p)


def test_tick_mismatch_rejected():
    p=P();p.draws.pop()
    with pytest.raises(ValueError,match='axis tick'):m.read_page(p)


def test_missing_point_font_rejected():
    p=P();p.traces=[t for t in p.traces if t['font']!='ZapfDingbats']
    with pytest.raises(ValueError,match='Missing observed point'):m.read_page(p)


def test_typographic_minus_mapping_and_no_fuzzy_taxon_match():
    assert m.numeric('\u221210')==-10
    assert m.numeric('10%') is None
    assert m.read_page(P())['title']=='Taxon'