from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_tf_plot_pages import read_page_marks
FONTS={'/F1':'/ZapfDingbats','/F2':'/Helvetica-Bold','/F3':'/Helvetica'}


def test_counts_only_explicit_green_dingbat_marks_and_keeps_title_text():
    ops=[(['/F1',12],b'Tf'),([.18,.545,.341],b'scn'),(['lll'],b'Tj'),
         (['/F3',12],b'Tf'),(['l'],b'Tj'),
         (['/F2',12],b'Tf'),([['Abies',-10,' alba']],b'TJ')]
    assert read_page_marks(ops,FONTS)=={'title':'Abies alba','green_point_glyph_count':3}


def test_nested_graphics_state_restores_font_and_color():
    ops=[(['/F1',12],b'Tf'),([.18,.545,.341],b'rg'),([],b'q'),
         (['/F2',12],b'Tf'),(['Name'],b'Tj'),([],b'Q'),(['ll'],b'Tj')]
    assert read_page_marks(ops,FONTS)=={'title':'Name','green_point_glyph_count':2}


@pytest.mark.parametrize('glyph,color',[('x',[.18,.545,.341]),('l',[0,0,0])])
def test_unexpected_glyph_or_color_is_not_silently_counted(glyph,color):
    with pytest.raises(ValueError):
        read_page_marks([(['/F1',12],b'Tf'),(color,b'rg'),([glyph],b'Tj')],FONTS)


def test_forms_are_not_ignored():
    with pytest.raises(ValueError,match='XObject'):
        read_page_marks([(['/Form1'],b'Do')],FONTS)


def test_untitled_page_is_not_assigned_from_order():
    with pytest.raises(ValueError,match='No bold title'):
        read_page_marks([],FONTS)
