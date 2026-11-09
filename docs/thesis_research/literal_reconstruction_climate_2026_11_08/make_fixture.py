"""Write synthetic_fixture.json: artificial cases for reconstruction_climate. No real pollen, TF or climate value."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
NA = 'NA'
X1, X2 = [-5, 0, 5, 10], [100, 200, 300, 400]
T1, T2 = [[1, 0, 0, 0], [0, 0, 1, 3]], [[1, 1, 0, 0], [0, 2, 1, 0]]


def c(core, tf1=T1, tf2=T2, recent=True, hand=None, **extra):
    return {'core': [[str(v) for v in r] for r in core], 'tf_x1': [[str(v) for v in r] for r in tf1],
            'tf_x2': [[str(v) for v in r] for r in tf2], 'recent': recent, 'hand': hand, **extra}


CASES = {
    'recent_basic': c([[1, 3], [9, 9]], hand={'tann': '25/3', 'pann': '220', 'tann_weights': ['1', '0', '3', '9'],
                                               'pann_weights': ['1', '7', '3', '0']}),
    'only_first_row_counts': c([[1, 3], [1000, 0], [0, 7]], hand={'tann': '25/3', 'pann': '220'},
                               note='the same first row as recent_basic with other later rows'),
    'prior_when_not_recent': c([['bad', NA]], tf1=[[NA]], tf2=[[NA]], recent=False, hand={'tann': '37/2', 'pann': '640'},
                               note='nothing but the prior is read: R never forces the other arguments'),
    'first_row_zero': c([[0, 0], [1, 1]], hand={'tann': NA, 'pann': NA}),
    'na_in_one_tf_cell': c([[1, 3]], tf1=[[1, 0, NA, 0], [0, 0, 1, 3]], hand={'tann': NA, 'pann': '220'}),
    'na_tf_at_zero_share': c([[0, 3]], tf1=[[NA, NA, NA, NA], [0, 0, 1, 3]], hand={'tann': NA, 'pann': None},
                             note='0 * NA is NA, a zero share does not protect against the NA row'),
    'na_in_first_row_share': c([[NA, 3]], hand={'tann': NA, 'pann': NA}),
    'inexact_shares': c([['1/3', '2/7'], ['1/9', '1/11']], tf1=[['1/10', '2/10', '3/10', '4/10'], ['1/3', '1/7', '1/9', '1/11']],
                        tf2=[['1/13', '1/17', '1/19', '1/23'], ['1/29', '1/31', '1/37', '1/41']]),
}


def main():
    fixture = {'kind': 'synthetic_reconstruction_climate_fixture_not_historical', 'historical_reproduction_ready': False,
               'selected_final_variant': None, 'x1range': X1, 'x2range': X2, 'prior': {'mean_tmp': '37/2', 'mean_pann': '640'},
               'cases': CASES, 'stress': {'seed': 20261109, 'cases': 200}}
    (OUT / 'synthetic_fixture.json').write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(len(CASES), 'cases')


if __name__ == '__main__':
    main()
