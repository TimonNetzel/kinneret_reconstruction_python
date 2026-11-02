"""Write the artificial which.tf.missing fixture. All inputs are invented small numbers, never historical data.

Every case lists the taxa (one shared order), the explicit NA mask of the x1 means and the objects of line 1664.
Rows of a missing TF are null (NaN after the 0/0 row norm in 1581); null in a mean vector stands for NA. The hand
expectations follow from the source text alone: positions are removed in ascending order and the rest keeps its
order; without a missing TF num_taxa stays ncol(all.avail) from 1435; round(n/4) goes to the even digit on exact
ties (n = 2, 6, 10, 14, 18 give 0, 2, 2, 4, 4); half-up would give 1, 2, 3, 4, 5.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
SAMPLES, AGES, CELLS = 3, 2, 3

# name -> (taxa, missing 1-based positions, x2-only NA positions, taxa_change, hand expectations)
SPEC = {
    'empty_mask_keeps_1435': (5, [], [], False,
                              dict(num_taxa=5, line=1435, nr_taxa_sd=1, nr_taxa_sd_half_up=1, phi=2.5,
                                   hand='no TF missing: guard 1665 skips the removal; num_taxa 5 from 1435; '
                                        'round(1.25) = 1')),
    'remove_two_keeps_order': (6, [2, 5], [], False,
                               dict(num_taxa=4, line=1678, nr_taxa_sd=1, nr_taxa_sd_half_up=1, phi=2.0,
                                    hand='which() gives 2, 5; A, C, D, F remain in this order; round(1) = 1')),
    'tie_n2_to_even_zero': (4, [1, 4], [], False,
                            dict(num_taxa=2, line=1678, nr_taxa_sd=0, nr_taxa_sd_half_up=1, phi=1.0,
                                 hand='round(0.5) = 0 (even digit); half-up would give 1; phi1/phi2 stay 2 and 1')),
    'tie_n6_to_even_two': (7, [7], [], False,
                           dict(num_taxa=6, line=1678, nr_taxa_sd=2, nr_taxa_sd_half_up=2, phi=3.0,
                                hand='round(1.5) = 2: even digit and half-up agree')),
    'tie_n10_to_even_two': (12, [3, 9], [], False,
                            dict(num_taxa=10, line=1678, nr_taxa_sd=2, nr_taxa_sd_half_up=3, phi=5.0,
                                 hand='round(2.5) = 2 (even digit); half-up would give 3')),
    'tie_n18_to_even_four': (19, [10], [], False,
                             dict(num_taxa=18, line=1678, nr_taxa_sd=4, nr_taxa_sd_half_up=5, phi=9.0,
                                  hand='round(4.5) = 4; differences to half-up repeat for n %% 8 == 2')),
    'x2_only_na_is_kept': (4, [2], [3], False,
                           dict(num_taxa=3, line=1678, nr_taxa_sd=1, nr_taxa_sd_half_up=1, phi=1.5,
                                hand='only is.na(tf.x1.current.mean) counts: taxon C with NA x2 mean stays; '
                                     'round(0.75) = 1')),
    'taxa_change_true_branch': (5, [2], [], True,
                                dict(num_taxa=4, line=1678, nr_taxa_sd=1, nr_taxa_sd_half_up=1, phi=2.0,
                                     hand='the _change means lose position 2 as well (1671-1674); branch is '
                                          'inactive in the bound source (1500 & FALSE)')),
}


def _case(n_taxa, missing, x2_na, taxa_change, hand):
    gone = set(missing)
    taxa = [f'Kunsttaxon {chr(65 + t)}' for t in range(n_taxa)]
    all_avail = [[(s + 1) * 100 + t + 1 for t in range(n_taxa)] for s in range(SAMPLES)]
    all_avail_age = [[t + 1 + a / 2 for t in range(n_taxa)] for a in range(AGES)]
    tf = lambda shift: [None if t + 1 in gone else [(t + c + shift) / 10 for c in range(CELLS)]
                        for t in range(n_taxa)]
    case = {
        'taxa': taxa, 'x1_mean_is_na': [t + 1 in gone for t in range(n_taxa)],
        'all_avail': all_avail, 'all_avail_age': all_avail_age, 'taxa_appear_core': all_avail_age,
        'tf_x1': tf(1), 'tf_x2': tf(2),
        'x1_mean': [None if t + 1 in gone else 10 + t for t in range(n_taxa)],
        'x2_mean': [None if t + 1 in gone or t + 1 in x2_na else 300 + 10 * t for t in range(n_taxa)],
        'taxa_change': taxa_change,
        'x1_mean_change': [None if t + 1 in gone else 20 + t for t in range(n_taxa)] if taxa_change else None,
        'x2_mean_change': [None if t + 1 in gone else 400 + t for t in range(n_taxa)] if taxa_change else None,
    }
    kept = [name for t, name in enumerate(taxa) if t + 1 not in gone]
    case['hand'] = {'which_tf_missing': missing, 'kept_taxa': kept, 'num_taxa': hand['num_taxa'],
                    'num_taxa_line': hand['line'], 'nr_taxa_sd': hand['nr_taxa_sd'],
                    'nr_taxa_sd_half_up': hand['nr_taxa_sd_half_up'], 'phi': hand['phi'],
                    'stale_phi1': n_taxa / 2, 'stale_phi2': (n_taxa - 2) / 2, 'note': hand['hand']}
    return case


def main():
    cases = {name: _case(*spec) for name, spec in SPEC.items()}
    for name, c in cases.items():
        assert len(c['hand']['kept_taxa']) == c['hand']['num_taxa'], name
    fixture = {
        'kind': 'synthetic_tf_missing_fixture_not_historical',
        'historical_reproduction_ready': False, 'selected_final_variant': None,
        'note': 'Invented taxa, counts, TF rows and means. The NA mask is given explicitly; in the source it '
                'comes from weightedMedian (open). null stands for NA/NaN.',
        'cases': cases,
    }
    (OUT / 'synthetic_fixture.json').write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + '\n',
                                                encoding='utf-8', newline='\n')
    print(f'{len(cases)} cases written')


if __name__ == '__main__':
    main()
