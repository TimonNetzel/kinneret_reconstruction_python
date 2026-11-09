"""Counterexamples on the stress cases, not candidate variants: what would change if a step of wt_power were done differently."""
import importlib.util
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('wt_check_a', OUT / 'check_wt_power.py')
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def changed(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return int(np.sum(a.real != b.real) + (np.sum(a.imag != b.imag) if np.iscomplexobj(a) or np.iscomplexobj(b) else 0)), int(a.size * (2 if np.iscomplexobj(a) or np.iscomplexobj(b) else 1))


def main():
    fixture = json.loads((OUT / 'synthetic_fixture.json').read_text(encoding='utf-8'))
    stress = check.stress_cases(fixture['stress'])
    totals = {k: [0, 0] for k in ('double_mean_instead_of_long_double', 're2_plus_im2_as_in_wt_cpp', 'scaled_ifft_times_n', 'real_input_to_fft')}

    def tally(key, pair):
        totals[key][0] += pair[0]
        totals[key][1] += pair[1]

    for case in stress:
        result = check.run_stress(case)
        x = np.array(result.x)
        d = np.array(case['d'], dtype=float)
        total = 0.0
        for v in d:  # plain double accumulation, in order
            total += v
        tally('double_mean_instead_of_long_double', changed(np.concatenate([d - total / case['l_d'], np.zeros(case['zeros'])]), x))
        wave = result.wave
        modulus = np.abs(wave)
        tally('re2_plus_im2_as_in_wt_cpp', changed(wave.real * wave.real + wave.imag * wave.imag, modulus * modulus))
        wb = np.array(case['wb'], dtype=float)
        scaled = np.fft.ifft(result.f[:, np.newaxis] * wb, axis=0) * result.n * case['invflen']
        tally('scaled_ifft_times_n', changed(scaled[:result.n_obs, :], wave))
        tally('real_input_to_fft', changed(np.fft.fft(x), result.f))
    report = {'kind': 'counterexamples_not_candidate_variants', 'stress_cases': len(stress), 'checks': {k: {'changed': v[0], 'compared': v[1]} for k, v in totals.items()}, 'historical_reproduction_ready': False}
    (OUT / 'arithmetic_check.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(report['checks'])


if __name__ == '__main__':
    main()
