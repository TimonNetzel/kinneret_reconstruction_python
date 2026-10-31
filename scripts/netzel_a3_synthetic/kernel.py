"""Finite factor kernel. No project imports, real inputs or float arithmetic."""
from fractions import Fraction as R
from itertools import product
from math import factorial
from .constants import CASES, DEFAULT, PSI, CLIMATES, CURVES, THETA, Q, TRUTH


class Rejected(ValueError):
    """A pre-normalization stop; no posterior is returned."""


def reject(status):
    raise Rejected(status)


def validate(metadata, q):
    if set(q) != set(PSI) or any(not isinstance(v, R) or v < 0 for v in q.values()) or sum(q.values()) != 1:
        reject('INVALID_CALIBRATION')
    if metadata.get('source') != 'synthetic' or metadata.get('unit') != 'dimensionless' or metadata.get('feedback') != 'allowed':
        reject('OUT_OF_DESIGN')
    if metadata.get('categories') != ('A','B','R'):
        reject('OUT_OF_DESIGN')
    if metadata.get('curve') not in CURVES.values():
        reject('INVALID_CHRONOLOGY')
    ids = metadata.get('evidence_ids', ())
    if len(ids) != len(set(ids)):
        reject('DUPLICATE_EVIDENCE')
    # parent_ids deliberately may repeat: an origin measurement ID may not.
    status = metadata.get('status')
    if status not in ('complete','absent_ignorable','partial_ignorable','percent'):
        reject('UNRESOLVED_STATUS')
    if metadata.get('pseudo_N') is not None:
        reject('PSEUDOCOUNT_FORBIDDEN')
    if status == 'percent':
        if metadata.get('precision') != 'S07':
            reject('MISSING_PRECISION_MODEL')
        return
    if status == 'absent_ignorable':
        return
    n, total = metadata.get('n'), metadata.get('N')
    if type(total) is not int or total < 0:
        reject('INVALID_OBSERVATION')
    if status == 'partial_ignorable':
        if metadata.get('coverage') != 'sum_B_R' or n != (1,None,None) or total != 2:
            reject('UNRESOLVED_STATUS')
    elif n is None or len(n) != 3:
        reject('UNRESOLVED_STATUS')
    elif any(type(x) is not int or x < 0 for x in n) or sum(n) != total:
        reject('INVALID_OBSERVATION')


def base_metadata():
    return dict(source='synthetic', unit='dimensionless', feedback='allowed',
                categories=('A','B','R'), curve=(1,2), status='complete',
                n=(2,0,0), N=2, evidence_ids=('E','D','P1','P2','Y1','Y2'),
                parent_ids=('archive','archive'))


def case_metadata(case):
    metadata = base_metadata()
    mode = (DEFAULT | CASES[case])['p1']
    if mode == 'zero_A':
        metadata['n'] = (0,2,0)
    elif mode == 'absent':
        metadata['status'] = 'absent_ignorable'
        metadata['n'] = metadata['N'] = None
    elif mode == 'empty':
        metadata.update(n=(0,0,0),N=0)
    elif mode == 'partial_A':
        metadata.update(status='partial_ignorable',n=(1,None,None),coverage='sum_B_R')
    elif mode == 'percent':
        metadata.update(status='percent',precision='S07',n=None,N=None)
    return metadata


def multinomial(n, p):
    total = sum(n)
    coefficient = factorial(total)
    for x in n:
        coefficient //= factorial(x)
    result = R(coefficient)
    for x, prob in zip(n, p):
        result *= prob ** x
    return result


def probabilities(c, psi):
    return (R(3+psi*c,8), R(3-psi*c,8), R(1,4))


def pollen(mode, c, psi):
    p = probabilities(c,psi)
    if mode in ('absent','empty'):
        return R(1)
    if mode == 'percent':
        return (1-p[0])/2 + (1-p[0])/2
    if mode == 'partial_A':
        return sum((multinomial((1,b,1-b), p) for b in (0,1)), R(0))
    return multinomial((2,0,0) if mode == 'counts' else (0,2,0), p)


def prior_c(c):
    return R(3 if c[0]==1 else 1,4) * R(3 if c[0]==c[1] else 1,4) * R(3 if c[1]==c[2] else 1,4)


def prior_u(c,t,h,cfg):
    if 'support' in cfg:
        return cfg['support'].get((c,t,h),R(0))
    weight = R(1)
    for key, val, base in (('c',c,prior_c(c)),('tau',t,R(1,2)),('theta',h,R(1,2))):
        weight *= R(val == cfg[key]) if key in cfg else base
    return weight


def factor(c,t,h,psi,cfg):
    indices = CURVES[t]
    result = prior_u(c,t,h,cfg) * (R(3 if t=='a' else 1,4) if cfg['d'] else 1)
    result *= pollen(cfg['p1'], c[indices[0]-1], psi)
    if cfg['p2']:
        result *= pollen('zero_A', c[indices[1]-1], psi)
    if cfg['y'] is not None:
        for j,y in enumerate(cfg['y']):
            result *= R(1,2)+R(y*c[indices[j]-1],4*h)
    return result * cfg['scale'][psi]


def calculate(case, *, metadata=None, q=None, mutation=None):
    if case not in CASES:
        reject('OUT_OF_DESIGN')
    q = Q if q is None else q
    expected_metadata = case_metadata(case)
    validate(expected_metadata if metadata is None else metadata, q)
    if q != Q or (metadata is not None and metadata != expected_metadata):
        reject('OUT_OF_DESIGN')
    if mutation not in (None,'zero_plus','negative','infinity','undefined'):
        reject('OUT_OF_DESIGN')
    cfg = dict(DEFAULT, **CASES[case])
    states = list(product(CLIMATES,CURVES,THETA))
    if 'support' in cfg:
        states = list(cfg['support'])
    f = {}
    for psi in PSI:
        for c,t,h in states:
            value = factor(c,t,h,psi,cfg)
            if mutation == 'zero_plus' and psi == 1:
                value = R(0)
            if mutation in ('negative','infinity','undefined') and (psi,c,t,h) == TRUTH:
                value = {'negative':R(-1),'infinity':float('inf'),'undefined':None}[mutation]
            if not isinstance(value,R) or value < 0:
                reject('INVALID_KERNEL')
            f[(psi,c,t,h)] = value
    z = {psi:sum((v for s,v in f.items() if s[0]==psi),R(0)) for psi in PSI}
    if any(v <= 0 for psi,v in z.items() if q[psi] > 0):
        reject('INVALID_NORMALIZER')
    posterior = {s:q[s[0]]*v/z[s[0]] for s,v in f.items()}
    return f,z,posterior


def marginals(posterior):
    # Include zero masses outside the explicitly restricted S11 support too.
    result = {('psi',str(s)):R(0) for s in PSI}
    result.update({('C',str(c)):R(0) for c in CLIMATES})
    result.update({('tau',t):R(0) for t in CURVES})
    result.update({('theta',str(h)):R(0) for h in THETA})
    for name in ('C1','C2','C3','C_at_d1','C_at_d2'):
        result.update({(name,str(c)):R(0) for c in PSI})
    for s in PSI:
        result.update({('psi_tau',f'{s}:{t}'):R(0) for t in CURVES})
        result.update({('psi_C',f'{s}:{c}'):R(0) for c in CLIMATES})
        result.update({('psi_theta',f'{s}:{h}'):R(0) for h in THETA})
    for state,mass in posterior.items():
        psi,c,t,h = state
        labels = [('psi',str(psi)),('C',str(c)),('tau',t),('theta',str(h))]
        labels += [(f'C{i+1}',str(ci)) for i,ci in enumerate(c)]
        labels += [(f'C_at_d{j+1}',str(c[k-1])) for j,k in enumerate(CURVES[t])]
        labels += [('psi_tau',f'{psi}:{t}'),('psi_C',f'{psi}:{c}'),('psi_theta',f'{psi}:{h}')]
        for label in labels:
            result[label] = result.get(label,R(0)) + mass
    return result


def predictive(case):
    if case == 'S03':
        return {(b1,b2):sum((Q[s]*R(3+s,8)**(b1+b2)*(1-R(3+s,8))**(2-b1-b2) for s in PSI),R(0))
                for b1,b2 in product((0,1),repeat=2)}
    if case == 'S10':
        return {(b1,b2):sum((R(1,2)*(R(1,2)+R(1,4*h))**(b1+b2)*(R(1,2)-R(1,4*h))**(2-b1-b2) for h in THETA),R(0))
                for b1,b2 in product((0,1),repeat=2)}
    if case == 'S05':
        return {(a,b,2-a-b):multinomial((a,b,2-a-b),probabilities(1,1))
                for a in range(3) for b in range(3-a)}
    reject('OUT_OF_DESIGN')
