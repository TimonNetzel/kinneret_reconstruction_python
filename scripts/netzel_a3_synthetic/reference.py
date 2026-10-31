"""Independent enumeration: imports only constants and standard arithmetic.

No kernel factors, normalizers, missingness or marginalization helpers.
The kernel's composition sum is replaced by direct binomial/polynomial forms.
"""
from fractions import Fraction as R
from .constants import CASES, DEFAULT, Q, CLIMATES, PSI, THETA


def enumerate_case(case):
    settings = DEFAULT | CASES[case]
    weights = {}
    # Independent nested enumeration, including a direct Markov formula.
    for sign in (-1,1):
        for curve, indices in (('a',(0,1)), ('b',(1,2))):
            for noise in (1,2):
                for c1 in (-1,1):
                    for c2 in (-1,1):
                        for c3 in (-1,1):
                            climate = (c1,c2,c3)
                            u = (climate,curve,noise)
                            if 'support' in settings:
                                if u not in settings['support']:
                                    continue
                                prior = settings['support'][u]
                            else:
                                pc = R((3 if c1==1 else 1)*(3 if c1==c2 else 1)*(3 if c2==c3 else 1),64)
                                if 'c' in settings:
                                    pc = R(climate == settings['c'])
                                pt = R(curve == settings['tau']) if 'tau' in settings else R(1,2)
                                ph = R(noise == settings['theta']) if 'theta' in settings else R(1,2)
                                prior = pc*pt*ph
                            a1 = R(3+sign*climate[indices[0]],8)
                            b1 = R(3-sign*climate[indices[0]],8)
                            b2 = R(3-sign*climate[indices[1]],8)
                            mode = settings['p1']
                            if mode == 'counts':
                                observation = a1*a1
                            elif mode == 'zero_A':
                                observation = b1*b1
                            elif mode == 'partial_A':
                                observation = 2*a1*(1-a1)
                            elif mode == 'percent':
                                observation = 1-a1
                            else:  # absent or the N=0 empty composition
                                observation = R(1)
                            if settings['p2']:
                                observation *= b2*b2
                            ys = settings['y']
                            if ys is not None:
                                observation *= R(2*noise+ys[0]*climate[indices[0]],4*noise)
                                observation *= R(2*noise+ys[1]*climate[indices[1]],4*noise)
                            date = R(3 if curve=='a' else 1,4) if settings['d'] else R(1)
                            weights[(sign,climate,curve,noise)] = prior*observation*date*settings['scale'][sign]
    # Accumulate each slice as it is enumerated, then normalize it separately.
    z = {-1:R(0),1:R(0)}
    for (sign,climate,curve,noise), weight in weights.items():
        z[sign] += weight
    joint = {}
    for sign in (-1,1):
        for state,weight in weights.items():
            if state[0] == sign:
                joint[state] = Q[sign]*weight/z[sign]
    return weights,z,joint


def marginal_table(joint):
    # Event-based summation instead of the kernel's per-state accumulation.
    events = {}
    for sign in PSI:
        events[('psi',str(sign))] = lambda s,sign=sign: s[0]==sign
        for curve in ('a','b'):
            events[('psi_tau',f'{sign}:{curve}')] = lambda s,sign=sign,curve=curve: s[0]==sign and s[2]==curve
        for c in CLIMATES:
            events[('psi_C',f'{sign}:{c}')] = lambda s,sign=sign,c=c: s[0]==sign and s[1]==c
        for h in THETA:
            events[('psi_theta',f'{sign}:{h}')] = lambda s,sign=sign,h=h: s[0]==sign and s[3]==h
    for c in CLIMATES:
        events[('C',str(c))] = lambda s,c=c: s[1]==c
    for curve in ('a','b'):
        events[('tau',curve)] = lambda s,curve=curve: s[2]==curve
    for h in THETA:
        events[('theta',str(h))] = lambda s,h=h: s[3]==h
    for i in range(3):
        for value in (-1,1):
            events[(f'C{i+1}',str(value))] = lambda s,i=i,value=value: s[1][i]==value
    for j in range(2):
        for value in (-1,1):
            events[(f'C_at_d{j+1}',str(value))] = lambda s,j=j,value=value: s[1][j+(s[2]=='b')]==value
    # Keep all declared event labels, including structural zero masses.
    return {label:sum((m for s,m in joint.items() if event(s)),R(0))
            for label,event in events.items()}


def predictive_table(case):
    if case == 'S05':
        # Six outcomes of two draws, evaluated by explicit ordered pairs.
        out = {(2,0,0):R(0),(1,1,0):R(0),(1,0,1):R(0),
               (0,2,0):R(0),(0,1,1):R(0),(0,0,2):R(0)}
        probs = (R(1,2),R(1,4),R(1,4))
        for first in range(3):
            for second in range(3):
                counts = tuple(int(first==j)+int(second==j) for j in range(3))
                out[counts] += probs[first]*probs[second]
        return out
    # Conditional two-indicator tables accumulated one latent component at a time.
    out = {(0,0):R(0),(0,1):R(0),(1,0):R(0),(1,1):R(0)}
    mixture = ((R(1,3),R(1,4)),(R(2,3),R(1,2))) if case=='S03' else ((R(1,2),R(3,4)),(R(1,2),R(5,8)))
    for weight,p in mixture:
        out[(0,0)] += weight*(1-p)*(1-p)
        out[(0,1)] += weight*(1-p)*p
        out[(1,0)] += weight*p*(1-p)
        out[(1,1)] += weight*p*p
    return out
