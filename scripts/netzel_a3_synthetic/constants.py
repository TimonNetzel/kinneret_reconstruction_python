"""Only declared constants shared by the two A3 calculations (no helpers)."""
from fractions import Fraction as R

DESIGN_COMMIT = 'a4c03c6752708dead6ecd98f016758538ddb71b2'
DESIGN_BLOB = '21879814c77f0341457f9f4790dd40e24b555d55'
PSI = (-1, 1)
CLIMATES = ((-1,-1,-1), (-1,-1,1), (-1,1,-1), (-1,1,1),
            (1,-1,-1), (1,-1,1), (1,1,-1), (1,1,1))
CURVES = {'a': (1, 2), 'b': (2, 3)}
THETA = (1, 2)
Q = {-1: R(1,3), 1: R(2,3)}
TRUTH = (1, (1,1,-1), 'b', 1)
UA = ((1,-1,-1), 'a', 1)
UB = ((1,1,-1), 'b', 1)
# Overrides of S01, exactly as in the locked table. Future measurements are
# evaluated separately; they are never supplied as posterior observations.
CASES = {
 'S00': {'p1':'absent','p2':False,'y':None,'d':False},
 'S01': {},
 'S02-scaled': {'scale': {-1:R(1), 1:R(3)}},
 'S02-witness': {'p1':'absent','p2':False,'y':None,'d':False,
                 'scale': {-1:R(1), 1:R(3)}},
 'S03': {'p1':'absent','p2':False,'y':None,'d':False,
         'c':(1,1,1),'tau':'a','theta':1},
 'S04': {'p1':'absent','p2':False,'y':None,'d':False},
 'S06a': {'p1':'zero_A'},
 'S06b': {'p1':'absent'},
 'S06c': {'p1':'partial_A'},
 'S06d': {'p1':'empty'},
 'S07': {'p1':'percent'},
 'S08a': {'p1':'absent','p2':False,'y':None},
 'S08b': {},
 'S08c-a': {'tau':'a'},
 'S08c-b': {'tau':'b'},
 'S08d': {'c':(1,1,-1),'theta':1,'p1':'absent','y':None},
 'S09-base': {},
 'S09-opposite': {'y':(-1,1)},
 'S09-noY': {'y':None},
 'S10': {'c':(1,1,1),'tau':'a','p1':'absent','p2':False,
         'd':False,'y':None},
 'S11-equal': {'support': {UA:R(1,2), UB:R(1,2)},'d':False},
 'S11-unequal': {'support': {UA:R(3,4), UB:R(1,4)},'d':False},
}
DEFAULT = {'p1':'counts','p2':True,'y':(1,-1),'d':True,
           'scale':{-1:R(1),1:R(1)}}
