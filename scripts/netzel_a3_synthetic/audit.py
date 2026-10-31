"""Execute only the locked finite A3 examination; archive every rational row.

Run from the repository root:
  python -m scripts.netzel_a3_synthetic.audit --output docs/thesis_research/A3_SYNTHETIC_AUDIT
"""
import argparse
import csv
import hashlib
import io
import json
from fractions import Fraction as R
from itertools import product
from math import comb
from pathlib import Path
from . import kernel, reference
from .constants import CASES, CLIMATES, CURVES, Q, TRUTH, UA, UB, DESIGN_COMMIT, DESIGN_BLOB
from .invalid_cases import INVALID


LIMIT = 'Nur gesperrter künstlicher Fall; keine reale Gültigkeit/Deckung/Neuheit.'
NOTES = {
 'S00':'Alle Messereignisse fehlen; q*pC*pTau*pTheta.',
 'S01':'Grund-P/Y/D; Wahrheit (+1,(+1,+1,-1),b,1), kein Pflichtmaximum.',
 'S02-scaled':'S01 mit r(-1)=1,r(+1)=3; Q invariant.',
 'S02-witness':'Keine Evidenz; F=r*pC*pTau*pTheta, psi-Marginale q.',
 'S03':'C=(1,1,1),tau=a,theta=1; zukünftige A-Indikatoren mit gemeinsamem psi.',
 'S04':'S00; C1/C2-Markov-Abhängigkeit und negative Kontrolle.',
 'S06a':'Nur P1=(0,2,0),N=2; positive Likelihood, keine strukturelle Null.',
 'S06b':'Nur P1 ausdrücklich ignorierbar fehlend; Faktor 1.',
 'S06c':'Nur P1:A=1,B/R fehlend,N=2; Summe über B+R=1.',
 'S06d':'Nur P1=(0,0,0),N=0; Faktor 1.',
 'S07':'Nur P1 ersetzt: H in {0,1/4,3/4,1}, R0=0; Wahrheit H=1/4.',
 'S08a':'P/Y fehlen,D vorhanden; ganze Kurven mit Massen 3/4,1/4.',
 'S08b':'S01; Kurvenmarginale innerhalb psi normiert, dann q gemischt.',
 'S08c-a':'S01 mit tau=a fixiert; bedingte Verläufe je psi vergleichen.',
 'S08c-b':'S01 mit tau=b fixiert; bedingte Verläufe je psi vergleichen.',
 'S08d':'C=C*,theta=1; nur P2=(0,2,0),N=2 und D; Rückkanalzeuge.',
 'S09-base':'S01; vollständiger Grundfall als Konfliktreferenz.',
 'S09-opposite':'Nur Y=(-1,+1), P/D unverändert; keine Proxywahl.',
 'S09-noY':'Nur Y fehlt vollständig; keine Proxywahl.',
 'S10':'C=(1,1,1),tau=a; P/D fehlen; zukünftige Y teilen theta.',
 'S11-equal':'Nur UA/UB mit Prior 1:1,D fehlt; Grund-P/Y.',
 'S11-unequal':'Nur UA/UB mit Prior 3:1,D fehlt; Grund-P/Y.',
 'S05':'Zukünftige vollständige Komposition N=2,p=(1/2,1/4,1/4).',
}


def examine():
    tables = {name:[] for name in ('states','normalizers','marginals','predictive',
              'likelihoods','percent_latent','witnesses','invalid','countermethods','counterstates','comparisons','cases')}
    results = {}

    def check(case, criterion, actual, expected, relation='=='):
        ok = {'==':lambda:actual==expected, '!=':lambda:actual!=expected,
              '>':lambda:actual>expected, '<':lambda:actual<expected}[relation]()
        # Full arrays live in dedicated tables. Here report the number of
        # unequal entries instead of duplicating entire arrays in each cell.
        if isinstance(actual,dict) and isinstance(expected,dict):
            differences = sum(actual.get(k) != expected.get(k) for k in actual.keys() | expected.keys())
            actual,expected = differences,0
        delta = actual-expected if isinstance(actual,(int,R)) and isinstance(expected,(int,R)) else ('0' if ok else 'different')
        tables['witnesses'].append(dict(case=case,criterion=criterion,relation=relation,
            expected=str(expected),actual=str(actual),difference=str(delta),
            status='bestanden' if ok else 'fehlgeschlagen'))
        return ok

    for case in CASES:
        f,z,p = kernel.calculate(case)
        rf,rz,rp = reference.enumerate_case(case)
        km,rm = kernel.marginals(p),reference.marginal_table(rp)
        results[case] = f,z,p,km
        check(case,'F alle Zustände exakt',f,rf)
        check(case,'Z je psi exakt',z,rz)
        check(case,'Q alle Zustände exakt',p,rp)
        check(case,'alle Marginalen exakt',km,rm)
        check(case,'Summe Q',sum(p.values()),R(1))
        check(case,'alle Q nichtnegativ',all(v>=0 for v in p.values()),True)
        for psi in Q:
            check(case,f'psi={psi} bleibt q',km['psi',str(psi)],Q[psi])
            check(case,f'Z({psi}) positiv',z[psi],R(0),'>')
            tables['normalizers'].append(dict(case=case,psi=psi,kernel=z[psi],reference=rz[psi],difference=z[psi]-rz[psi]))
        for state in sorted(f):
            psi,c,t,h = state
            tables['states'].append(dict(case=case,psi=psi,C1=c[0],C2=c[1],C3=c[2],tau=t,theta=h,
                F_kernel=f[state],F_reference=rf[state],F_difference=f[state]-rf[state],
                Q_kernel=p[state],Q_reference=rp[state],Q_difference=p[state]-rp[state]))
        for label in sorted(km):
            tables['marginals'].append(dict(case=case,variable=label[0],value=label[1],
                kernel=km[label],reference=rm[label],difference=km[label]-rm[label]))

    for psi,likelihood in ((-1,R(1,3)),(1,R(2,3))):
        check('S01',f'q from E only,psi={psi}',Q[psi],R(1,2)*likelihood/(R(1,2)*R(1,3)+R(1,2)*R(2,3)))

    # Algebraic identities are deliberately separate from both posterior engines.
    prior = results['S00'][2]
    direct = {}
    for psi,c,t,h in prior:
        direct[psi,c,t,h] = Q[psi]*R(3 if c[0]==1 else 1,4)*R(3 if c[1]==c[0] else 1,4)*R(3 if c[2]==c[1] else 1,4)/4
    check('S00','direct product of priors',prior,direct)
    check('S02-scaled','Q invariant under r',results['S02-scaled'][2],results['S01'][2])
    check('S02-witness','Q equals prior despite r',results['S02-witness'][2],prior)

    predictive = {}
    for case in ('S03','S05','S10'):
        kp,rp = kernel.predictive(case),reference.predictive_table(case)
        predictive[case] = kp
        check(case,'entire predictive table',kp,rp)
        check(case,'predictive sum',sum(kp.values()),R(1))
        for outcome in sorted(kp):
            tables['predictive'].append(dict(case=case,outcome=str(outcome),kernel=kp[outcome],reference=rp[outcome],difference=kp[outcome]-rp[outcome]))
    for case,mean,second,variance in (
        ('S03',Q[-1]*R(1,4)+Q[1]*R(1,2),Q[-1]*R(1,16)+Q[1]*R(1,4),R(1,72)),
        ('S10',(R(3,4)+R(5,8))/2,(R(9,16)+R(25,64))/2,R(1,256))):
        pp = predictive[case]
        e1 = sum(m*x[0] for x,m in pp.items())
        e2 = sum(m*x[1] for x,m in pp.items())
        covariance = pp[1,1]-e1*e2
        check(case,'mean formula',e1,mean)
        check(case,'joint second moment',pp[1,1],second)
        check(case,'Cov = Var(shared parameter)',covariance,variance)
        check(case,'shared covariance positive',covariance,R(0),'>')

    p4 = results['S04'][2]
    pair = {(a,b):sum(m for (_,c,_,_),m in p4.items() if c[:2]==(a,b)) for a,b in product((-1,1),repeat=2)}
    direct_pair = {(a,b):R(3 if a==1 else 1,4)*R(3 if a==b else 1,4) for a,b in pair}
    check('S04','direct Markov pair table',pair,direct_pair)
    check('S04','Pr(C2=C1)',pair[-1,-1]+pair[1,1],R(3,4))
    ec1 = sum(a*m for (a,b),m in pair.items())
    ec2 = sum(b*m for (a,b),m in pair.items())
    cov4 = sum(a*b*m for (a,b),m in pair.items())-ec1*ec2
    check('S04','Cov(C1,C2)',cov4,R(3,8))
    check('S04','Cov positive',cov4,R(0),'>')
    p5 = predictive['S05']
    cov5 = sum(a*b*m for (a,b,r),m in p5.items())-sum(a*m for (a,b,r),m in p5.items())*sum(b*m for (a,b,r),m in p5.items())
    check('S05','fixed denominator every outcome',all(sum(x)==2 for x in p5),True)
    check('S05','Cov(nA,nB)=-N*pA*pB',cov5,-2*R(1,2)*R(1,4))
    check('S05','negative category covariance',cov5,R(0),'<')

    # Direct likelihood witnesses: no reference to kernel missingness helpers.
    for psi,c in product((-1,1),repeat=2):
        pa,pb = R(3+psi*c,8),R(3-psi*c,8)
        expected = {'S06a':pb**2,'S06b':R(1),'S06c':2*pa*(1-pa),'S06d':R(1),'S07':1-pa}
        modes = {'S06a':'zero_A','S06b':'absent','S06c':'partial_A','S06d':'empty','S07':'percent'}
        for case in expected:
            actual = kernel.pollen(modes[case],c,psi)
            check(case,f'P1 likelihood psi={psi},c={c}',actual,expected[case])
            tables['likelihoods'].append(dict(case=case,psi=psi,c=c,kernel=actual,reference=expected[case],difference=actual-expected[case]))
        check('S06a',f'measured zero positive psi={psi},c={c}',pb**2,R(0),'>')
        # Four latent H states, zero likelihood for reports excluded by R0=0.
        for h,weight in ((R(0),(1-pa)/2),(R(1,4),(1-pa)/2),(R(3,4),pa/2),(R(1),pa/2)):
            report = int(h>=R(1,2))
            conditional = weight/(1-pa) if report==0 else R(0)
            tables['percent_latent'].append(dict(psi=psi,c=c,H=h,prior=weight,report=report,posterior_R0_zero=conditional))
        check('S07',f'Pr(H=1/4 | R0=0),psi={psi},c={c}',((1-pa)/2)/(1-pa),R(1,2))
    check('S06d','N=0 Q equals missing P1 Q',results['S06d'][2],results['S06b'][2])

    check('S08a','tau=a date only',results['S08a'][3]['tau','a'],R(3,4))
    check('S08a','tau=b date only',results['S08a'][3]['tau','b'],R(1,4))
    check('S08b','same whole-curve posterior as S01',results['S08b'][2],results['S01'][2])
    base = results['S01'][2]
    # Derive each curve's measurement evidence using reference's independently
    # enumerated unnormalized table, then form the prescribed mixture formula.
    rf,rz,_ = reference.enumerate_case('S01')
    for t in ('a','b'):
        formula = sum(Q[psi]*sum(v for s,v in rf.items() if s[0]==psi and s[2]==t)/rz[psi] for psi in Q)
        check('S08b',f'whole curve formula {t}',results['S08b'][3]['tau',t],formula)
        # Fixing tau before the cut preserves q; conditioning mixed Q on tau
        # would reweight psi. Compare the conditionals *within* each psi.
        fixed = results['S08c-'+t][2]
        for psi in Q:
            denom = sum(v for s,v in base.items() if s[0]==psi and s[2]==t)
            check('S08c-'+t,f'conditional trajectory at psi={psi}',
                  {s:v/Q[psi] for s,v in fixed.items() if s[0]==psi and s[2]==t},
                  {s:v/denom for s,v in base.items() if s[0]==psi and s[2]==t})
    d = results['S08d'][3]
    actual_odds = d['psi_tau','1:a']/d['psi_tau','1:b']
    direct_odds = (R(1,2)*R(3,4)*R(1,4)**2)/(R(1,2)*R(1,4)*R(1,2)**2)
    check('S08d','tau a/b odds for psi=+1',actual_odds,direct_odds)
    check('S08d','likelihood ratio modifies D odds',actual_odds,R(3),'!=')
    for case in ('S09-base','S09-opposite','S09-noY'):
        check(case,'truth has positive posterior support',results[case][2][TRUTH],R(0),'>')
    check('S09-opposite','Q differs from base',results['S09-opposite'][2],base,'!=')
    check('S09-noY','Q differs from base',results['S09-noY'][2],base,'!=')
    # Direct equal measurement likelihood at UA and UB and direct prior odds.
    for psi in Q:
        direct_likelihoods = []
        for c,t,h in (UA,UB):
            k = (0,1) if t=='a' else (1,2)
            x,y = c[k[0]],c[k[1]]
            direct_likelihoods.append(R(3+psi*x,8)**2*R(3-psi*y,8)**2*(R(1,2)+R(x,4))*(R(1,2)-R(y,4)))
        check('S11-equal',f'equal likelihoods at psi={psi}',direct_likelihoods[0],direct_likelihoods[1])
        for case,odds in (('S11-equal',R(1)),('S11-unequal',R(3))):
            p = results[case][2]
            check(case,f'UA/UB posterior odds psi={psi}',p[(psi,)+UA]/p[(psi,)+UB],odds)

    # Intentionally defective procedures: full probability tables, not labels.
    def counter(name,case,dist,witness,expected):
        check(case,'counter '+name+' violates witness',witness,expected,'!=')
        tables['countermethods'].append(dict(method=name,case=case,expected_correct=expected,
            defective_actual=witness,difference=witness-expected,
            status='absichtlich defekt; erkannt' if witness!=expected else 'fehlgeschlagen: nicht erkannt'))
        for state,mass in sorted(dist.items(),key=lambda x:str(x[0])):
            tables['counterstates'].append(dict(method=name,case=case,state=str(state),mass=mass))
        check(case,'counter '+name+' distribution sum',sum(dist.values()),R(1))
    f = results['S02-witness'][0]
    total = sum(Q[s[0]]*v for s,v in f.items())
    global_q = {s:Q[s[0]]*v/total for s,v in f.items()}
    counter('global_normalization','S02-witness',global_q,
            sum(v for s,v in global_q.items() if s[0]==1),Q[1])
    check('S02-witness','global psi+ witness formula',sum(v for s,v in global_q.items() if s[0]==1),R(6,7))
    for case,name,pmean in (('S03','independent_psi',R(5,12)),('S10','independent_theta',R(11,16))):
        dist = {(a,b):pmean**(a+b)*(1-pmean)**(2-a-b) for a,b in product((0,1),repeat=2)}
        cov = dist[1,1]-pmean**2
        correct = R(1,72) if case=='S03' else R(1,256)
        check(case,'defective covariance zero',cov,R(0))
        counter(name,case,dist,cov,correct)
    independent = {(a,b,r):R(comb(2,a)*comb(2,b)*comb(2,r))*R(1,2)**2*R(1,4)**(b+r)*R(3,4)**(4-b-r)
                   for a,b,r in product(range(3),repeat=3)}
    violation = sum(v for x,v in independent.items() if sum(x)!=2)
    counter('independent_categories','S05',independent,violation,R(0))
    cm1 = {a:sum(v for (x,y),v in pair.items() if x==a) for a in (-1,1)}
    cm2 = {a:sum(v for (x,y),v in pair.items() if y==a) for a in (-1,1)}
    time_independent = {(a,b):cm1[a]*cm2[b] for a,b in pair}
    counter('independent_C_marginals','S04',time_independent,
            time_independent[-1,-1]+time_independent[1,1],R(3,4))
    counter('D_only_chronology','S08d',{'a':R(3,4),'b':R(1,4)},R(3),actual_odds)

    for ident,changes,expected in INVALID:
        meta = kernel.base_metadata()
        kwargs = {}
        for key,value in changes.items():
            if key in ('q','mutation'):
                kwargs[key] = value
            elif key == 'duplicate':
                meta['evidence_ids'] += (value,)
            else:
                meta[key] = value
        try:
            kernel.calculate('S01',metadata=meta,**kwargs)
            actual = 'Q_RETURNED'
        except kernel.Rejected as error:
            actual = str(error)
        check('S12',ident,actual,expected)
        tables['invalid'].append(dict(case=ident,input=repr(changes),expected=expected,actual=actual,
            Q_returned=actual=='Q_RETURNED',status='bestanden' if actual==expected else 'fehlgeschlagen',
            difference='0' if actual==expected else 'different',limit='Ablehnung dieses einen künstlichen Fehlereingangs.'))

    # Complete marginal changes: report all changes, including zeros and adverse
    # shifts. These differences are diagnostics, never variant selection.
    for case,baseline in (('S08b','S08a'),('S08c-a','S08b'),
                          ('S08c-b','S08b'),('S08c-b','S08c-a'),
                          ('S09-opposite','S09-base'),('S09-noY','S09-base'),
                          ('S06a','S01'),('S06b','S01'),('S06c','S01'),
                          ('S06d','S01'),('S07','S01'),
                          ('S11-unequal','S11-equal')):
        current,original = results[case][3],results[baseline][3]
        for label in sorted(current.keys() | original.keys()):
            a,b = current.get(label,R(0)),original.get(label,R(0))
            tables['comparisons'].append(dict(case=case,baseline=baseline,
                variable=label[0],value=label[1],actual=a,baseline_mass=b,difference=a-b))

    # Compact case-level evidence, with every subcase still present in raw tables.
    for case in (*CASES,'S05','S12'):
        ww = [x for x in tables['witnesses'] if x['case']==case]
        passed = all(x['status']=='bestanden' for x in ww)
        tables['cases'].append(dict(case=case,input=NOTES.get(case,'33 einzelne Mutationen I01–I09; siehe invalid.tsv'),
            expectation='Sämtliche gesperrten Gleichheiten, Ungleichheiten und Ablehnungen erfüllen.',
            actual=f'{len(ww)} Nachweise; '+('alle erfüllt' if passed else 'Abweichung vorhanden'),
            status='bestanden' if passed else 'fehlgeschlagen',difference='0' if passed else 'siehe witnesses.tsv',limit=LIMIT))
    return tables


def encode_rows(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def artifacts(tables):
    payload = {name+'.tsv':encode_rows(rows) for name,rows in tables.items()}
    manifest = dict(design_commit=DESIGN_COMMIT,design_blob=DESIGN_BLOB,
        arithmetic='fractions.Fraction; exact reduced rationals',
        posterior_cases=len(CASES),invalid_subcases=len(INVALID),
        failures=sum(row['status']=='fehlgeschlagen' for row in tables['witnesses']),
        files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),rows=len(tables[name[:-4]])) for name,data in payload.items()})
    payload['manifest.json'] = (json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    design = Path(__file__).resolve().parents[2] / 'docs/thesis_research/A3_SYNTHETIC_DESIGN.md'
    data = design.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if blob != DESIGN_BLOB:
        parser.error('Design blob differs from locked design; stop before calculation.')
    tables = examine()
    payload = artifacts(tables)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,data in payload.items():
        (args.output/name).write_bytes(data)
    failures = [x for x in tables['witnesses'] if x['status']!='bestanden']
    print(json.dumps(dict(posterior_cases=len(CASES),invalid_subcases=len(INVALID),
        witnesses=len(tables['witnesses']),failures=len(failures),output=str(args.output)),ensure_ascii=False))
    for failure in failures:
        print(json.dumps(failure,ensure_ascii=False))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
