"""Independent retained-array and omitted-residual derivation checks, audit only."""
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.polynomial import polyadd,polymul,polyder
from deterministic_reference import eng

B=Path(__file__).resolve().parent
rng=np.random.default_rng(20261029)
records=[]
for degree in [2,4]:
    for spot in [True,False]:
        before=after=quadratic=free=residual_before=residual_after=0.
        for _ in range(50):
            theta,k1,k2,beta,eps=rng.uniform(.1,.8),rng.uniform(.5,8),rng.uniform(.5,8),rng.uniform(-3,3),rng.uniform(.3,3)
            phi=(-.5 if spot else .5)+1j*rng.uniform(0,8);psi=.03-.02j
            v2=beta*beta+eps*eps;lam=0. if spot else beta*theta**2
            kp=k1+k2*theta-(0. if spot else 2*beta*theta);k2p=k2-(0. if spot else beta)
            c=np.array([theta**2,2*theta,1.]);d=np.array([lam,-kp,-k2p])-beta*phi*c
            expected_m=np.zeros((degree+1,degree+1,degree+1),complex)
            expected_l=np.zeros((degree+1,degree+1),complex)
            for k in range(degree+1):
                for n in range(1,degree+1):
                    if 0<=k-n+1<3:expected_l[k,n]+=n*d[k-n+1]
                    if 0<=k-n+2<3:expected_l[k,n]+=.5*v2*n*(n-1)*c[k-n+2]
                    for j in range(1,degree+1):
                        if 0<=k-n-j+2<3:expected_m[k,n,j]=.5*v2*n*j*c[k-n-j+2]
            expected_h=np.zeros(degree+1,complex);expected_h[:3]=.5*(phi*(phi+(1 if spot else -1))-2*psi)*c
            order=eng.afe.ExpansionOrder.FIRST if degree==2 else eng.afe.ExpansionOrder.SECOND
            m,l,h=eng.afe.func_a_ode_quadratic_terms(theta,k1,k2,beta,eps,phi,psi,spot,order)
            before=max(before,float(np.max(abs(l-expected_l))))
            if degree==4:l[4,4]+=4*v2;l[2,3]+=3*lam;l[3,4]+=4*lam
            after=max(after,float(np.max(abs(l-expected_l))))
            quadratic=max(quadratic,float(np.max(abs(m-expected_m))))
            free=max(free,float(np.max(abs(h-expected_h))))
            a=rng.normal(size=degree+1)+1j*rng.normal(size=degree+1)
            full=polyadd(polymul(d,polyder(a)),.5*v2*polymul(c,polyadd(polyder(a,2),polymul(polyder(a),polyder(a)))))
            full=polyadd(full,.5*(phi*(phi+(1 if spot else -1))-2*psi)*c)
            if degree==2:
                paper=np.array([2*a[2]*(v2*(2*theta*a[2]+a[1])-beta*phi-k2),2*v2*a[2]**2])
                fixed=paper.copy();fixed[0]+=2*a[2]*(k2-k2p)
            else:
                a1,a2,a3,a4=a[1:]
                paper=np.array([3*v2*a3*(3*theta*a3+2*a2)+4*a4*(v2*(3*theta**2*a3+4*theta*a2+a1)-k2),
                    .5*v2*(16*theta**2*a4**2+16*a4*(3*theta*a3+a2)+9*a3**2),
                    4*v2*a4*(4*theta*a4+3*a3),8*v2*a4**2])
                fixed=paper.copy();fixed[0]+=4*a4*(k2-k2p-beta*phi)
            residual_before=max(residual_before,float(np.max(abs(paper-full[degree+1:]))))
            residual_after=max(residual_after,float(np.max(abs(fixed-full[degree+1:]))))
        assert max(after,quadratic,free,residual_after)<1e-9
        if degree==4:assert before>1 and residual_before>1
        if degree==2 and not spot:assert residual_before>1
        records.append(dict(degree=degree,measure='MMA' if spot else 'inverse',cases=50,
            current_linear_max_error=before,corrected_linear_max_error=after,quadratic_max_error=quadratic,
            free_max_error=free,paper_residual_max_error=residual_before,corrected_residual_max_error=residual_after))
report={'scope':'Flat backbone; complete retained arrays and discarded polynomial coefficients',
        'cases':records,'published_equations':['4.17','4.19','4.25','4.27'],
        'additional_printed_typos':{'4.17_L22':'vartheta must be vartheta squared','4.25_L23':'q must be theta'},
        'lemma_A2_counterexample':{'ODE':'A_prime = 1; A(0) = 0','time':1,'true_norm_squared':1,'printed_bound':0},
        'status':'No production or manuscript files modified'}
(B/'paper_derivation_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(report,indent=2))
