"""Independent quadrature/algebra sanity checks for the review's stationary laws.

No stochastic simulation or calibration is performed. Numerical checks supplement,
rather than replace, the scale/speed and integrability arguments in the audit.
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import kv, gamma

out=[]
for A,B,C,v in [(0.2,-0.3,0.4,0.8),(0.2,0.5,0.4,0.8),(0.7,0.32,0.2,0.8)]:
    q,b,eta=2*A/v**2,2*C/v**2,2*B/v**2-1
    norm=(b/q)**(eta/2)/(2*kv(eta,2*math.sqrt(q*b)))
    for r in [-2,-0.5,0,1,2]:
        integral=quad(lambda y:norm*math.exp((eta+r)*y-q*math.exp(-y)-b*math.exp(y)), -25,25,epsabs=1e-11,epsrel=1e-11)[0]
        formula=(q/b)**(r/2)*kv(eta+r,2*math.sqrt(q*b))/kv(eta,2*math.sqrt(q*b))
        err=abs(integral/formula-1)
        assert err<1e-9
        out.append({'family':'GIG','A':A,'B':B,'C':C,'v':v,'r':r,'quadrature':integral,'formula':formula,'relative_error':err})
    # Stationary zero-current equation: mu*pi = (1/2)*(v^2*s^2*pi)'.
    for s in np.geomspace(.01,100,20):
        dlog=(2*B/v**2-2)/s+2*A/(v**2*s*s)-2*C/v**2
        residual=A+B*s-C*s*s-0.5*v*v*(2*s+s*s*dlog)
        assert abs(residual)<1e-10*max(1,abs(A+B*s-C*s*s))

for family,alpha,scale in [('IG',1.7,0.6),('Gamma',1.2,1.8)]:
    for r in [-0.5,0,1]:
        if family=='IG':
            f=lambda y:scale**alpha/gamma(alpha)*math.exp((r-alpha)*y-scale*math.exp(-y))
            formula=scale**r*gamma(alpha-r)/gamma(alpha)
        else:
            f=lambda y:scale**alpha/gamma(alpha)*math.exp((r+alpha)*y-scale*math.exp(y))
            formula=scale**(-r)*gamma(alpha+r)/gamma(alpha)
        integral=quad(f,-60,60,epsabs=1e-10,epsrel=1e-10)[0]
        err=abs(integral/formula-1)
        assert err<1e-9
        out.append({'family':family,'shape':alpha,'scale_or_rate':scale,'r':r,'quadrature':integral,'formula':formula,'relative_error':err})

# Ito reciprocal identity for the 3/2 state: independent substitution on sample states.
for a,c,nu in [(1,0.4,.8),(1,-.5*.8**2,.8)]:
    for V in np.geomspace(.01,100,20):
        drift=-(a*V-c*V**2)/V**2+nu**2*V**3/V**3
        assert abs(drift-(c+nu**2-a/V))<1e-10
    assert 2*(c+nu**2)>=nu**2-1e-14

# The Heston table's rounded fitted parameters are well below Feller equality.
feller={}
for name,k,theta,nu in [('SPXW',24.26049,.03453,2.79164),('GLD',30,.06558,2.77467),('TLT',26.68251,.01156,1.71334)]:
    feller[name]={'2_kappa_theta':2*k*theta,'nu_squared':nu*nu,'ratio':2*k*theta/(nu*nu)}
    assert 2*k*theta<nu*nu

result={'checks':out,'max_moment_relative_error':max(r['relative_error'] for r in out),'zero_flux_residual_checks':'passed, 60 state/parameter combinations','reciprocal_ito_checks':'passed, 40 state/parameter combinations including equality','heston_feller_checks_from_rounded_table':feller,'scope':'Numerical/algebra sanity checks; boundary integrability and finite-horizon martingality established by the written arguments, not these checks.'}
Path(__file__).with_name('math_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
