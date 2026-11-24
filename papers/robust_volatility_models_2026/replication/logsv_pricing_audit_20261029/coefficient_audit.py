"""Compare current tensors and proposed corrections with independently derived algebra."""
import deterministic_reference as ref
from deterministic_reference import eng
from pathlib import Path
import numpy as np
from numpy.polynomial.polynomial import polymul,polyder,polyadd
import json,hashlib,difflib

def audit():
    rng=np.random.default_rng(291031);records=[]
    for is_spot in [True,False]:
        for eta in [.8,1.,1.2]:
            for degree,order in [(2,eng.afe.ExpansionOrder.FIRST),(4,eng.afe.ExpansionOrder.SECOND)]:
                before=[];after=[]
                for _ in range(24):
                    theta=rng.uniform(.1,1.);k1=rng.uniform(.5,8.);k2=rng.uniform(1.,10.);beta=rng.uniform(-2.,2.);eps=rng.uniform(.3,3.)
                    phi=(-.5 if is_spot else .5)+1j*rng.uniform(0,10);psi=-.2+.1j
                    a=rng.normal(size=degree+1)+1j*rng.normal(size=degree+1)
                    lam=0. if is_spot else beta*theta**2*eta
                    kap=k1+k2*theta-(0. if is_spot else 2*beta*theta*eta)
                    kap2=k2-(0. if is_spot else beta*eta)
                    s2=np.array([theta**2,2*theta,1.]);mu=np.array([lam,-kap,-kap2]);v2=beta**2+eps**2
                    expected=polyadd(polymul(polyadd(mu,-phi*beta*eta*s2),polyder(a)),
                        .5*v2*polymul(s2,polyadd(polyder(a,2),polymul(polyder(a),polyder(a)))))
                    expected=polyadd(expected,.5*eta**2*(phi*(phi+(1 if is_spot else -1))-2*psi)*s2)[:degree+1]
                    M,L,H=eng.afe.func_a_ode_quadratic_terms(theta,k1,k2,beta,eps,phi,psi,is_spot,order,eta)
                    got=np.einsum('i,kij,j->k',a,M,a)+L@a+H
                    before.append(float(np.max(abs(got-expected))))
                    if degree==4:
                        L[4,4]+=4*v2;L[2,3]+=3*lam;L[3,4]+=4*lam
                    got=np.einsum('i,kij,j->k',a,M,a)+L@a+H
                    after.append(float(np.max(abs(got-expected))))
                if degree==4:assert max(before)>1e-2
                assert max(after)<5e-12
                records.append({'measure':'MMA' if is_spot else 'inverse','eta':eta,'degree':degree,
                    'cases':24,'before_max_error':max(before),'after_max_error':max(after)})
    return records

if __name__=='__main__':
    base=Path(__file__).resolve().parent
    source=Path(eng.afe.__file__);text=source.read_text()
    replacements={
        'L[2, 3], L[2, 4] = 3.0 * (2.0 * qv - theta2 * beta * vol_backbone_eta * phi), 6.0 * qv2':
        'L[2, 3], L[2, 4] = 3.0 * (lamda + 2.0 * qv - theta2 * beta * vol_backbone_eta * phi), 6.0 * qv2',
        '4.0 * (3.0 * qv - theta2 * beta * vol_backbone_eta * phi)':
        '4.0 * (lamda + 3.0 * qv - theta2 * beta * vol_backbone_eta * phi)',
        '2.0 * (vartheta2 - 2.0 * kappa_p - 4.0 * theta * beta * vol_backbone_eta * phi)':
        '2.0 * (3.0 * vartheta2 - 2.0 * kappa_p - 4.0 * theta * beta * vol_backbone_eta * phi)'}
    corrected=text
    for old,new in replacements.items():assert corrected.count(old)==1;corrected=corrected.replace(old,new)
    diff=''.join(difflib.unified_diff(text.splitlines(True),corrected.splitlines(True),
        fromfile='a/src/stochvolmodels/pricers/logsv/affine_expansion.py',tofile='b/src/stochvolmodels/pricers/logsv/affine_expansion.py'))
    (base/'proposed_coefficients.patch').write_text(diff,encoding='utf-8')
    records=audit()
    from stochvolmodels.utils import mgf_pricer
    phi=mgf_pricer.get_phi_grid(max_phi=129,vol_scaler=.2)
    nan_result=mgf_pricer.vanilla_slice_pricer_with_mgf_grid(np.full(129,np.nan+0j),phi,1.,np.array([1.]),np.array(['C']))
    report={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'algebra_checks':records,'injected_all_nan_transform_call_price':[
            float(x) if np.isfinite(x) else str(x) for x in nan_result],
        'nan_test_interpretation':'All-NaN input returns NaN in the actual compiled kernel; no silent finite price was demonstrated. Explicit finite checks are still recommended.',
        'published_reference':'IJTAF 26(8) 2450003, equation 4.25, page 32',
        'patch_status':'proposed only; no installed or canonical file edited'}
    (base/'coefficient_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2))
