"""Use the existing conditional split-step pricer with the constant kernel K=1."""
import pricing_engine as eng
import numpy as np
import warnings
from scipy.special import ndtri
from scipy.stats import qmc
from stochvolmodels.pricers.logsv_pricer import rough_logsv_mc_chain_pricer_split_step


def bridge_normals(power=13,steps=128,seed=291026):
    """Scrambled Sobol Brownian bridge; input format is the existing pricer's Z0."""
    assert steps&(steps-1)==0
    uniforms=qmc.Sobol(d=steps,scramble=True,bits=64,seed=seed).random_base2(power)
    z=ndtri(np.clip(uniforms,2.**-53,1.-2.**-53)).T
    w=np.zeros((steps+1,z.shape[1]));w[-1]=z[0]
    queue=[(0,steps)];col=1
    while queue:
        left,right=queue.pop(0)
        mid=(left+right)//2
        if mid==left: continue
        w[mid]=((right-mid)*w[left]+(mid-left)*w[right])/(right-left)+np.sqrt((mid-left)*(right-mid)/(steps*(right-left)))*z[col]
        col+=1;queue.extend([(left,mid),(mid,right)])
    assert col==steps
    return np.ascontiguousarray(np.diff(w,axis=0)*np.sqrt(steps))


class MonteCarlo:
    def __init__(self,chain,power=13,steps=128,seed=291026):
        self.chain=chain;self.power=power;self.steps=steps;self.seed=seed
        self.z=bridge_normals(power,steps,seed)
        self.zero=np.zeros_like(self.z)
        self.grids=[np.linspace(0.,float(t),steps+1) for t in chain.ttms]

    def prices(self,p):
        c=self.chain
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            prices,unused_iid_errors=rough_logsv_mc_chain_pricer_split_step(
                ttms=c.ttms,forwards=c.forwards,discfactors=c.discfactors,
                strikes_ttms=c.strikes_ttms,optiontypes_ttms=c.optiontypes_ttms,
                Z0=self.z,Z1=self.zero,sigma0=p.sigma0,theta=p.theta,kappa1=p.kappa1,kappa2=p.kappa2,
                beta=p.beta,orthog_vol=p.volvol,weights=np.array([1.]),nodes=np.array([0.]),timegrids=self.grids)
        self.warnings=[str(w.message) for w in caught]
        if any('floored' in w for w in self.warnings):raise FloatingPointError('Floored volatility paths')
        # IID path standard errors are invalid for Sobol draws; use independent scrambles.
        return list(prices)

    def coarsen(self):
        other=object.__new__(MonteCarlo)
        other.chain=self.chain;other.power=self.power;other.steps=self.steps//2;other.seed=self.seed
        other.z=np.ascontiguousarray(self.z.reshape((other.steps,2,-1)).sum(axis=1)/np.sqrt(2.))
        other.zero=np.zeros_like(other.z)
        other.grids=[np.linspace(0.,float(t),other.steps+1) for t in self.chain.ttms]
        return other
