
import numpy as np
from numba import njit
from stochvolmodels import npdf, infer_bsm_implied_vol


@njit
def compute_normal_pdf(x: np.ndarray):
    dx = x[1] - x[0]
    return dx*npdf(x)


@njit
def value_under_q_kernel(b: float, pdf: np.ndarray, x: np.ndarray, payoff: np.ndarray, forward: float = 1.0):
    c = -0.5 + (2.0*b+1.0)*np.log(forward)
    norm = np.exp(0.5*np.square(c)/(2.0*b+1))/np.sqrt(2.0*b+1)
    q_payoff = np.sum(pdf*np.exp(c*x-b*np.square(x))*payoff) / norm
    return q_payoff


@njit
def value_payoff(pdf: np.ndarray, payoff: np.ndarray):
    return np.sum(pdf*payoff)


x = np.linspace(-5.0, 5.0, 20000)
pdf = compute_normal_pdf(x)

print(f"sum={np.sum(pdf)}, mean={np.sum(x * pdf)}, std={np.sqrt(np.sum(np.square(x) * pdf) - np.square(np.sum(x * pdf)))}")

payoff = np.exp(x)
q_payoff = value_under_q_kernel(b=2.0, pdf=pdf, x=x, payoff=payoff, forward=1)
print(f"q_payoff={q_payoff}")

strikes = np.linspace(0.2, 2.0, 21)

values, model_vols = np.zeros_like(strikes), np.zeros_like(strikes)
values_q, model_vols_q = np.zeros_like(strikes), np.zeros_like(strikes)
for idx, strike in enumerate(strikes):
    spot = np.exp(x-0.5)
    payoff = np.maximum(spot-strike, 0.0)
    model_price = value_payoff(pdf=pdf, payoff=payoff)
    values[idx] = model_price
    model_vols[idx] = infer_bsm_implied_vol(forward=1.0, ttm=1.0, given_price=model_price, strike=strike, optiontype='C')
    payoff = np.maximum(np.exp(x)-strike, 0.0)
    model_price_q = value_under_q_kernel(b=0.25, pdf=pdf, x=x, payoff=payoff)
    values_q[idx] = model_price_q
    model_vols_q[idx] = infer_bsm_implied_vol(forward=1.0, ttm=1.0, given_price=model_price_q, strike=strike, optiontype='C')

print(f"values={values}")
print(f"values_q={values_q}")
print(f"model_vols={model_vols}")
print(f"model_vols_q={model_vols_q}")
