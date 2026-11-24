"""
Growth rate of the linear-drift second moment above the stability threshold, and related checks.

Model (paper eq. (1) with kappa2 = 0): v(t) = E[sigma_t^2] solves the renewal equation
    v = m^2 + h * v,   h = vartheta^2 G^2,   G(t) = t^{alpha-1} E_{alpha,alpha}(-kappa1 t^alpha),
with alpha = H + 1/2. The Laplace transform of the kernel (paper eq:h_hat) is, by Parseval's
identity for exp(-lambda t / 2) G(t),
    h_hat(lambda) = (vartheta^2 / pi) int_0^inf dw / |(lambda/2 + i w)^alpha + kappa1|^2,
with h_hat(0+) = gamma = vartheta^2 ||G||_2^2 and h_hat(lambda) = a lambda^{-2H} when kappa1 = 0,
a = vartheta^2 Gamma(2H) / Gamma(alpha)^2.
Growth-rate proposition: for gamma > 1 the unique root lambda* of h_hat(lambda*) = 1 gives, for
sigma_0 = theta,
    E[sigma_t^2] / theta^2 ~ exp(lambda* t) / (lambda* |h_hat'(lambda*)|),
with the constant -1 / (gamma - 1) as the next term (exact at H = 1/2).
Also here: the stability parameter gamma_N of a Markovian kernel K^N(t) = sum_i w_i exp(-x_i t),
the rough-Bergomi-equivalent critical eta_c = 2 vartheta_c / (sqrt(2H) Gamma(alpha)), and the
bound of the time-averaged second moment under quadratic drift (paper eq:cesaro),
    kappa2 A_T <= kappa1 theta + (kappa2 theta - kappa1)^+ sqrt(A_T) + sigma_0 L([0,T]) / T,
where L is the resolvent of the first kind of the kernel, K * L = 1. For the fractional kernel
L([0,T]) / T = T^{-alpha} / Gamma(2 - alpha). For K^N with distinct positive nodes
    L(dt) = delta_0(dt) / sum_i w_i + (a_0 + sum_j a_j exp(-r_j t)) dt,   a_0 = 1 / sum_i w_i / x_i,
where -r_j are the roots of P(z) = sum_i w_i prod_{k != i} (z + x_k) and a_j are the residues of
prod_i (z + x_i) / (z P(z)); L([0,T]) / T tends to a_0 = 1 / int_0^inf K^N.
Units: time in years, kappa1 in year^(-H-1/2), vartheta and eta in year^(-H).
Entry points: h_hat, growth_rate, log10_second_moment_ratio, markov_stability_parameter,
critical_eta, time_averaged_bound, markov_first_kind_resolvent, markov_time_averaged_bound,
and the cases of run_local_test.
Not in this module: simulation, and the product-integration solver of the second-moment
equation, which stays in rough_logsv_stationarity_check.py.
"""
# packages
import hashlib
import json
import platform
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma, rgamma
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
# project
from rough_logsv_stationarity_check import g2_norm

BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'


def _validate(H: float, kappa1: float, vartheta: float) -> None:
    if not 0.0 < H <= 0.5:
        raise ValueError(f"H must lie in (0, 1/2], got {H!r}")
    if kappa1 < 0.0:
        raise ValueError(f"kappa1 must be non-negative, got {kappa1!r}")
    if vartheta <= 0.0:
        raise ValueError(f"vartheta must be positive, got {vartheta!r}")


def _half_line_integral(integrand,
                        scale: float,
                        ) -> float:
    """int_0^inf integrand(w) dw, split at 4*scale; the tail uses w = split / s on (0, 1]."""
    split = 4.0 * scale
    head = quad(integrand, 0.0, split, epsabs=0.0, epsrel=1e-12, limit=400,
                points=[0.25 * scale, scale])[0]
    tail = quad(lambda s: integrand(split / s) * split / s ** 2, 0.0, 1.0,
                epsabs=0.0, epsrel=1e-12, limit=400)[0]
    return head + tail


def h_hat(lam: float,
          H: float,
          kappa1: float,
          vartheta: float,
          ) -> float:
    """Laplace transform of h = vartheta^2 G^2 at lam > 0 (paper eq:h_hat)."""
    _validate(H=H, kappa1=kappa1, vartheta=vartheta)
    if lam <= 0.0:
        raise ValueError(f"lam must be positive, got {lam!r}")
    alpha = H + 0.5
    if kappa1 == 0.0:
        return vartheta ** 2 * gamma(2.0 * H) / gamma(alpha) ** 2 * lam ** (-2.0 * H)

    def integrand(w: float) -> float:
        z = (0.5 * lam + 1j * w) ** alpha + kappa1
        return 1.0 / (z.real ** 2 + z.imag ** 2)

    scale = max(lam, kappa1 ** (1.0 / alpha))
    return vartheta ** 2 * _half_line_integral(integrand, scale=scale) / np.pi


def h_hat_derivative(lam: float,
                     H: float,
                     kappa1: float,
                     vartheta: float,
                     ) -> float:
    """d h_hat / d lam, by differentiation under the integral; negative."""
    _validate(H=H, kappa1=kappa1, vartheta=vartheta)
    if lam <= 0.0:
        raise ValueError(f"lam must be positive, got {lam!r}")
    alpha = H + 0.5
    if kappa1 == 0.0:
        a = vartheta ** 2 * gamma(2.0 * H) / gamma(alpha) ** 2
        return -2.0 * H * a * lam ** (-2.0 * H - 1.0)

    def integrand(w: float) -> float:
        base = 0.5 * lam + 1j * w
        z = base ** alpha + kappa1
        dz = 0.5 * alpha * base ** (alpha - 1.0)           # d z / d lam
        modulus2 = z.real ** 2 + z.imag ** 2
        return -2.0 * (np.conj(z) * dz).real / modulus2 ** 2

    scale = max(lam, kappa1 ** (1.0 / alpha))
    return vartheta ** 2 * _half_line_integral(integrand, scale=scale) / np.pi


def h_hat_series(lam: float,
                 H: float,
                 kappa1: float,
                 vartheta: float,
                 n_terms: int = 60,
                 ) -> float:
    """
    Large-lam expansion of h_hat, used as an independent check of the quadrature:
    h_hat = vartheta^2 sum_n (-kappa1)^n Gamma(2H + alpha n) c_n lam^{-2H - alpha n},
    c_n = sum_{j<=n} 1 / (Gamma(alpha (j+1)) Gamma(alpha (n-j+1))),
    from the termwise transform of the square of the Mittag-Leffler series of G.
    Reliable when kappa1 lam^{-alpha} is small.
    """
    _validate(H=H, kappa1=kappa1, vartheta=vartheta)
    alpha = H + 0.5
    if kappa1 * lam ** (-alpha) > 0.05:
        raise ValueError("series needs kappa1 * lam^(-alpha) <= 0.05, "
                         f"got {kappa1 * lam ** (-alpha)!r}")
    total = 0.0
    for n in range(n_terms):
        j = np.arange(n + 1)
        c_n = float(np.sum(rgamma(alpha * (j + 1)) * rgamma(alpha * (n - j + 1))))
        total += (-kappa1) ** n * gamma(2.0 * H + alpha * n) * c_n * lam ** (-2.0 * H - alpha * n)
    return vartheta ** 2 * total


def stability_gamma(H: float,
                    kappa1: float,
                    vartheta: float,
                    ) -> float:
    """gamma = vartheta^2 ||G||_2^2 (paper eq:gamma); infinite for kappa1 = 0."""
    _validate(H=H, kappa1=kappa1, vartheta=vartheta)
    return np.inf if kappa1 == 0.0 else vartheta ** 2 * g2_norm(H=H, kappa1=kappa1)


def growth_rate(H: float,
                kappa1: float,
                vartheta: float,
                ) -> Tuple[float, float]:
    """
    Root lambda* of h_hat(lambda*) = 1 and the prefactor 1 / (lambda* |h_hat'(lambda*)|).
    Requires gamma > 1. For kappa1 = 0 the root is a^{1/(2H)} and the prefactor is 1/(2H).
    """
    gamma_ = stability_gamma(H=H, kappa1=kappa1, vartheta=vartheta)
    if gamma_ <= 1.0:
        raise ValueError(f"growth rate needs gamma > 1, got gamma = {gamma_!r}")
    alpha = H + 0.5
    a = vartheta ** 2 * gamma(2.0 * H) / gamma(alpha) ** 2
    lam_upper = a ** (1.0 / (2.0 * H))                     # h_hat <= a lam^{-2H}
    if kappa1 == 0.0:
        return lam_upper, 1.0 / (2.0 * H)

    def objective(log_lam: float) -> float:
        return h_hat(lam=float(np.exp(log_lam)), H=H, kappa1=kappa1, vartheta=vartheta) - 1.0

    log_lower = np.log(lam_upper) - 60.0                   # h_hat -> gamma > 1 as lam -> 0
    log_lam = brentq(objective, log_lower, np.log(lam_upper), xtol=1e-14, rtol=1e-14)
    lam_star = float(np.exp(log_lam))
    slope = h_hat_derivative(lam=lam_star, H=H, kappa1=kappa1, vartheta=vartheta)
    return lam_star, 1.0 / (lam_star * abs(slope))


def log10_second_moment_ratio(t: float,
                              H: float,
                              kappa1: float,
                              vartheta: float,
                              add_constant: bool = True,
                              ) -> float:
    """
    log10 of E[sigma_t^2] / theta^2 for sigma_0 = theta and gamma > 1 from the leading term
    exp(lambda* t) / (lambda* |h_hat'|). With add_constant the constant -1/(gamma-1) is added
    when the leading term is representable in double precision.
    """
    if t <= 0.0:
        raise ValueError(f"t must be positive, got {t!r}")
    lam_star, prefactor = growth_rate(H=H, kappa1=kappa1, vartheta=vartheta)
    log_leading = lam_star * t + np.log(prefactor)
    gamma_ = stability_gamma(H=H, kappa1=kappa1, vartheta=vartheta)
    if add_constant and np.isfinite(gamma_) and log_leading < 600.0:
        return float(np.log10(np.exp(log_leading) - 1.0 / (gamma_ - 1.0)))
    return float(log_leading / np.log(10.0))


def markov_stability_parameter(nodes: np.ndarray,
                               weights: np.ndarray,
                               kappa1: float,
                               vartheta: float,
                               ) -> float:
    """
    gamma_N = vartheta^2 ||G_N||_2^2 for K^N(t) = sum_i w_i exp(-x_i t), where the Laplace
    transform of G_N is K_hat / (1 + kappa1 K_hat) and K_hat(z) = sum_i w_i / (z + x_i).
    """
    nodes, weights = np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)
    if nodes.shape != weights.shape or nodes.ndim != 1:
        raise ValueError("nodes and weights must be 1-d of equal length, "
                         f"got {nodes.shape}, {weights.shape}")
    if np.any(weights <= 0.0) or np.any(nodes < 0.0) or kappa1 <= 0.0:
        raise ValueError("require positive weights, non-negative nodes and positive kappa1")

    def integrand(w: float) -> float:
        k_hat = np.sum(weights / (1j * w + nodes))
        g_hat = k_hat / (1.0 + kappa1 * k_hat)
        return g_hat.real ** 2 + g_hat.imag ** 2

    scale = float(np.max(nodes) + kappa1 * np.sum(weights))
    return vartheta ** 2 * _half_line_integral(integrand, scale=scale) / np.pi


def critical_eta(H: float,
                 kappa1: float,
                 ) -> float:
    """
    Critical volatility-of-volatility in rough Bergomi units. Matching the short-time kernels of
    log-volatility gives vartheta = eta sqrt(2H) Gamma(alpha) / 2, hence
    eta_c = 2 vartheta_c / (sqrt(2H) Gamma(alpha)) with vartheta_c = ||G||_2^{-1}.
    """
    alpha = H + 0.5
    vartheta_c = 1.0 / np.sqrt(g2_norm(H=H, kappa1=kappa1))
    return float(2.0 * vartheta_c / (np.sqrt(2.0 * H) * gamma(alpha)))


def time_averaged_bound(T: float,
                        H: float,
                        sigma0: float,
                        theta: float,
                        kappa1: float,
                        kappa2: float,
                        ) -> Tuple[float, float]:
    """
    Upper bound of A_T = E[I_T] / T from paper eq:cesaro, and its long-run limit
    theta * max(theta, kappa1 / kappa2). Holds for every vartheta.
    """
    if T <= 0.0 or kappa2 <= 0.0 or theta <= 0.0 or sigma0 < 0.0 or kappa1 < 0.0:
        raise ValueError("require T > 0, kappa2 > 0, theta > 0, sigma0 >= 0, kappa1 >= 0")
    if not 0.0 < H <= 0.5:
        raise ValueError(f"H must lie in (0, 1/2], got {H!r}")
    alpha = H + 0.5
    c0 = kappa1 * theta
    c1 = kappa2 * theta - kappa1
    transient = sigma0 * T ** (-alpha) / gamma(2.0 - alpha)
    if c1 <= 0.0:
        bound = (c0 + transient) / kappa2
    else:
        bound = ((c1 + np.sqrt(c1 ** 2 + 4.0 * kappa2 * (c0 + transient))) / (2.0 * kappa2)) ** 2
    return float(bound), float(theta * max(theta, kappa1 / kappa2))


def markov_first_kind_resolvent(nodes: np.ndarray,
                                weights: np.ndarray,
                                ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Resolvent of the first kind of K^N(t) = sum_i w_i exp(-x_i t), distinct positive nodes:
    L(dt) = atom delta_0(dt) + (a0 + sum_j coefficients_j exp(-rates_j t)) dt, with K^N * L = 1.
    Returns (atom, a0, coefficients, rates); atom = 1 / sum_i w_i and a0 = 1 / sum_i w_i / x_i.
    """
    nodes, weights = np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)
    if nodes.shape != weights.shape or nodes.ndim != 1:
        raise ValueError("nodes and weights must be 1-d of equal length, "
                         f"got {nodes.shape}, {weights.shape}")
    if np.any(weights <= 0.0) or np.any(nodes <= 0.0) or np.unique(nodes).size != nodes.size:
        raise ValueError(f"require positive weights and distinct positive nodes, got {nodes!r}")
    q_poly = np.poly(-nodes)                                  # prod_i (z + x_i)
    p_poly = np.zeros(nodes.size)                             # sum_i w_i prod_{k != i} (z + x_k)
    for i in range(nodes.size):
        p_poly = p_poly + weights[i] * np.poly(-np.delete(nodes, i))
    roots = np.roots(p_poly)
    if np.max(np.abs(roots.imag), initial=0.0) > 1e-9 * np.max(nodes) or np.any(roots.real >= 0.0):
        raise ArithmeticError(f"roots of P must be real and negative, got {roots!r}")
    roots = np.sort(roots.real)[::-1]
    z_p_derivative = np.polyder(np.polymul([1.0, 0.0], p_poly))
    coefficients = np.polyval(q_poly, roots) / np.polyval(z_p_derivative, roots)
    a0 = float(np.polyval(q_poly, 0.0) / np.polyval(z_p_derivative, 0.0))
    if np.any(coefficients <= 0.0) or a0 <= 0.0:
        raise ArithmeticError(f"resolvent coefficients must be positive, got {coefficients!r}")
    return float(1.0 / np.sum(weights)), a0, coefficients, -roots


def markov_resolvent_mass(T: float,
                          nodes: np.ndarray,
                          weights: np.ndarray,
                          ) -> float:
    """L([0, T]) for the resolvent of the first kind of K^N."""
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T!r}")
    atom, a0, coefficients, rates = markov_first_kind_resolvent(nodes=nodes, weights=weights)
    return float(atom + a0 * T + np.sum(coefficients * (-np.expm1(-rates * T)) / rates))


def markov_time_averaged_bound(T: float,
                               nodes: np.ndarray,
                               weights: np.ndarray,
                               sigma0: float,
                               theta: float,
                               kappa1: float,
                               kappa2: float,
                               ) -> Tuple[float, float]:
    """
    Upper bound of A_T = E[I_T] / T from paper eq:cesaro for the kernel K^N, and its long-run
    limit with sigma_0 L([0,T]) / T replaced by sigma_0 / int_0^inf K^N. Holds for every vartheta.
    """
    if kappa2 <= 0.0 or theta <= 0.0 or sigma0 < 0.0 or kappa1 < 0.0:
        raise ValueError("require kappa2 > 0, theta > 0, sigma0 >= 0, kappa1 >= 0")
    _, a0, _, _ = markov_first_kind_resolvent(nodes=nodes, weights=weights)
    c0 = kappa1 * theta
    c1_plus = max(kappa2 * theta - kappa1, 0.0)
    bounds = []
    for transient in (sigma0 * markov_resolvent_mass(T=T, nodes=nodes, weights=weights) / T,
                      sigma0 * a0):
        root = (c1_plus + np.sqrt(c1_plus ** 2 + 4.0 * kappa2 * (c0 + transient))) / (2.0 * kappa2)
        bounds.append(float(root ** 2))
    return bounds[0], bounds[1]


def _producer() -> Dict:
    sources = [Path(__file__).resolve(), BASE / 'rough_logsv_stationarity_check.py']
    return {'generated_utc': datetime.now(timezone.utc).isoformat(),
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
            'scipy_version': scipy.__version__,
            'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources}}


def _load_json(name: str) -> Optional[Dict]:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


class LocalTests(Enum):
    GROWTH_RATE_TABLE = 1
    THREE_FACTOR_STABILITY = 2
    RBERGOMI_UNITS = 3
    TIME_AVERAGED_BOUND = 4
    THREE_FACTOR_TIME_AVERAGED_BOUND = 5


def run_local_test(local_test: LocalTests) -> Dict:
    """Run one case, print its table and return its records."""
    if local_test == LocalTests.GROWTH_RATE_TABLE:
        # (H, kappa1, vartheta, t, product-integration reference in the paper or its result files)
        f3 = _load_json('F3_second_moment_curves.json')
        f3_terminal = {} if f3 is None else {(c['H'], c['gamma']): c for c in f3['checks']}
        cases = [(0.5, 2.0, None, 5.0, 'F3 terminal ratio'),
                 (0.3, 2.0, None, 5.0, 'F3 terminal ratio'),
                 (0.1, 2.0, None, 5.0, 'F3 terminal ratio'),
                 (0.1, 2.0, 1.0, 1.0, 'F4 fractional benchmark 32637.6 (32,000 steps)'),
                 (0.2, 3.0, 1.85, 1.0, 'Table 2: 4.76336e7, 4.76591e7 (32,000, 64,000 steps)'),
                 (0.1, 3.0, 1.85, 1.0, 'Table 2: last entry'),
                 (0.1, 0.0, 1.85, 1.0, 'Table 2 upper panel: log10 = 7753.46')]
        records = []
        print("Growth rate above the threshold: H, kappa1, vartheta, t, gamma, lambda*, "
              "prefactor, log10 ratio (with constant), reference")
        for H, kappa1, vartheta, t, reference in cases:
            reference_value = None
            if vartheta is None:                           # Figure 3 curve with gamma = 1.2
                check = f3_terminal.get((H, 1.2))
                vartheta = np.sqrt(1.2 / g2_norm(H=H, kappa1=kappa1)) if check is None \
                    else check['vartheta']
                reference_value = None if check is None else check['terminal_ratio']
            gamma_ = stability_gamma(H=H, kappa1=kappa1, vartheta=vartheta)
            lam_star, prefactor = growth_rate(H=H, kappa1=kappa1, vartheta=vartheta)
            log10_ratio = log10_second_moment_ratio(t=t, H=H, kappa1=kappa1, vartheta=vartheta)
            record = {'H': H, 'kappa1': kappa1, 'vartheta': float(vartheta), 't': t,
                      'gamma': float(gamma_), 'lambda_star': lam_star, 'prefactor': prefactor,
                      'log10_ratio': log10_ratio,
                      'ratio': float(10.0 ** log10_ratio) if log10_ratio < 300.0 else None,
                      'reference': reference, 'reference_value': reference_value}
            if kappa1 > 0.0 and kappa1 * lam_star ** (-(H + 0.5)) <= 0.05:
                record['h_hat_series_at_root'] = h_hat_series(lam=lam_star, H=H, kappa1=kappa1,
                                                              vartheta=vartheta)
            records.append(record)
            shown = f"10^{log10_ratio:.4f}" if record['ratio'] is None else f"{record['ratio']:.8g}"
            target = '' if reference_value is None else f": {reference_value:.8g}"
            series = ''
            if 'h_hat_series_at_root' in record:
                series = f"  series h_hat(lambda*)={record['h_hat_series_at_root']:.12f}"
            print(f"  H={H} kappa1={kappa1} vartheta={vartheta:.6g} t={t} gamma={gamma_:.5g} "
                  f"lambda*={lam_star:.10g} prefactor={prefactor:.6g} ratio={shown} "
                  f"[{reference}{target}]{series}")
        return {'growth_rate_table': records}

    elif local_test == LocalTests.THREE_FACTOR_STABILITY:
        f4 = _load_json('F4_quadratic_drift.json')
        if f4 is None:
            raise FileNotFoundError("results/F4_quadratic_drift.json holds the kernel nodes")
        nodes, weights = np.array(f4['kernel']['nodes']), np.array(f4['kernel']['weights'])
        H, kappa1, vartheta = 0.1, 2.0, 1.0                 # parameters of Figure 4, right panel
        g2_markov = markov_stability_parameter(nodes=nodes, weights=weights, kappa1=kappa1,
                                               vartheta=1.0)
        gamma_markov = vartheta ** 2 * g2_markov
        gamma_fractional = stability_gamma(H=H, kappa1=kappa1, vartheta=vartheta)
        lam_star, _ = growth_rate(H=H, kappa1=kappa1, vartheta=vartheta)
        record = {'H': H, 'kappa1': kappa1, 'vartheta': vartheta,
                  'nodes': nodes.tolist(), 'weights': weights.tolist(),
                  'gamma_three_factor': float(gamma_markov),
                  'critical_vartheta_three_factor': float(1.0 / np.sqrt(g2_markov)),
                  'long_run_ratio_three_factor': float(1.0 / (1.0 - gamma_markov)),
                  'gamma_fractional': float(gamma_fractional),
                  'critical_vartheta_fractional': float(g2_norm(H=H, kappa1=kappa1) ** -0.5),
                  'growth_rate_fractional': lam_star}
        print("Three-factor kernel of Figure 4 against the fractional kernel "
              "(H=0.1, kappa1=2, vartheta=1)")
        for key, value in record.items():
            if key not in ('nodes', 'weights'):
                print(f"  {key}: {value:.6g}")
        return {'three_factor_stability': record}

    elif local_test == LocalTests.RBERGOMI_UNITS:
        records = []
        print("Critical eta in rough Bergomi units, "
              "eta_c = 2 vartheta_c / (sqrt(2H) Gamma(H+1/2))")
        for H in (0.07, 0.1, 0.2, 0.3, 0.4, 0.5):
            row = {'H': H}
            for kappa1 in (1.0, 2.0, 3.0, 5.0, 10.0):
                row[f'kappa1={kappa1:g}'] = critical_eta(H=H, kappa1=kappa1)
            records.append(row)
            print(f"  H={H}: "
                  + ", ".join(f"{value:.3f}" for key, value in row.items() if key != 'H'))
        return {'critical_eta': records}

    elif local_test == LocalTests.TIME_AVERAGED_BOUND:
        records = []
        sigma0, theta, kappa1, H = 0.4, 0.4, 2.0, 0.1      # parameters of Figure 4, right panel
        print("Bound of E[I_T]/T under quadratic drift, in units of theta^2 "
              "(sigma0=theta=0.4, kappa1=2, H=0.1)")
        for kappa2 in (0.3, 0.6, 0.9, 1.2):
            for T in (1.0, 5.0):
                bound, limit = time_averaged_bound(T=T, H=H, sigma0=sigma0, theta=theta,
                                                   kappa1=kappa1, kappa2=kappa2)
                records.append({'kappa2': kappa2, 'T': T, 'bound_over_theta2': bound / theta ** 2,
                                'limit_over_theta2': limit / theta ** 2})
                print(f"  kappa2={kappa2} T={T}: bound {bound / theta ** 2:.4g}, "
                      f"long-run limit {limit / theta ** 2:.4g}")
        return {'time_averaged_bound': records}

    elif local_test == LocalTests.THREE_FACTOR_TIME_AVERAGED_BOUND:
        f4 = _load_json('F4_quadratic_drift.json')
        if f4 is None:
            raise FileNotFoundError("results/F4_quadratic_drift.json holds the kernel nodes")
        nodes, weights = np.array(f4['kernel']['nodes']), np.array(f4['kernel']['weights'])
        sigma0, theta, kappa1 = 0.4, 0.4, 2.0               # parameters of Figure 4, right panel
        atom, a0, coefficients, rates = markov_first_kind_resolvent(nodes=nodes, weights=weights)

        def density(s: float) -> float:
            return a0 + float(np.sum(coefficients * np.exp(-rates * s)))

        def kernel(s: float) -> float:
            return float(np.sum(weights * np.exp(-nodes * s)))

        # K^N * L = 1: the atom contributes atom * K^N(t), the density a convolution integral
        identity = {f't={t:g}': atom * kernel(t)
                    + quad(lambda s: kernel(t - s) * density(s), 0.0, t,
                           epsabs=0.0, epsrel=1e-12, limit=400)[0] for t in (0.3, 2.0)}
        record = {'nodes': nodes.tolist(), 'weights': weights.tolist(),
                  'resolvent_atom': atom, 'resolvent_constant': a0,
                  'resolvent_coefficients': coefficients.tolist(),
                  'resolvent_rates': rates.tolist(),
                  'integral_of_kernel': float(np.sum(weights / nodes)),
                  'identity_check_K_conv_L': identity,
                  'resolvent_mass_over_T': {
                      f'T={T:g}': markov_resolvent_mass(T=T, nodes=nodes, weights=weights) / T
                      for T in (1.0, 2.0, 5.0)},
                  'bounds': []}
        print("First-kind resolvent of the three-factor kernel of Figure 4 and the bound of "
              "E[I_T]/T in units of theta^2 (sigma0=theta=0.4, kappa1=2)")
        print(f"  L(dt) = {atom:.6g} delta_0 + ({a0:.6g} + "
              + " + ".join(f"{c:.6g} exp(-{r:.6g} t)" for c, r in zip(coefficients, rates))
              + ") dt;  K*L at t=0.3, 2: "
              + ", ".join(f"{value:.12f}" for value in identity.values()))
        for kappa2 in (0.3, 0.6, 0.9, 1.2):
            row = {'kappa2': kappa2}
            for T in (1.0, 2.0, 5.0):
                bound, limit = markov_time_averaged_bound(T=T, nodes=nodes, weights=weights,
                                                          sigma0=sigma0, theta=theta,
                                                          kappa1=kappa1, kappa2=kappa2)
                row[f'T={T:g}'] = bound / theta ** 2
            row['limit_three_factor'] = limit / theta ** 2
            row['limit_fractional'] = max(theta, kappa1 / kappa2) / theta
            record['bounds'].append(row)
            print(f"  kappa2={kappa2}: " + ", ".join(
                f"{key} {value:.4f}" for key, value in row.items() if key != 'kappa2'))
        return {'three_factor_time_averaged_bound': record}

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    output = {'producer': _producer()}
    for local_test in LocalTests:
        output.update(run_local_test(local_test=local_test))
        print()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'growth_rate_checks.json').write_text(json.dumps(output, indent=2))
    print(f"written {RESULTS / 'growth_rate_checks.json'}")
