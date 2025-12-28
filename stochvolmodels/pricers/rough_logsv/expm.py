import numpy as np
import math
import cmath
from numba import njit, prange


# ---------- small complex helpers (Numba-friendly) ----------

@njit(cache=True)
def _cabs(z):
    return math.hypot(z.real, z.imag)

@njit(cache=True)
def _csqrt(z):
    # principal square root
    r = _cabs(z)
    if r == 0.0:
        return 0.0 + 0.0j
    # sqrt in polar form
    theta = math.atan2(z.imag, z.real)
    sr = math.sqrt(r)
    ht = 0.5 * theta
    return complex(sr * math.cos(ht), sr * math.sin(ht))

@njit(cache=True)
def _ccbrt(z):
    # principal cube root
    r = _cabs(z)
    if r == 0.0:
        return 0.0 + 0.0j
    theta = math.atan2(z.imag, z.real)
    cr = r ** (1.0 / 3.0)
    tt = theta / 3.0
    return complex(cr * math.cos(tt), cr * math.sin(tt))


# ---------- Step 1: cubic roots for p(r)=0 ----------

@njit(cache=True)
def _cubic_roots_monic(a2, a1, a0):
    """
    Solve r^3 + a2*r^2 + a1*r + a0 = 0 (monic cubic).
    Returns 3 complex roots (principal-branch Cardano).
    """
    # Depress: r = y - a2/3
    one_third = 1.0 / 3.0
    shift = a2 * one_third

    p = a1 - (a2 * a2) * one_third
    q = (2.0 * a2 * a2 * a2) / 27.0 - (a2 * a1) / 3.0 + a0

    # Discriminant: Δ = (q/2)^2 + (p/3)^3
    q2 = 0.5 * q
    p3 = p * one_third
    Delta = q2 * q2 + p3 * p3 * p3  # complex-safe

    sqrtD = _csqrt(Delta)
    u = _ccbrt(-q2 + sqrtD)
    v = _ccbrt(-q2 - sqrtD)

    y1 = u + v

    # cube roots of unity:
    # y2 = -(u+v)/2 + i*sqrt(3)/2*(u-v)
    # y3 = -(u+v)/2 - i*sqrt(3)/2*(u-v)
    half = 0.5
    s3_2 = 0.5 * math.sqrt(3.0)

    uv = u - v
    y2 = -half * y1 + complex(-s3_2 * uv.imag, s3_2 * uv.real)  # i*s3_2*(u-v)
    y3 = -half * y1 - complex(-s3_2 * uv.imag, s3_2 * uv.real)

    r1 = y1 - shift
    r2 = y2 - shift
    r3 = y3 - shift
    return r1, r2, r3


# ---------- Step 2: solve Vandermonde for c0,c1,c2 ----------

@njit(cache=True)
def _solve_3x3_complex(M, b):
    """
    Solve M x = b for 3x3 complex M with partial pivoting.
    Returns x (complex length-3).
    """
    A = M.copy()
    rhs = b.copy()

    # Forward elimination
    for k in range(3):
        # pivot
        piv = k
        best = _cabs(A[k, k])
        for i in range(k + 1, 3):
            v = _cabs(A[i, k])
            if v > best:
                best = v
                piv = i
        if piv != k:
            # swap rows
            for j in range(3):
                tmp = A[k, j]
                A[k, j] = A[piv, j]
                A[piv, j] = tmp
            tmp = rhs[k]
            rhs[k] = rhs[piv]
            rhs[piv] = tmp

        akk = A[k, k]
        # If akk is (near) zero, this can blow up; caller should avoid repeated roots.
        for i in range(k + 1, 3):
            factor = A[i, k] / akk
            A[i, k] = 0.0 + 0.0j
            for j in range(k + 1, 3):
                A[i, j] = A[i, j] - factor * A[k, j]
            rhs[i] = rhs[i] - factor * rhs[k]

    # Back substitution
    x = np.empty(3, dtype=np.complex128)
    for i in range(2, -1, -1):
        s = rhs[i]
        for j in range(i + 1, 3):
            s = s - A[i, j] * x[j]
        x[i] = s / A[i, i]
    return x


# ---------- Step 3: build e^A = c0 I + c1 A + c2 A^2 ----------

@njit(cache=True)
def expA_n3_numba(kappa, lambdas, w, tol=1e-10):
    """
    Computes exp(A) for A = -kappa * 1*w^T - diag(lambdas), where 1=(1,1,1)^T.
    Steps:
      1) build cubic p(r)=det(rI-A) and find roots r1,r2,r3
      2) solve Vandermonde for c0,c1,c2 so exp(A)=c0 I + c1 A + c2 A^2
      3) evaluate polynomial in A

    Returns: 3x3 float64 matrix.
    """
    l1, l2, l3 = lambdas[0], lambdas[1], lambdas[2]
    w1, w2, w3 = w[0], w[1], w[2]

    # --- Build monic cubic coefficients for p(r)=0:
    # p(r) = (r+λ1)(r+λ2)(r+λ3) + κ[ w1(r+λ2)(r+λ3) + w2(r+λ1)(r+λ3) + w3(r+λ1)(r+λ2) ]
    s1 = l1 + l2 + l3
    s2 = l1*l2 + l1*l3 + l2*l3
    s3 = l1*l2*l3

    W = w1 + w2 + w3
    B = w1*(l2+l3) + w2*(l1+l3) + w3*(l1+l2)
    C = w1*(l2*l3) + w2*(l1*l3) + w3*(l1*l2)

    a2 = s1 + kappa * W
    a1 = s2 + kappa * B
    a0 = s3 + kappa * C

    r1, r2, r3 = _cubic_roots_monic(a2 + 0.0j, a1 + 0.0j, a0 + 0.0j)

    # Guard against repeated/near-repeated roots (Vandermonde gets unstable)
    if _cabs(r1 - r2) < tol or _cabs(r1 - r3) < tol or _cabs(r2 - r3) < tol:
        # Simple fallback: scaling-and-squaring with truncated Taylor (kept Numba-friendly)
        # This is not "steps 1-2-3" anymore, but prevents crashes when roots collide.
        A = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                A[i, j] = -kappa * w[j]
            A[i, i] -= lambdas[i]

        # scale
        # crude bound on ||A||_1
        colsum_max = 0.0
        for j in range(3):
            s = 0.0
            for i in range(3):
                s += abs(A[i, j])
            if s > colsum_max:
                colsum_max = s
        s_pow = 0
        if colsum_max > 0.5:
            s_pow = int(math.ceil(math.log(colsum_max / 0.5) / math.log(2.0)))

        As = A / (2.0 ** s_pow)

        # exp(As) via Taylor up to N terms
        E = np.eye(3, dtype=np.float64)
        term = np.eye(3, dtype=np.float64)
        N = 30
        for n in range(1, N + 1):
            # term = term @ As / n
            newterm = np.zeros((3, 3), dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    acc = 0.0
                    for k in range(3):
                        acc += term[i, k] * As[k, j]
                    newterm[i, j] = acc / n
            term = newterm
            E += term

        # square back
        for _ in range(s_pow):
            EE = np.zeros((3, 3), dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    acc = 0.0
                    for k in range(3):
                        acc += E[i, k] * E[k, j]
                    EE[i, j] = acc
            E = EE
        return E

    # Vandermonde: [1 r r^2][c0,c1,c2]^T = exp(r)
    V = np.empty((3, 3), dtype=np.complex128)
    b = np.empty(3, dtype=np.complex128)

    roots = (r1, r2, r3)
    for i in range(3):
        r = roots[i]
        V[i, 0] = 1.0 + 0.0j
        V[i, 1] = r
        V[i, 2] = r * r
        b[i] = np.exp(r)

    c = _solve_3x3_complex(V, b)
    c0, c1, c2 = c[0], c[1], c[2]

    # Build A (real)
    A = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            A[i, j] = -kappa * w[j]
        A[i, i] -= lambdas[i]

    # A^2
    A2 = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += A[i, k] * A[k, j]
            A2[i, j] = acc

    # E = c0 I + c1 A + c2 A^2  (complex -> take real part)
    E = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            val = c1 * (A[i, j] + 0.0j) + c2 * (A2[i, j] + 0.0j)
            if i == j:
                val += c0
            E[i, j] = val.real  # imaginary parts should be ~0 for real A
    return E

@njit(cache=True)
def expA_n1_numba (kappa, lambdas, w):
    """
    n=1: A = (lambda1 + kappa*w1). Returns scalar
    lambdas: shape (1,)
    w: shape (1,)
    """
    a = -(lambdas[0] + kappa * w[0])
    out = math.exp(a)
    return out

@njit(cache=True)
def _sinhc(z, tol=1e-12):
    """sinh(z)/z with small-z series for stability (Numba-friendly)."""
    if abs(z) < tol:
        z2 = z * z
        return 1.0 + 22 / 6.0 + (z2 *z2) / 120.0
    return cmath.sinh(z) / z


@njit(cache=True)
def expA_n2_numba(kappa, lambdas, w):
    """n=2: A = -kappa*1*w^T - diag(lambdas). Returns 2x2 exp(A).
    lambdas: shape (2,)
    w: shape (2,)
    """
    l1, l2 = lambdas[0], lambdas [1]
    w1, w2 = w[0], w[1]

    # A entries
    a11 = -l1 - kappa * w1
    a12 = -kappa * w2
    a21 = -kappa * w1
    a22 = -l2 - kappa * w2

    tr = a11 + a22
    det = a11 * a22 - a12 * a21

    half_tr = 0.5 * tr
    Delta = cmath.sqrt((half_tr * half_tr) - det + 0.0j)

    ehalf = math.exp(half_tr) # half_tr is real
    c = cmath.cosh(Delta)
    s_over = _sinhc(Delta)

    # B = A  - (tr/2) I
    b11 = a11 - half_tr
    b22 = a22 - half_tr
    b12 = a12
    b21 = a21

    # exp(A) = e^{tr/2} [ c I + (sinh(Delta)/Delta) * B ]
    E11 = ehalf * (c + s_over * (b11 + 0.0j))
    E22 = ehalf * (c + s_over * (b22 + 0.0j))
    E12 = ehalf * (s_over * (b12 + 0.0j))
    E21 = ehalf * (s_over * (b21 + 0.0j))

    out = np.empty((2, 2), dtype = np.float64)
    out[0, 0] = E11.real
    out[0, 1] = E12.real
    out[1, 0] = E21.real
    out[1, 1] = E22.real
    return out



@njit(parallel=True, cache=True)
def batch_expA_n3(kappas: np.ndarray, lambdas_all: np.ndarray, w_all: np.ndarray):
    """
    kappas: shape (N,)
    lambdas_all: shape (N,3)
    w_all: shape (N,3)
    returns: shape (N,3,3)
    """
    N = kappas.shape[-1]
    out = np.empty((N,3,3), dtype=np.float64)
    for i in prange(N):
        out[i] = expA_n3_numba(kappas[i], lambdas_all[i], w_all[i])
    return out


@njit(parallel=True, cache=True)
def batch_expA_n2(kappas: np.ndarray, lambdas_all: np.ndarray, w_all: np.ndarray):
    """
    kappas: shape (N,)
    lambdas_all: shape (N,2)
    w_all: shape (N,2)
    returns: shape (N,2,2)
    """
    N = kappas.shape[-1]
    out = np.empty((N,2,2), dtype=np.float64)
    for i in prange(N):
        out[i] = expA_n2_numba(kappas[i], lambdas_all[i], w_all[i])
    return out


@njit(parallel=True, cache=True)
def batch_expA_n1(kappas: np.ndarray, lambdas_all: np.ndarray, w_all: np.ndarray):
    """
    kappas: shape (N,)
    lambdas_all: shape (N,1)
    w_all: shape (N,1)
    returns: shape (N,1,1)
    """
    N = kappas.shape[-1]
    out = np.empty((N,1,1), dtype=np.float64)
    for i in prange(N):
        out[i,0,0] = expA_n1_numba(kappas[i], lambdas_all[i], w_all[i])
    return out



@njit(parallel=True, cache=True)
def batch_expA(kappas: np.ndarray, lambdas_all: np.ndarray, w_all: np.ndarray):
    """
    kappas: shape (N,)
    lambdas_all: shape (N,d)
    w_all: shape (N,d)
    returns: shape (N,d,d)
    """
    d = lambdas_all.shape[-1]
    if d==1:
        return batch_expA_n1(kappas, lambdas_all, w_all)
    elif d==2:
        return batch_expA_n2(kappas, lambdas_all, w_all)
    elif d==3:
        return batch_expA_n3(kappas, lambdas_all, w_all)
    else:
        raise ValueError("Only d=1,2,3 are supported")


#########################################################################
@njit(cache=True)
def invA_rank1_numba_general(kappa, lambdas, w, tol=1e-14):
    """Analytic inverse for A = -kappa*1*w^T - diag(lambdas), any n>=1.

    Inputs:
    kappa: float
    lambdas: (n,) float64, diagonal entries of L (must be nonzero)
    w: (n.) float64
    tol: singularity tolerance for denom

    Returns:
    Ainv: (n,n) float64. If singular/ill-conditioned, returns all-Nan matrix.
    """
    n = lambdas.shape[0]
    Ainv = np.empty((n, n), dtype=np.float64)

    # d_i = 1/lambda_i, z_i = w_i/lambda_i, s = sum z_i
    d = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)

    s = 0.0
    for i in range(n):
        lam = lambdas[i]
        if lam == 0.0:
            # singular because L^{-1} doesn't exist
            for r in range(n):
                for c in range(n):
                    Ainv[r, c] = np.nan
            return Ainv
        di = 1.0 / lam
        d[i] = di
        zi = w[i] * di
        z[i] = zi
        s += zi

    denom = 1.0 + kappa * s
    if abs(denom) < tol:
        for r in range(n):
            for c in range(n):
                Ainv[r, c] = np.nan
        return Ainv

    alpha = kappa / denom
    for i in range(n):
        di = d[i]
        for j in range(n):
            val = alpha * di * z[j]
            if i == j:
                val -= di
            Ainv[i, j] = val
    return Ainv


@njit(parallel=True, cache=True)
def batch_invA(kappas: np.ndarray, lambdas_all: np.ndarray, w_all: np.ndarray):
    """
    kappas: shape (N,)
    lambdas_all: shape (N,3)
    w_all: shape (N,3)
    returns: shape (N,3,3)
    """
    N, d = w_all.shape
    out = np.empty((N,d,d), dtype=np.float64)
    for i in prange(N):
        out[i] = invA_rank1_numba_general(kappas[i], lambdas_all[i], w_all[i])
    return out
