import numpy as np
from scipy.linalg import hankel, pinv
from scipy.sparse.linalg import eigs
from scipy.optimize import root_scalar


def beylkin_monzon_2005(fOrig,
                        delta: float,
                        b: float,
                        N: int,
                        epsilon: float,
                        m: int = None,
                        approx_roots: bool = True,
                        Nz: int = 10000,
                        weight_tol: float = None,
                        error_measure: str = 'uniform_relative',
                        mu: float = None,
                        disable_warn: bool = False,
                        maxIter: int = 20):
    """
    Implements the method of (Beylkin and Monzon, 2005).

    Parameters
    ----------
    fOrig:  Kernel, assumed completely monotonic, that is approximated as a weighted sum of exponentials
    delta:  Kernel function will be approximated on the interval [delta,b]
    b:      The approximation is performed over the interval [delta,T] where 'delta'  is a parameter
    N:
    epsilon: Sets the error tolerance of the approximation. If error_measure = 'l2_normalised' we therefore seek a minimal
            representation ensuring ||K(t)-K_hat(t)||_2/||K(t)||_2 <= epsilon where ||.||_2 denotes the euclidian norm and t
            is a vector of sampled points. If instead error_measure = 'uniform_relative' we seek a minimal representation ensuring
            max(abs(K(t)-K_hat(t))./K(t)) <= epsilon
    m:      Number of terms in the sum of exponentials approximation. Better to leave unspecified
            Default = None
    approx_roots:  If set to true then roots are found by sampling 'Nz' equidistant points in [0,1] and then running a local optimizer in
            each interval where we know a root exists. Default is true
    Nz:     Used in conjunction with parameter 'approx_roots'. Default is 10^4
    weight_tol: Let f denote K transformed to the domain [0,1] and let w_j, j=1,...,m, denote the weights of f
            (corresponding to the c_j's for K). We then remove all weights where abs(w_j) < f(0)*weight_tol.
            Although, if all weights are deemed small per this inequality we do not remove the
            largest weight. Set to 0 to disable. Defaults to epsilon/10^4 if left unspecified.
    error_measure: flag that specifies norm for error calculation. Can be ['l2_normalised', 'uniform_relative']
    mu:     Only applies to the method 'BM2005' and only if the error_measure parameter is set to 'uniform_relative'.
            Parameter specifies an initial guess on the normalised l2-error corresponding to a relative uniform error
            of epsilon. Defaults to epsilon/10 if left unspecified.
    disable_warn: For the methods that use numerical optimization we by default throw a warning if something goes
            wrong during the optimization. See also the output parameter 'exitflag'. By setting this parameter to true,
            these warnings are disabled. Default value (if left unspecified) is false
    maxIter: Use only when the error measure is set to 'uniform_relative'. Specifies the maximum number of
            iterations to find a minimal representation ensuring a uniform relative error below epsilon.
            Default is 20.

    Returns: Triple, {c_j, gamma_j, exitflag}, where
            c_j: weights,
            gamma_j: exponents
            exitflag: exitflag from numerical optimization if such is used
    -------

    """
    if epsilon is not None and m is not None:
        raise ValueError("You cannot specify both 'epsilon' and 'm' at the same time.")
    if N is None:
        raise ValueError("You must specify the 'N' parameter.")

    if epsilon is None:
        epsilon = 1e-3 if error_measure.lower() == 'l2_normalised' else 1e-2

    mu = mu if mu is not None else epsilon / 10
    weight_tol = weight_tol if weight_tol is not None else epsilon * 1e-4

    if m is not None:
        force_one_iteration = True
    elif error_measure.lower() == 'l2_normalised':
        force_one_iteration = True
        epsilon_l2 = epsilon
    elif error_measure.lower() == 'uniform_relative':
        force_one_iteration = False
        epsilon_l2 = mu
    else:
        raise ValueError("Unsupported error measure.")

    if m is not None and m > N:
        raise ValueError("We require m <= N")
    m_idx = m + 1 if m is not None else None

    z_grid = np.linspace(0, 1, Nz + 1) if approx_roots else None

    f = lambda x: fOrig(x * (b - delta) + delta)

    ii = np.arange(0, 2 * N + 1)
    x = ii / (2 * N)
    h = f(x)
    h_norm = np.sqrt(np.sum(h ** 2))

    H = hankel(h[:N + 1], h[N:])

    num_eigs_init = 20 if m is None else m + 1
    Lambda, V = eigs(H, k=min(H.shape[0], num_eigs_init))
    V = np.real(V)
    sigmas = np.real(np.sort(np.maximum(Lambda, 0))[::-1])

    if m is None:
        m_idx = np.argmax(sigmas / h_norm <= epsilon_l2)
        m = m_idx

    if m is None:
        V, Lambda = eigs(H, k=H.shape[0])
        sigmas = np.sort(np.maximum(np.diag(Lambda), 0))[::-1]
        m_idx = np.searchsorted(sigmas / h_norm, epsilon_l2)
        m = m_idx - 1 if m_idx is not None else H.shape[0] - 1

    solution_found = False
    first_iter = True
    iter_count = 1
    while not solution_found:
        u = np.flipud(V[:, m_idx])

        if not approx_roots:
            gamm_ = np.roots(u)
        else:
            y = np.polyval(u, z_grid)
            pos = y > 0
            sign_change = np.where(pos[:-1] != pos[1:])[0]

            if len(sign_change) == 0:
                gamm_ = np.array([])
            else:
                gamm_ = []
                for idx in sign_change:
                    result = root_scalar(lambda xIn: np.polyval(u, xIn), bracket=[z_grid[idx], z_grid[idx + 1]])
                    if not result.converged:
                        raise ValueError(f"Root finding failed to converge.")
                    gamm_.append(result.root)
                gamm_ = np.array(gamm_)

        gamm_ = np.unique(gamm_)

        if gamm_.size > 0:
            idxKeep = (np.imag(gamm_) == 0) & (np.real(gamm_) > 0) & (np.real(gamm_) <= 1)
            gamm_ = np.real(gamm_[idxKeep])
            if gamm_.size > 0:
                gamm_ = np.sort(gamm_)[::-1][:min(m, gamm_.size)]

        if gamm_.size == 0:
            gamm_ = np.array([1])

        A = np.vander(gamm_, len(ii), increasing=True).T
        w = pinv(A) @ h

        idxRemove = np.abs(w) < f(0) * weight_tol
        if np.any(idxRemove):
            if np.sum(idxRemove) == len(w):
                idxRemove[np.argmax(np.abs(w))] = False
            gamm_ = gamm_[~idxRemove]
            A = np.vander(gamm_, N + 1, increasing=True)
            w = pinv(A) @ h

        if force_one_iteration:
            solution_found = True
        else:
            err = np.max(np.abs(h - A @ w) / h)

            if (first_iter and err > epsilon) or (err > epsilon and err_prev > epsilon):
                if m < N:
                    m += 1
                    m_idx += 1
                    if m_idx > V.shape[1]:
                        V, Lambda = eigs(H, k=H.shape[0])
                        sigmas = np.sort(np.maximum(np.diag(Lambda), 0))[::-1]
                else:
                    if not disable_warn:
                        print("Warning: Algorithm was unable to attain the desired precision.")
                    return None, None, -1

            elif (first_iter and err <= epsilon) or (err <= epsilon and err_prev <= epsilon):
                if m == 1:
                    solution_found = True
                else:
                    m -= 1
                    m_idx -= 1

            elif err <= epsilon and err_prev > epsilon:
                solution_found = True
            elif err > epsilon and err_prev <= epsilon:
                solution_found = True
                gamm_ = gamm_prev
                w = w_prev

            first_iter = False
            gamm_prev, w_prev, err_prev = gamm_, w, err

        iter_count += 1
        if iter_count > maxIter and not solution_found:
            if err <= epsilon:
                solution_found = True
            else:
                if not disable_warn:
                    print(
                        f"Warning: Algorithm was unable to attain the desired precision within maxIter = {maxIter} iterations.")
                return None, None, -1

    t = 2 * N * np.log(gamm_)
    gamm = -t / (b - delta)
    c = w * np.exp(gamm * delta)
    return c, gamm, 0


def approximat_kernel(K,
                      n: int,
                      delta: float,
                      T: float,
                      epsilon: float = 1e-2,
                      approx_roots: bool = True,
                      Nz: int = 10 ** 4):
    """
    Uses the method of (Beylkin and Monzon, 2005) to locate a (nearly) minimal
    sum of exponentials approximation that approximately ensures an error below
    some specified level. The error is computed across equidistant points in
    in an interval of the form [delta,T]. The error can either be measured
    as the normalised l2-error or as the relative uniform error.

    Parameters
    ----------
    K:  Kernel, assumed completely monotonic, that is approximated as a weighted sum of exponentials
    n:  Sets the number of intervals to divide [delta,T] into. A number n + 1 equidistant points will then be sampled.
        Must be even number
    delta: Kernel function will be approximated on the interval [delta,T]
    T:  The approximation is performed over the interval [delta,T] where 'delta'  is a parameter
    epsilon: Sets the error tolerance of the approximation. If error_measure = 'l2_normalised' we therefore seek a minimal
        representation ensuring ||K(t)-K_hat(t)||_2/||K(t)||_2 <= epsilon where ||.||_2 denotes the euclidian norm and t
        is a vector of sampled points. If instead error_measure = 'uniform_relative' we seek a minimal representation ensuring
        max(abs(K(t)-K_hat(t))./K(t)) <= epsilon. Default is 10^(-2)
    approx_roots: If set to true then roots are found by sampling 'Nz' equidistant points in [0,1] and then running a local optimizer in
        each interval where we know a root exists. Default is true
    Nz: Used in conjunction with parameter 'approx_roots'. Default is 10^4

    Returns
    -------
    Pair of vectors, consisting of weights (c_j) and exponents (gamma_j). Vectors are sorted so that exponents are decreasing
    """
    c, gamm, exitflag = beylkin_monzon_2005(K, delta=delta, b=T, N=int(np.ceil(n / 2)), epsilon=epsilon,
                                            approx_roots=approx_roots, Nz=Nz)
    if gamm.size > 0 and c.size > 0:
        # Sort gamma in descending order and sort c accordingly
        idxSort = np.argsort(-gamm)
        gamm = gamm[idxSort]
        c = c[idxSort]

    # def K_hat(t):
    #     return np.sum(np.exp(np.outer(-gamm, t)) * c[:, np.newaxis], axis=0)

    return c, gamm
