import numpy as np
import scipy.integrate as integ
from scipy.optimize import minimize, lsq_linear
from scipy.special import gamma, gammainc

#import orthopy
#import quadpy


def sort(a, b):
    """
    Sorts two numpy arrays jointly according to the ordering of the first.
    :param a: First numpy array
    :param b: Second numpy array
    :return: Sorted numpy arrays
    """
    perm = np.argsort(a)
    return a[perm], b[perm]


def rel_err(x, x_approx):
    """
    Computes the absolute relative error of x_approx compared to x.
    :param x: The true value
    :param x_approx: The approximated value
    :return: The absolute relative error
    """
    return np.abs((x - x_approx) / x)


def single_param_search(f, rel_tol=1e-03, n=100, factor=2):
    """
    Finds the optimal parameter n for approximating f.
    :param f: A function having two inputs n and reusable. The input n is some discretization parameter. For example,
        f might be the solution of an ODE and n is the number of time steps. The input reusable is used to supply
        previously computed information, so that it does not have to be computed again. For example, if f is the
        trapezoidal integral of some function, this may be an array of previously computed function values that can
        be reused without needing to recompute them. If there is nothing sensible that could be reused, just leave it as
        a dead parameter. The function gives two outputs a and b. The output a is the result (e.g. the final point of
        the ODE solution), and the output b is the reusable information. In the next call of f, b will be given as the
        parameter reusable.
    :param rel_tol: Relative error tolerance of the result
    :param n: Initial parameter n
    :param factor: Factor by which we should multiply n if we need higher accuracy.
    :return: The approximated result, the final parameter n that was used, and the reusable information
    """
    int_calc = isinstance(n, int)
    approx_res, reusable = f(n=n // factor if int_calc else n / factor, reusable=None)
    current_res, reusable = f(n=n, reusable=reusable)
    while rel_err(current_res, approx_res) > rel_tol:
        n = int(factor * n) if int_calc else factor * n
        approx_res = current_res
        current_res, reusable = f(n=n, reusable=reusable)
    return current_res, n, reusable


def exp_underflow(x):
    """
    Computes exp(-x) while avoiding underflow errors.
    :param x: Float of numpy array
    :return: exp(-x)
    """
    if isinstance(x, np.ndarray):
        if x.dtype == int:
            x = x.astype(np.float)
        eps = np.finfo(x.dtype).tiny
    else:
        if isinstance(x, int):
            x = float(x)
        eps = np.finfo(x.__class__).tiny
    log_eps = -np.log(eps) / 2
    result = np.exp(-np.fmin(x, log_eps))
    result = np.where(x > log_eps, 0, result)
    return result


def fractional_kernel(H, t):
    """
    The fractional kernel.
    :param H: Hurst parameter
    :param t: Time, may also be a numpy array
    :return: The value of the fractional kernel at t
    """
    return t ** (H - 0.5) / gamma(H + 0.5)


def kernel_norm(H, T, p=2.):
    """
    Returns the L^p-norm of the fractional kernel.
    :param H: Hurst parameter
    :param T: Final time
    :param p: The order of the norm
    :return: The L^p-norm (root has been taken) of the fractional kernel
    """
    return T ** (H - 0.5 + 1 / p) / (gamma(0.5 + H) * (1 + p * H - p / 2) ** (1 / p))


def c_H(H):
    """
    Returns the constant c_H.
    :param H: Hurst parameter
    :return: c_H
    """
    return 1. / (gamma(0.5 + H) * gamma(0.5 - H))


def fractional_kernel_laplace(H, t, nodes):
    """
    The Laplace transform of the fractional kernel.
    :param H: Hurst parameter
    :param t: Time, may also be a numpy array
    :param nodes: Laplace transform argument, may also be a numpy array
    :return: The Laplace transform. May be a number, a one-dimensional or a two-dimensional numpy array depending on
        the shape of t and nodes. If both t and nodes are a numpy array, the tensor product that we take is
        nodes x time
    """
    if isinstance(t, np.ndarray) and isinstance(nodes, np.ndarray):
        return c_H(H) * exp_underflow(np.tensordot(nodes, t, axes=0))
    return c_H(H) * exp_underflow(nodes * t)


def fractional_kernel_approximation(H, t, nodes, weights):
    """
    Returns the Markovian approximation of the fractional kernel.
    :param H: Hurst parameter
    :param t: Time points
    :param nodes: Nodes of the quadrature rule
    :param weights: Weights of the quadrature rule
    :return: The approximated kernel using nodes and weights at times t (a numpy array)
    """
    return 1 / c_H(H) * np.tensordot(fractional_kernel_laplace(H, t, nodes), weights, axes=([0, 0]))


def AK_improved_rule(H, N, K=None, T=1.):
    """
    The quadrature rule from Alfonsi and Kebaier in Table 6, left column.
    :param H: Hurst parameter
    :param N: Total number of nodes
    :param K: Cutoff point where the regime changes
    :param T: Final time
    :return: The quadrature rule in the form nodes, weights
    """
    if N == 1:
        return np.array([0.]), np.array([0.])

    N = N // 2

    if K is None:
        K = N ** 0.8

    def AK_initial_guess(A_):
        partition = np.empty(2 * N + 1)
        partition[:N + 1] = np.linspace(0, K, N + 1)
        partition[N + 1:] = K * A_ ** np.arange(1, N + 1)
        a = partition ** (1.5 - H)
        b = partition ** (0.5 - H)
        nodes_ = (0.5 - H) / (1.5 - H) * (a[1:] - a[:-1]) / (b[1:] - b[:-1])
        weights_ = c_H(H) / (0.5 - H) * (b[1:] - b[:-1])
        return nodes_, weights_

    def error_func(A_):
        nodes_, weights_ = AK_initial_guess(A_[0])
        return error_l2(H, nodes_, weights_, T)

    res = minimize(fun=lambda A_: error_func(A_), x0=np.array([1.2]), bounds=((0, None),))
    A = res.x
    nodes, weights = AK_initial_guess(A[0])

    res = minimize(fun=lambda x: error_l2(H, nodes, x * weights, T), x0=np.array([1]), bounds=((0, None),))
    return nodes, res.x * weights


def AbiJaberElEuch_quadrature_rule(H, N, T):
    """
    Computes the quadrature as suggested in "Multi-factor approximation of rough volatility models" by Abi Jaber and
    El Euch.
    :param H: Hurst parameter
    :param N: Number of quadrature nodes
    :param T: Maturity / Final time
    :return: The nodes and weights, two numpy arrays
    """
    pi_n = N ** (-0.2) / T * (np.sqrt(10) * (1 - 2 * H) / (5 - 2 * H)) ** 0.4
    eta = pi_n * np.arange(N + 1)
    c_vec = (eta[1:] ** (0.5 - H) - eta[:-1] ** (0.5 - H)) / (gamma(H + 0.5) * gamma(1.5 - H))
    gamma_vec = (eta[1:] ** (1.5 - H) - eta[:-1] ** (1.5 - H)) / ((1.5 - H) * gamma(H + 0.5) + gamma(0.5 - H)) / c_vec
    return gamma_vec, c_vec


def Gaussian_parameters(H, N, T, mode):
    """
    Returns the parameters of the Gaussian quadrature rule.
    :param H: Hurst parameter
    :param N: Total number of nodes
    :param T: Final time
    :param mode: The kind of theorem or observation that should be used
    :return: The partition of the middle part, and the quadrature level m
    """
    if ' geometric ' in mode or mode == "OLD" or mode == "GG":
        if mode == "old geometric theorem l2":
            N = N - 1
            A = np.sqrt(1 / H + 1 / (1.5 - H))
            beta = 0.4275
            alpha = 1.06418
            gamma_ = np.exp(alpha * beta)
            exponent = 1 / (3 * gamma_ / (8 * (gamma_ - 1)) + 6 * H - 4 * H * H)
            temp_1 = ((9 - 6 * H) / (2 * H)) ** (gamma_ / (8 * (gamma_ - 1)))
            temp_2 = 5 * np.pi ** 3 * gamma_ * (gamma_ - 1) * A ** (2 - 2 * H) * float(N) ** (1 - H) / (
                        beta ** (2 - 2 * H))
            base_0 = temp_1 * (temp_2 * (3 - 2 * H) / (768 * H)) ** (2 * H)
            a = 1 / T * base_0 ** exponent * np.exp(-alpha / ((1.5 - H) * A) * np.sqrt(N))
            base_n = temp_1 * (temp_2 / 1152) ** (2 * H - 3)
            b = 1 / T * base_n ** exponent * np.exp(alpha / (H * A) * np.sqrt(N))
            m = int(np.fmax(np.round(beta / A * np.sqrt(N)), 1))
            n = int(np.round(N / m))
        elif mode == "old geometric observation l2" or mode == "OLD":
            N = N - 1
            A = np.sqrt(1 / H + 1 / (1.5 - H))
            beta = 0.9
            alpha = 1.8
            a = 0.65 * 1 / T * np.exp(3.1 * H) * np.exp(-alpha / ((1.5 - H) * A) * np.sqrt(N))
            b = 1 / T * np.exp(3 * H ** (-0.4)) * np.exp(alpha / (H * A) * np.sqrt(N))
            m = int(np.fmax(np.round(beta / A * np.sqrt(N)), 1))
            n = int(np.round(N / m))
        elif mode == "new geometric theorem l1" or mode == "GG":
            beta = 1
            alpha = np.log(3 + 2 * np.sqrt(2))
            a = 4 / T
            b = 1 / 2 / T * np.exp(alpha / np.sqrt(H + 0.5) * np.sqrt(N))
            m = int(np.fmax(np.round(beta * np.sqrt((H + 0.5) * N)), 1))
            n = int(np.round(N / m)) - 1
        else:
            raise NotImplementedError(f'The mode {mode} has not been implemented')

        partition = np.exp(np.log(a) + np.log(b / a) * np.linspace(0, 1, n + 1))
    else:
        if mode == 'non-geometric l1' or mode == "NGG":
            beta = 0.92993273
            a = 3 / T
            m = int(np.fmax(np.round(beta * np.sqrt((H + 0.5) * N)), 1))
            c = 3.60585021
        else:
            raise NotImplementedError(f'The mode {mode} has not been implemented')

        kappa = 1 / (2 * beta ** 2)
        n = int(np.round(N / m)) - 1
        partition = np.empty(n + 1)
        partition[0] = a
        for i in range(n):
            partition[i + 1] = partition[i] \
                * ((c + partition[i] ** (kappa / (n + 1))) / (c - partition[i] ** (kappa / (n + 1)))) ** 2
    return partition, m


def Gaussian_interval(H, m, a, b, fractional_weight=True):
    """
    Returns the nodes and weights of the Gauss quadrature rule of level m on [a, b].
    :param H: Hurst parameter
    :param m: Level of the Gaussian quadrature
    :param a: Left end of interval
    :param b: Right end of interval
    :param fractional_weight: If True, computes the Gaussian quadrature rule with respect to the fractional weight. If
        False, computes the Gaussian quadrature with respect to the weight function w(x) = c_H
    :return: The nodes and weights
    """
    if fractional_weight:
        k = np.arange(2 * m) + 0.5 - H
    else:
        k = np.arange(1, 2 * m + 1)
    alpha, beta, int_1 = orthopy.tools.chebyshev(moments=c_H(H) / k * (b ** k - a ** k))
    return quadpy.tools.scheme_from_rc(alpha, beta, int_1)


def Gaussian_on_partition(H, m, partition, fractional_weight=True):
    """
    Returns the nodes and weights of the Gaussian quadrature rule of level m on a partition.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param partition: The partition, where the Gaussian quadrature rule is applied on each interval
    :param fractional_weight: If True, computes the Gaussian quadrature rule with respect to the fractional weight
        function. If False, computes the Gaussian quadrature with respect to the weight function w(x) = c_H, and
        then multiplies the weights by nodes ** (-H - 1/2)
    :return: All the nodes and weights
    """
    nodes = np.empty(m * (len(partition) - 1))
    weights = np.empty(m * (len(partition) - 1))
    for i in range(len(partition) - 1):
        new_nodes, new_weights = Gaussian_interval(H=H, m=m, a=partition[i], b=partition[i + 1],
                                                   fractional_weight=fractional_weight)
        nodes[m * i:m * (i + 1)] = new_nodes
        weights[m * i:m * (i + 1)] = new_weights
    if not fractional_weight:
        weights = weights * nodes ** (-H - 0.5)
    return nodes, weights


def Gaussian_optimal_zero_weight(H, T, nodes, weights):
    """
    Computes the optimal weight in the L^2-approximation of an additional node at 0 given that we are already using the
    specified nodes and weights.
    :param H: Hurst parameter
    :param T: Final time
    :param nodes: The nodes of the Markovian approximation, a numpy array
    :param weights: The weights of the Markovian approximation, a numpy array
    :return: The optimal weight in the L^2-sense of an additional node at 0
    """
    if len(nodes) == 0:
        return T ** (H - 0.5) / gamma(H + 1.5)
    return (T ** (H + 0.5) / gamma(H + 1.5) - np.sum(weights / nodes * (1 - exp_underflow(nodes * T)))) / T


def Gaussian_rule(H, N, T, mode):
    """
    Returns the nodes and weights of the Gaussian rule with roughly N nodes.
    :param H: Hurst parameter
    :param N: Number of nodes
    :param T: Final time
    :param mode: The Gaussian parameters that should be used
    :return: The nodes and weights, ordered by the size of the nodes
    """
    if isinstance(T, np.ndarray):
        T = T[-1]
    partition, m = Gaussian_parameters(H, N, T, mode)

    if mode == 'old geometric theorem l2' or mode == 'old geometric observation l2':
        if N == 1:
            w_0 = Gaussian_optimal_zero_weight(H=H, T=T, nodes=np.array([]), weights=np.array([]))
            nodes, weights = np.array([0.]), np.array([w_0])
        else:
            nodes, weights = np.zeros(m * (len(partition) - 1) + 1), np.empty(m * (len(partition) - 1) + 1)
            nodes[1:], weights[1:] = Gaussian_on_partition(H=H, m=m, partition=partition, fractional_weight=True)
            weights[0] = Gaussian_optimal_zero_weight(H=H, T=T, nodes=nodes[1:], weights=weights[1:])
    else:
        nodes, weights = np.empty(m * len(partition)), np.empty(m * len(partition))
        nodes[:m], weights[:m] = Gaussian_interval(H=H, m=m, a=0, b=partition[0], fractional_weight=True)
        if len(partition) > 1:
            nodes[m:], weights[m:] = Gaussian_on_partition(H=H, m=m, partition=partition,
                                                           fractional_weight='old' in mode)
    return nodes, weights


def error_l2(H, nodes, weights, T, output='error'):
    """
    Computes an error estimate of the squared L^2-norm of the difference between the rough kernel and its approximation
    on [0, T].
    :param H: Hurst parameter
    :param nodes: The nodes of the approximation. Assumed that they are all non-zero
    :param weights: The weights of the approximation
    :param T: Final time, may also be a numpy array
    :param output: If error, returns the error. If gradient, returns the error and the gradient of the error
    :return: An error estimate
    """
    nodes = np.fmin(np.fmax(nodes, 1e-08), 1e+150)
    weights = np.fmin(weights, 1e+75)
    weight_matrix = np.outer(weights, weights)
    summand = T ** (2 * H) / (2 * H * gamma(H + 0.5) ** 2)
    node_matrix = nodes[:, None] + nodes[None, :]
    if isinstance(T, np.ndarray):
        gamma_ints = gammainc(H + 0.5, np.outer(T, nodes))
        nmT = np.einsum('i,jk->ijk', T, node_matrix)
        exp_node_matrix = exp_underflow(nmT)
        sum_1 = np.sum((weight_matrix / node_matrix)[None, :, :] * (1 - exp_node_matrix), axis=(1, 2))
        sum_2 = 2 * np.sum((weights / nodes ** (H + 0.5))[None, :] * gamma_ints, axis=1)
    else:
        gamma_ints = gammainc(H + 0.5, nodes * T)
        nmT = node_matrix * T
        exp_node_matrix = exp_underflow(nmT)
        sum_1 = np.sum(weight_matrix / node_matrix * (1 - exp_node_matrix))
        sum_2 = 2 * np.sum(weights / nodes ** (H + 0.5) * gamma_ints)
    err = summand + sum_1 - sum_2
    if output == 'error' or output == 'err':
        return err

    N = len(nodes)
    if isinstance(T, np.ndarray):
        grad = np.empty((len(T), 2 * N))
        exp_node_vec = exp_underflow(np.outer(T, nodes)) / nodes[None, :]
        first_summands = (weight_matrix / (node_matrix * node_matrix))[None, :] * (1 - (1 + nmT) * exp_node_matrix)
        second_summands = weights[None, :] * ((T ** (H + 1 / 2) / gamma(H + 1 / 2))[:, None] * exp_node_vec - (
                ((H + 1 / 2) * nodes ** (-H - 3 / 2))[None, :] * gamma_ints))
        grad[:, :N] = -2 * np.sum(first_summands, axis=2) - 2 * second_summands
        third_summands = np.einsum('ijk,k->ij', ((1 - exp_node_matrix) / node_matrix[None, :, :]), weights)
        forth_summands = (nodes ** (-(H + 1 / 2)))[None, :] * gamma_ints
        grad[:, N:] = 2 * third_summands - 2 * forth_summands
    else:
        grad = np.empty(2 * N)
        exp_node_vec = np.zeros(N)
        indices = nodes * T < 300
        exp_node_vec[indices] = np.exp(- T * nodes[indices]) / nodes[indices]
        first_summands = weight_matrix / (node_matrix * node_matrix) * (1 - (1 + nmT) * exp_node_matrix)
        second_summands = weights * (T ** (H + 1 / 2) / gamma(H + 1 / 2) * exp_node_vec - (H + 1 / 2) * nodes ** (
                -H - 3 / 2) * gamma_ints)
        grad[:N] = -2 * np.sum(first_summands, axis=1) - 2 * second_summands
        third_summands = ((1 - exp_node_matrix) / node_matrix) @ weights
        forth_summands = nodes ** (-(H + 1 / 2)) * gamma_ints
        grad[N:] = 2 * third_summands - 2 * forth_summands
    return err, grad


def error_l1(H, nodes, weights, T, method='intersections', tol=1e-08):
    """
    Computes an error estimate of the L^1-norm of the difference between the rough kernel and its approximation
    on [0, T].
    :param H: Hurst parameter
    :param nodes: The nodes of the approximation. Assumed that they are all non-zero
    :param weights: The weights of the approximation
    :param T: Final time, may also be a numpy array
    :param method: Method for computing the error
    :param tol: Relative error tolerance with which the error should be computed (relative error of the error)
    :return: An error estimate
    """
    if method == 'trapezoidal':
        def error_(n, reusable):
            t = np.linspace(0, T, n + 1)[1:]
            if reusable is None:
                reusable = np.abs(fractional_kernel(H, t) - fractional_kernel_approximation(H, t, nodes, weights))
            else:
                error_t_1 = np.empty(n)
                error_t_1[1::2] = reusable
                error_t_1[::2] = np.abs(fractional_kernel(H, t[1::2])
                                        - fractional_kernel_approximation(H, t[1::2], nodes, weights))
                reusable = error_t_1
            total_error = np.trapz(reusable, dx=T / n)
            return total_error + np.abs(fractional_kernel(H, T / (2 * n))
                                        - fractional_kernel_approximation(H, T / (2 * n), nodes, weights)) * T / n, \
                reusable
        return single_param_search(f=error_, rel_tol=tol, n=100, factor=2)[0:2]
    elif method == 'exact - trapezoidal':
        gamma_ = gamma(H + 0.5)

        def find_first_intersection():
            current_error_ = 10.
            current_t = 0.
            current_kernel_approximation = fractional_kernel_approximation(H=H, t=current_t, nodes=nodes,
                                                                           weights=weights)
            while current_error_ > tol and current_t < T:
                current_t = (current_kernel_approximation * gamma_) ** (1 / (H - 0.5))
                current_kernel_approximation = fractional_kernel_approximation(H=H, t=current_t, nodes=nodes,
                                                                               weights=weights)
                current_kernel = fractional_kernel(H=H, t=current_t)
                current_error_ = rel_err(current_kernel, current_kernel_approximation)
            return np.fmin(current_t, T)

        def error_(n, reusable):
            t = np.linspace(t_0, T, n + 1)
            if reusable is None:
                reusable = np.abs(fractional_kernel(H, t) - fractional_kernel_approximation(H, t, nodes, weights))
            else:
                error_t_1 = np.empty(n + 1)
                error_t_1[::2] = reusable
                error_t_1[1::2] = np.abs(fractional_kernel(H, t[1::2])
                                         - fractional_kernel_approximation(H, t[1::2], nodes, weights))
                reusable = error_t_1
            total_error = np.trapz(reusable, dx=(T - t_0) / n)
            return error_to_t_0 + total_error, reusable

        t_0 = find_first_intersection()
        error_to_t_0 = t_0 ** (H + 0.5) / (gamma_ * (H + 0.5)) \
            - np.sum(weights / nodes * (1 - exp_underflow(nodes * t_0)))
        if t_0 == T:
            return error_to_t_0
        return single_param_search(f=error_, rel_tol=tol, n=100, factor=2)[0:2]
    elif method == 'upper bound':
        gamma_ = gamma(H + 0.5)
        nm = nodes[:, None] + nodes[None, :]
        wm = weights[:, None] * weights[None, :]
        err = - 2 * gamma_ * gamma(1.5 + H) * np.sum(weights / nodes ** (1.5 + H) * gammainc(1.5 + H, nodes * T))\
            + gamma_ ** 2 * gamma(2) * np.sum(wm / nm ** 2 * gammainc(2, nm * T))
        return err, 0
    elif method == 'intersections':
        gamma_1 = gamma(H + 0.5)

        def step(t_, ker_, ker_approx_):
            nonlocal n_steps
            n_steps = n_steps + 1
            ker_larger = ker_ > ker_approx_
            rel_err_ = rel_err(ker_, ker_approx_)
            if rel_err_ > tol:
                d_ker = (H - 0.5) / gamma_1 * t_ ** (H - 1.5)
                d_ker_approx = - np.sum(weights * nodes * exp_underflow(nodes * t_))
                if ker_larger:
                    dd_ker_approx = np.sum(weights * nodes ** 2 * exp_underflow(nodes * t_))
                    return t_ + (d_ker - d_ker_approx
                                 + np.sqrt((d_ker - d_ker_approx) ** 2 - 2 * dd_ker_approx * (ker_approx_ - ker_))) \
                        / dd_ker_approx
                else:
                    dd_ker = (H - 0.5) * (H - 1.5) / gamma_1 * t_ ** (H - 2.5)
                    return t_ + (d_ker_approx - d_ker + np.sqrt((d_ker - d_ker_approx) ** 2
                                                                - 2 * dd_ker * (ker_ - ker_approx_))) / dd_ker
            else:
                t_1 = t_ + tol * ker_ / np.sum(weights * nodes * exp_underflow(nodes * t_))
                t_2 = t_ + tol * ker_ ** 2 / (tol * ker_ + ker_approx_) * gamma_1 / (0.5 - H) * t_ ** (1.5 - H)
                return np.fmin(t_1, t_2)

        def find_next_intersection(t_):
            ker_ = fractional_kernel(H=H, t=t_)
            ker_approx_ = fractional_kernel_approximation(H=H, t=t_, nodes=nodes, weights=weights)
            ker_larger = ker_ > ker_approx_
            t_old = t_
            while (ker_ > ker_approx_) == ker_larger and t_ < T:
                t_old = t_
                t_ = step(t_=t_, ker_=ker_, ker_approx_=ker_approx_)
                ker_ = fractional_kernel(H=H, t=t_)
                ker_approx_ = fractional_kernel_approximation(H=H, t=t_, nodes=nodes, weights=weights)
            return t_old, np.fmin(t_, T)

        n_steps = 0
        err = 0
        ker_approx = fractional_kernel_approximation(H=H, t=0, nodes=nodes, weights=weights)
        last_t = 0
        while last_t < T:
            t_left, t_right = find_next_intersection(t_=(ker_approx * gamma_1) ** (1 / (H - 0.5))
                                                     if last_t == 0 else last_t)
            this_t = (t_left + t_right) / 2 if t_right < T else T
            err = err + np.abs((this_t ** (H + 0.5) - last_t ** (H + 0.5)) / (gamma_1 * (H + 0.5))
                               - np.sum(weights / nodes * exp_underflow(last_t * nodes)
                                        * (1 - exp_underflow((this_t - last_t) * nodes))))
            last_t = this_t
        return err, n_steps
    elif method == 'reparametrized trapezoidal':

        def error_(n, reusable):
            t = np.linspace(0, T ** (0.5 + H), n + 1)[1:] ** (1 / (0.5 + H))
            if reusable is None:
                reusable = np.empty(n + 1)
                reusable[1:] = np.abs(1 / gamma(H + 1.5)
                                      - t ** (0.5 - H) * fractional_kernel_approximation(H, t, nodes,
                                                                                         weights) / (0.5 + H))
                reusable[0] = 1 / gamma(H * 1.5)
            else:
                error_t_1 = np.empty(n + 1)
                error_t_1[::2] = reusable
                error_t_1[1::2] = np.abs(1 / gamma(H + 1.5)
                                         - t[1::2] ** (0.5 - H) * fractional_kernel_approximation(H, t[1::2], nodes,
                                                                                                  weights) / (0.5 + H))
                reusable = error_t_1
            total_error = np.trapz(reusable, dx=T ** (0.5 + H) / n)
            return total_error, reusable

        return single_param_search(f=error_, rel_tol=tol, n=100, factor=2)[0:2]
    elif method == 'gaussian':
        return kernel_norm(H=H, T=T, p=1.) - np.sum(weights / nodes * (1 - exp_underflow(nodes * T)))
    else:
        raise NotImplementedError(f'The method {method} for computing the L^1 kernel error has not been implemented.')


def error_l2_optimal_weights(H, T, nodes, output='error'):
    """
    Computes an error estimate of the squared L^2-norm of the difference between the rough kernel and its approximation
    on [0, T]. Uses the best possible weights given the nodes specified.
    :param H: Hurst parameter
    :param nodes: The nodes of the approximation. Assumed that they are all non-zero
    :param output: If error, returns the error and the optimal weights. If gradient, returns the error, the gradient
        (of the nodes only), and the optimal weights. If hessian, returns the error, the gradient, the Hessian, and
        the optimal weights
    :param T: Final time, may also be a numpy array
    :return: An error estimate
    """
    if len(nodes) == 1:
        node = np.fmax(1e-04, nodes[0])
        gamma_1 = gamma(H + 0.5)

        if isinstance(T, np.ndarray):
            nT = node * T
            gamma_ints = gammainc(H + 0.5, nT)
            exp_node_matrix = exp_underflow(2 * nT)
            exp_node_vec = exp_underflow(nT)
            A = (1 - exp_node_matrix) / (2 * node)
            b = -2 * gamma_ints / node ** (H + 0.5)
            c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
            v = b / A
            err = c - 0.25 * b * v
            opt_weights = -0.5 * v
            if len(opt_weights.shape) > 1:
                opt_weights = opt_weights[-1, ...]
            if output == 'error' or output == 'err':
                return err, opt_weights

            A_grad = (-1 + (1 + 2 * nT) * exp_node_matrix) / (2 * node) ** 2
            b_grad = -2 * (nT ** (H + 0.5) * exp_node_vec[None, :] / gamma_1 - (H + 0.5) * gamma_ints) \
                / node ** (H + 1.5)
            grad = 0.5 * A_grad * v ** 2 - 0.5 * b_grad * v
            if output == 'gradient' or output == 'grad':
                return err, grad, opt_weights

            A_hess = 2 * (1 - (1 + 2 * nT + 2 * nT ** 2) * exp_node_matrix) / (8 * node ** 3)
            b_hess = -2 * (-(nT ** (H + 1.5) + (H + 1.5) * nT ** (H + 0.5)) * exp_node_vec / gamma_1 + (H + 0.5) * (
                    H + 1.5) * gamma_ints) / nodes ** (H + 2.5)
            U = b_grad / A
            Y = 2 * A_grad * v
            hess = 0.5 * (2 * Y * U - Y ** 2 / A + 2 * A_hess * v ** 2 - b_hess * v - b_grad * U)
            return err, grad, hess, opt_weights

        gamma_ints = gammainc(H + 0.5, node * T)
        exp_node_matrix = exp_underflow(2 * node * T)
        exp_node_vec = exp_underflow(node * T)
        A = (1 - exp_node_matrix) / (2 * node)
        b = -2 * gamma_ints / node ** (H + 0.5)
        if H > 0:
            c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
            v = b / A
            err = c - 0.25 * b * v
            opt_weight = np.array([-0.5 * v])
            if output == 'error' or output == 'err':
                return err, opt_weight
        else:
            v = b / A
            err = - 0.25 * b * v
            opt_weight = np.array([-0.5 * v])
            if output == 'error' or output == 'err':
                return err, opt_weight

        A_grad = (-1 + (1 + 2 * node * T) * exp_node_matrix) / (4 * node ** 2)
        b_grad = -2 * ((node * T) ** (H + 0.5) * exp_node_vec / gamma_1 - (H + 0.5) * gamma_ints) / node ** (H + 1.5)
        grad = 0.5 * (A_grad * v - b_grad) * v
        if output == 'gradient' or output == 'grad':
            return err, grad, opt_weight

        A_hess = 2 * (1 - (1 + 2 * node * T + 2 * (node * T) ** 2) * exp_node_matrix) / (8 * node ** 3)
        b_hess = -2 * (-((node * T) ** (H + 1.5) + (H + 1.5) * (node * T) ** (H + 0.5)) * exp_node_vec / gamma_1
                       + (H + 0.5) * (H + 1.5) * gamma_ints) / node ** (H + 2.5)
        U = b_grad / A
        Y = 2 * A_grad * v
        hess = 0.5 * (2 * Y * U - Y ** 2 / A + 2 * A_hess * v ** 2 - b_hess * v - b_grad * U)
        return err, grad, hess, opt_weight

    def invert_permutation(p):
        s = np.empty_like(p)
        s[p] = np.arange(p.size)
        return s

    perm = np.argsort(nodes)
    nodes = nodes[perm]
    nodes[0] = np.fmax(1e-04, nodes[0])
    for i in range(len(nodes) - 1):
        if 1.01 * nodes[i] > nodes[i + 1]:
            nodes[i + 1] = nodes[i] * 1.01
    nodes = nodes[invert_permutation(perm)]

    node_matrix = nodes[:, None] + nodes[None, :]
    gamma_1 = gamma(H + 0.5)

    if isinstance(T, np.ndarray):
        nT = np.outer(T, nodes)
        nmT = np.einsum('i,jk->ijk', T, node_matrix)
        gamma_ints = gammainc(H + 0.5, nT)
        exp_node_matrix = exp_underflow(nmT)
        exp_node_vec = exp_underflow(nT)
        A = (1 - exp_node_matrix) / node_matrix[None, :, :]
        b = -2 * gamma_ints / nodes[None, :] ** (H + 0.5)
        c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
        try:
            v = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            v = np.empty((len(T), len(nodes)))
            for i in range(len(T)):
                try:
                    v[i, :] = np.linalg.solve(A[i, ...], b[i, ...])
                except np.linalg.LinAlgError:
                    v[i, :] = np.linalg.lstsq(A[i, ...], b[i, ...], rcond=None)[0]
        err = c - 0.25 * np.sum(b * v, axis=1)
        opt_weights = -0.5 * v
        if len(opt_weights.shape) > 1:
            opt_weights = opt_weights[-1, ...]
        if output == 'error' or output == 'err':
            return err, opt_weights

        def mvp(A_, b_):
            return np.sum(A_ * b_[:, None, :], axis=-1)

        A_grad = (-1 + (1 + nmT) * exp_node_matrix[None, :, :]) / node_matrix[None, :, :] ** 2
        b_grad = -2 * (nT ** (H + 0.5) * exp_node_vec[None, :] / gamma_1 - (H + 0.5) * gamma_ints) \
            / nodes[None, :] ** (H + 1.5)
        grad = 0.5 * v * mvp(A_grad, v) - 0.5 * b_grad * v
        if output == 'gradient' or output == 'grad':
            return err, grad, opt_weights

        def diagonalize(x):
            new_x = np.empty((x.shape[0], x.shape[1], x.shape[1]))
            for k in range(x.shape[0]):
                new_x[k, :, :] = np.diag(x[k, :])
            return new_x

        def trans(x):
            return np.transpose(x, (0, 2, 1))

        A_hess = 2 * (1 - (1 + nmT + nmT ** 2 / 2) * exp_node_matrix[None, :, :]) / node_matrix[None, :, :] ** 3
        b_hess = -2 * (-(nT ** (H + 1.5) + (H + 1.5) * nT ** (H + 0.5)) * exp_node_vec / gamma_1 + (H + 0.5) * (
                    H + 1.5) * gamma_ints) / nodes[None, :] ** (H + 2.5)
        try:
            U = np.linalg.solve(A, diagonalize(b_grad))
        except np.linalg.LinAlgError:
            diag_b = diagonalize(b_grad)
            U = np.empty((len(T), len(nodes), len(nodes)))
            for i in range(len(T)):
                for j in range(len(nodes)):
                    try:
                        U[i, j, :] = np.linalg.solve(A[i, ...], diag_b[i, j, :])
                    except np.linalg.LinAlgError:
                        U[i, j, :] = np.linalg.lstsq(A[i, ...], diag_b[i, j, :])[0]
        Y = diagonalize(mvp(A_grad, v)) + A_grad * v[:, None, :]
        YTU = trans(Y) @ U
        hess = 0.5 * (YTU - trans(np.linalg.solve(A, Y)) @ Y + diagonalize(v * mvp(A_hess, v))
                      + v[:, None, :] * v[:, :, None] * A_hess - diagonalize(b_hess * v) - b_grad[:, :, None] * U
                      + trans(YTU))
        return err, grad, hess, opt_weights

    nT = nodes * T
    nmT = node_matrix * T
    gamma_ints = gammainc(H + 0.5, nT)
    exp_node_matrix = exp_underflow(nmT)
    A = (1 - exp_node_matrix) / node_matrix
    b = -2 * gamma_ints / nodes ** (H + 0.5)
    c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
    try:
        v = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        v = np.linalg.lstsq(A, b, rcond=None)[0]
    if np.amax(v) > 0:
        v = lsq_linear(A, b).x
    err = 0.25 * v @ A @ v - 0.5 * np.dot(b, v) + c  # c - 0.25 * np.dot(b, v)
    opt_weights = -0.5 * v
    if output == 'error' or output == 'err':
        return err, opt_weights

    exp_node_vec = exp_underflow(nT)
    A_grad = (-1 + (1 + nmT) * exp_node_matrix) / node_matrix ** 2
    b_grad = -2 * (nT ** (H + 0.5) * exp_node_vec / gamma_1 - (H + 0.5) * gamma_ints) / nodes ** (H + 1.5)
    grad = 0.5 * v * (A_grad @ v) - 0.5 * b_grad * v
    if output == 'gradient' or output == 'grad':
        return err, grad, opt_weights

    A_hess = 2 * (1 - (1 + nmT + nmT ** 2 / 2) * exp_node_matrix) / node_matrix ** 3
    b_hess = -2 * (-(nT ** (H + 1.5) + (H + 1.5) * nT ** (H + 0.5)) * exp_node_vec / gamma_1 + (H + 0.5) * (
            H + 1.5) * gamma_ints) / nodes ** (H + 2.5)
    try:
        U = np.linalg.solve(A, np.diag(b_grad))
    except np.linalg.LinAlgError:
        U = np.linalg.lstsq(A, b, rcond=None)[0]
    Y = np.diag(A_grad @ v) + A_grad * v[None, :]
    YTU = Y.T @ U
    hess = 0.5 * (YTU - np.linalg.solve(A, Y).T @ Y + np.diag(v * (A_hess @ v)) + v[None, :] * v[:, None] * A_hess
                  - np.diag(b_hess * v) - b_grad[:, None] * U + YTU.T)
    return err, grad, hess, opt_weights


def optimize_error_l2(H, N, T, tol=1e-08, bound=None, method='gradient', force_order=False, init_nodes=None,
                      iterative=False):
    """
    Optimizes the L^2 error with N points for the fractional kernel. Always uses the best weights and only numerically
    optimizes over the nodes.
    :param H: Hurst parameter
    :param N: Number of points
    :param T: Final time, may be a numpy array (only if grad is False and fast is True)
    :param tol: Error tolerance
    :param bound: Upper bound on the nodes. If no upper bound is desired, use None
    :param method: If error, uses only the error estimates for optimizing over the nodes, and uses the optimizer
        L-BFGS-B. If gradient, uses also the gradient of the error with respect to the nodes, and uses the optimizer
        L-BFGS-B. If hessian, uses also the gradient and the Hessian of the error with respect to the nodes, and uses
        the optimizer trust-constr
    :param force_order: Forces the nodes to stay in order, i.e. not switch places. May improve numerical stability
    :param init_nodes: May specify a starting point for the nodes
    :param iterative: If True, starts with 1 node and iteratively solves the optimization problem before adding another
        node
    :return: The minimal error together with the associated nodes and weights.
    """
    error_fun = error_l2_optimal_weights
    all_errors = np.empty(1)

    if iterative and not init_nodes and N >= 2:
        all_errors = np.empty(N)
        init_nodes = np.empty(N)
        all_errors[:-1], init_nodes[:-1], _ = optimize_error_l2(H=H, N=N - 1, T=T, tol=tol, bound=bound, method=method,
                                                                force_order=force_order, init_nodes=None,
                                                                iterative=iterative)
        init_nodes[:-1] = init_nodes[:N - 1] / 1.03 ** np.fmin(np.arange(1, N) ** 2, 100)
        if bound is not None:
            init_nodes[N - 1] = np.fmax(bound, 10 * init_nodes[N - 2])
        else:
            init_nodes[N - 1] = 5 * init_nodes[N - 2]

    # get starting value and bounds for the optimization problem
    if init_nodes is None:
        if bound is None:
            bound = 1e+100
            nodes_, w = quadrature_rule(H, N, T, mode='old geometric observation l2')
            if N == 2:
                bound = np.fmax(bound, np.amax(nodes_))
            if len(nodes_) < N:
                nodes = np.zeros(N)
                nodes[:len(nodes_)] = nodes_
                for i in range(len(nodes_), N):
                    nodes[i] = nodes_[-1] * 10 ** (i - len(nodes_) + 1)
            else:
                nodes = nodes_[:N]
        else:
            nodes = np.exp(np.linspace(0, np.log(np.fmin(bound, 5. ** (np.fmin(140, N - 1)) / T)), N))
    else:
        if bound is None:
            bound = 1e+100
        nodes = init_nodes
    lower_bound = 1 / (10 * N * np.amin(T)) * ((0.5 - H) / 0.4) ** 2
    nodes = np.fmin(np.fmax(nodes, lower_bound), bound)
    bounds = ((np.log(lower_bound), np.log(bound)),) * N
    original_error, original_weights = error_fun(H=H, T=T, nodes=nodes, output='error')
    original_nodes = nodes.copy()

    # carry out the optimization
    if force_order:
        constraints = []
        for i in range(1, N):
            def jac_here(x):
                res_ = np.zeros(N)
                res_[i] = 1
                res_[i - 1] = -1
                return res_
            constraints = constraints + [{'type': 'ineq', 'fun': lambda x: x[i] - x[i - 1] - 0.3, 'jac': jac_here}]

        if method == 'error' or method == 'err':
            def func(x):
                return error_fun(H, T, np.exp(x), output='error')[0] \
                    * (1 + np.sum(np.exp(- (x[1:] - x[:-1]) ** 2 * 3 * N / (0.72 * np.log(5 / H) - np.log(T)))))

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, constraints=constraints)

        else:
            def func(x):
                err_, grad, _ = error_fun(H, T, np.exp(x), output='gradient')
                return err_, np.exp(x) * grad

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, constraints=constraints, jac=True)

    else:
        if method == 'error' or method == 'err':
            def func(x):
                return error_fun(H, T, np.exp(x), output='error')[0]

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds)

        elif method == 'gradient' or method == 'grad':
            def func(x):
                err_, grad, _ = error_fun(H, T, np.exp(x), output='gradient')
                return err_, np.exp(x) * grad

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, jac=True)

        else:
            def func(x):
                err_, grad, _ = error_fun(H, T, np.exp(x), output='gradient')
                return err_, np.exp(x) * grad

            def hess(x):
                _, grad, hessian, _ = error_fun(H, T, np.exp(x), output='hessian')
                return hessian * np.exp(x[None, :] + x[:, None]) + np.diag(grad * np.exp(x))

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, jac=True, hess=hess, method='trust-constr')

    # post-processing, ensuring that the results are of good quality
    nodes = np.exp(res.x)
    err, weights = error_fun(H=H, T=T, nodes=nodes, output='error')
    if H > 0:
        if err > 2 * np.fmax(original_error, 1e-9):
            # return np.sqrt(np.fmax(original_error, 0)) / kernel_norm(H, T), original_nodes, original_weights
            return np.sqrt(np.fmax(original_error, 0)), original_nodes, original_weights
        # all_errors[-1] = np.sqrt(np.fmax(err, 0)) / kernel_norm(H, T)
        all_errors[-1] = np.sqrt(np.fmax(err, 0))
    else:
        if err > 0.5 * original_error:
            # return np.sqrt(np.fmax(original_error, 0)) / kernel_norm(H, T), original_nodes, original_weights
            return original_error, original_nodes, original_weights
        # all_errors[-1] = np.sqrt(np.fmax(err, 0)) / kernel_norm(H, T)
        all_errors[-1] = err
    return all_errors, nodes, weights


def optimize_error_l1(H, N, T, iterative=False, init_nodes=None, init_weights=None):
    """
    Optimizes the L^1 error with N points for the fractional kernel.
    :param H: Hurst parameter
    :param N: Number of points
    :param T: Final time, may be a numpy array (only if grad is False and fast is True)
    :param iterative: If True, starts with one node and iteratively adds nodes, while always optimizing
    :param init_nodes: May specify a starting point for the nodes
    :param init_weights: May specify a starting point for the weights
    :return: The minimal relative error together with the associated nodes and weights.
    """
    def optimize_error_given_rule(nodes_1, weights_1):
        N_ = len(nodes_1)
        coefficient = 1 / kernel_norm(H=H, T=T, p=1.)
        rule = np.log(np.concatenate((nodes_1, weights_1)))

        def func(x):
            err_, grad = error_l1(H=H, nodes=np.exp(x[:N_]), weights=np.exp(x[N_:]), T=T, method='intersections')
            return coefficient * err_

        res = minimize(func, rule, tol=1e-04)
        nodes_1, weights_1 = sort(np.exp(res.x[:N_]), np.exp(res.x[N_:]))
        return res.fun, nodes_1, weights_1

    if not iterative:
        if init_nodes is not None and init_weights is not None:
            nodes_, weights_ = init_nodes, init_weights
        else:
            nodes_, weights_ = quadrature_rule(H=H, N=N, T=T, mode='non-geometric l1')
        if len(nodes_) < N:
            nodes = np.zeros(N)
            weights = np.zeros(N)
            nodes[:len(nodes_)] = nodes_
            weights[:len(weights_)] = weights_
            for i in range(len(nodes_), N):
                nodes[i] = nodes_[-1] * 2 ** (i - len(nodes_) + 1)
                weights[i] = weights_[-1]
        else:
            nodes = nodes_[:N]
            weights = weights_[:N]

        return optimize_error_given_rule(nodes, weights)

    if init_nodes is None or init_weights is None:
        nodes, weights = quadrature_rule(H=H, N=1, T=T, mode='non-geometric l1')
        err, nodes, weights = optimize_error_given_rule(nodes, weights)
    else:
        err, nodes, weights = -1, init_nodes, init_weights

    while len(nodes) < N:
        print(len(nodes))
        nodes = np.append(nodes, 2 * nodes[-1])
        weights = np.append(weights, np.amax(weights))
        err, nodes, weights = optimize_error_given_rule(nodes, weights)

    return err, nodes, weights


def european_rule(H, N, T):
    """
    Returns a quadrature rule that is optimized for pricing European options under the rough Heston model.
    :param H: Hurst parameter
    :param N: Number of nodes
    :param T: Final time/Maturity
    :return: Nodes and weights
    """

    def optimizing_func(N_, tol_, bound_):
        if N_ == 1:
            nod = np.array([1 / T])
        else:
            nod = np.empty(N_)
            if len(last_nodes) == N_:
                nod = last_nodes
            else:
                nod[:-1] = last_nodes
                nod[-1] = bound_
        nod = nod / 1.03 ** np.fmin(np.arange(1, N_ + 1) ** 2, 100)
        return optimize_error_l2(H=H, N=N_, T=T, tol=tol_, bound=bound_, method='gradient', force_order=False,
                                 init_nodes=nod)

    if H > 0:
        _, nodes, weights = optimizing_func(N_=1, tol_=1e-06, bound_=None)
    else:
        _, nodes, weights = optimize_error_l1(H=H, N=1, T=T)
    if N == 1:
        return nodes, weights

    L_step = 1.15
    bound = np.amax(nodes) / L_step
    current_N = 1
    last_nodes = nodes

    while current_N < N:
        increase_N = 0
        L_step = 1.15

        while increase_N < 2:
            bound = bound * L_step
            error_, nodes, weights = optimizing_func(N_=current_N+1, tol_=1e-07/current_N, bound_=bound)
            p = np.argsort(nodes)
            nodes = nodes[p]
            weights = weights[p]
            if np.amin(nodes[1:] / nodes[:-1]) < 1.4 or np.abs(np.amin(weights)) < 1e-02 \
                    or np.abs(np.amin(weights[1:] / weights[:-1])) < 0.4:
                increase_N = 0
                L_step = 1.15
            elif error_ < optimizing_func(N_=current_N, tol_=1e-07/current_N, bound_=bound)[0]:
                increase_N += 1
                if L_step > 1.06:
                    L_step = 1.05
                    bound = bound / 1.15
            else:
                increase_N = 0
                L_step = 1.15

        current_N = current_N + 1
        last_nodes = nodes

    if N >= 4:
        return nodes, weights
    if N == 2:
        L_4 = bound * 2
        L_5 = bound * 3
        L_6 = bound * 4
    else:  # N == 3
        L_4 = bound
        L_5 = bound * 1.25
        L_6 = bound * 1.5
    error_4, nodes_4, weights_4 = optimizing_func(N_=N, tol_=1e-08, bound_=L_4)
    error_5, nodes_5, weights_5 = optimizing_func(N_=N, tol_=1e-08, bound_=L_5)
    error_6, nodes_6, weights_6 = optimizing_func(N_=N, tol_=1e-08, bound_=L_6)
    if error_4 <= error_5 and error_4 <= error_6:
        return nodes_4, weights_4
    if error_5 <= error_6:
        return nodes_5, weights_5
    return nodes_6, weights_6


def harms_rule(H, n, m):
    """
    The quadrature rule for fBm proposed by Harms.
    :param H: Hurst parameter
    :param n: Number of Gaussian quadrature intervals
    :param m: Degree of Gaussian quadrature
    :return: The nodes and weights of the quadrature rule
    """
    alpha_, beta_, gamma_, delta_ = H + 1/2, m - 1, 1/2 - H, H
    r = delta_ * m / (1 - alpha_ - beta_ + delta_ + m)
    xi_0 = n ** (-r / gamma_)
    xi_n = n ** (r / delta_)
    xi = xi_0 * np.exp(np.log(xi_n / xi_0) * np.linspace(0, 1, n + 1))
    return Gaussian_on_partition(H=H, m=m, partition=xi, fractional_weight=True)


def quadrature_rule(H, N, T, mode="european"):
    """
    Returns the nodes and weights of a quadrature rule for the fractional kernel with Hurst parameter H. The nodes are
    sorted in increasing order.
    :param H: Hurst parameter
    :param N: Total number of nodes
    :param T: Final time
    :param mode: The kind of quadrature rule that should be used
    :return: All the nodes and weights, in the form [node1, node2, ...], [weight1, weight2, ...]
    """
    if isinstance(T, np.ndarray):
        if N == 1:
            T = np.amin(T) ** (3 / 5) * np.amax(T) ** (2 / 5)
        elif N == 2:
            T = np.amin(T) ** (1 / 2) * np.amax(T) ** (1 / 2)
        elif N == 3:
            T = np.amin(T) ** (1 / 3) * np.amax(T) ** (2 / 3)
        elif N == 4:
            T = np.amin(T) ** (1 / 4) * np.amax(T) ** (3 / 4)
        elif N == 5:
            T = np.amin(T) ** (1 / 6) * np.amax(T) ** (5 / 6)
        elif N == 6:
            T = np.amin(T) ** (1 / 10) * np.amax(T) ** (9 / 10)
        else:
            T = np.amax(T)

    if mode == "optimized l2" or mode == "OL2":
        nodes, weights = optimize_error_l2(H=H, N=N, T=T)[1:3]
    elif mode == "optimized l1" or mode == "OL1":
        nodes, weights = optimize_error_l1(H=H, N=N, T=T, iterative=True)[1:3]
    elif mode == "european" or mode == "BL2":
        nodes, weights = european_rule(H=H, N=N, T=T)
    elif mode == "abi jaber" or mode == "AE":
        nodes, weights = AbiJaberElEuch_quadrature_rule(H=H, N=N, T=T)
    elif mode == "alfonsi" or mode == "AK":
        nodes, weights = AK_improved_rule(H=H, N=N, T=T)
    elif mode == "paper" or mode == "OLD":
        nodes, weights = Gaussian_rule(H=H, N=N, T=T, mode="old geometric observation l2")
    else:
        nodes, weights = Gaussian_rule(H=H, N=N, T=T, mode=mode)
    weights[np.logical_and(nodes < 1, np.abs(weights) > 100)] = 0
    return sort(nodes, weights)

# ---------------------------------------------------------------------------------------

# Functions taken from Christian




class kernel_frac:
    """Class representing the RL kernel."""

    def __init__(self, H, eta):
        self.H = H
        self.eta = eta
        self.eta_tilde = np.sqrt(2 * H) * eta

    def K_diag(self, Delta, N):
        """
        Return the diagonal values of calligraphic K_{i,j} as defined by Jim.
        Parameters
        ----------
        Delta : double
            Time increment for the simulation scheme, Delta = T / N.
        N : int
            Number of time steps in the simulation scheme.
        Returns
        -------
        numpy array
            The values mathcal{K}_{j,j}(Delta), j=0, ..., N-1. Size = N.
        """
        i = np.arange(N + 1)
        # Hint: i[-N:] = (i[1],...,i[N]), i[:N] = (i[0],...,i[N-1])
        return self.eta ** 2 * Delta ** (2 * self.H) * (i[-N:] ** (2 * self.H) - i[:N] ** (2 * self.H))

    def K_0(self, Delta):
        """
        Return the value of calligraphic K_0 as defined by Jim.
        Parameters
        ----------
        Delta : double
            Time increment for the simulation scheme, Delta = T / N..
        Returns
        -------
        double
            The value mathcal{K}_0(Delta).
        """
        return self.eta_tilde * Delta ** (self.H + 0.5) / (self.H + 0.5)


class kernel_rheston:
    """Kernel for the rHeston model, seen as a forward variance model.

    The shape of the kernel is defined in the paper of Jim and Martin, 2019."""

    def __init__(self, H, lam, zeta, eps=1e-3):
        self.alpha = H + 0.5
        self.H = H  # Is this really the "H"???
        self.lam = lam
        self.zeta = zeta
        self.eps = eps

    def _k(self, r):
        """The kernel."""
        return self.zeta * r ** (self.alpha - 1) * mittag_leffler(- self.lam * r ** self.alpha, self.alpha, self.alpha)

    def K_0(self, Delta):
        """
        Return the value of calligraphic K_0 as defined by Jim. Computed by
        computationally expensive numerical integration.
        Parameters
        ----------
        Delta : double
            Time increment for the simulation scheme, Delta = T / N..
        Returns
        -------
        double
            The value mathcal{K}_0(Delta).
        """
        return integ.quad(lambda r: self._k(r), 0.0, Delta, epsabs=self.eps, epsrel=self.eps)[0]

    def K_diag(self, Delta, N):
        """
        Return the diagonal values of calligraphic K_{i,j} as defined by Jim.
        Computed by computationally expensive numerical integration.
        Parameters
        ----------
        Delta : double
            Time increment for the simulation scheme, Delta = T / N.
        N : int
            Number of time steps in the simulation scheme.
        Returns
        -------
        numpy array
            The values mathcal{K}_{j,j}(Delta), j=0, ..., N-1. Size = N.
        """
        return np.array([integ.quad(lambda r: self._k(r + i * Delta) ** 2, 0.0, Delta,
                                    epsabs=self.eps, epsrel=self.eps)[0] for i in range(N)])

    def xi(self, t_grid, v0, lam, theta, eps=1e-6):
        """
        Compute the forward variance curve for the rough Heston model.

        Note that the forward variance curve is a constant if v0 is equal
        to theta.
        Parameters
        ----------
        t_grid : numpy array or scalar.
            Time grid or value, along which the forward variance curve is
            requested. Assumed to be non-negative.
        v0 : scalar.
            Initial variance.
        lam : scalar.
            Speed of mean reversion.
        theta : scalar
            Long-term mean variance.
        eps : positive scalar, optional
            Accuracy target for the quadrature method. The default is 1e-6.
        Returns
        -------
        Numpy array.
            The forward variance curve.
        """
        if np.isclose(v0, theta, rtol=eps):
            return np.full_like(t_grid, v0)

        t = np.unique(np.append(0.0, t_grid))

        # integrate between t_i and t_{i+1}
        def f(i):
            return integ.quad(self._k, t[i], t[i + 1], epsabs=eps, epsrel=eps)[0]

        int_k = np.array([f(i) for i in range(len(t) - 1)])
        if isinstance(t_grid, np.ndarray) and t_grid[0] == 0.0:
            int_k = np.append(0.0, int_k)
        elif t_grid == 0.0:
            int_k = np.append(0.0, int_k)

        # Note that the forward variance curve uses the kernel without zeta.
        return v0 + self.lam * (theta - v0) * np.cumsum(int_k) / self.zeta
