import numpy as np
from numba import njit

from . import pde
from .utils import ncdf, npdf


@njit
def compute_volterra_green(maturity: float,
                           mt: int,
                           xi: float,
                           gridx: np.ndarray,
                           advection: np.ndarray,
                           diffusion: np.ndarray
                           ) -> np.ndarray:
    """
    %This function calculates Green's function of xi and x by solving the linear system of Volterra
    %equations for the Heston or any other model.
    %xi is the location of the boundary
    %advection defines the shape of the boundary, diffusion maps
    %calendar time into stretched time, hence also changing the shape of the
    %boundary
    """
    nt = advection.shape[0] - 1
    dt = maturity/nt
    tt = np.zeros(nt+1)
    bb = np.zeros(nt+1)
    tt[0] = 0.0
    bb[0] = 0.0
    for k in np.arange(1, nt+1):
        tt[k] = tt[k-1] + dt*diffusion[k]
        bb[k] = bb[k-1] + dt*advection[k]

    nt1 = (mt + 1) * nt  # for python array
    tt1 = np.zeros(nt1 + 1)
    bb1 = np.zeros(nt1 + 1)
    for k in np.arange(0, nt):
        for l in np.arange(0, mt+1):  # linear interp
            tt1[k + k*mt + l] = ((mt+1-l)*tt[k]+l*tt[k+1])/(mt+1)
            bb1[k + k*mt + l] = ((mt+1-l)*bb[k]+l*bb[k+1])/(mt+1)
    tt1[nt1] = tt[nt]
    bb1[nt1] = bb[nt]

    ker1 = np.zeros((nt1+1, nt1+1))
    for k in np.arange(1, nt1+1):
        for l in np.arange(0, k):
            ker1[k, l] = -(bb1[k]-bb1[l])/(tt1[k]-tt1[l])
    for k in np.arange(1, nt1+1):
        ker1[k, k] = -(bb1[k]-bb1[k-1])/(tt1[k]-tt1[k-1])
    ker1[0, 0] = ker1[1, 1]

    ker2 = np.zeros((nt1+1, nt1+1))
    ker2[0, 0] = 1.0
    for k in np.arange(1, nt1+1):
        for l in np.arange(0, k):
            ker2[k, l] = np.exp(-(tt1[k]-tt1[l])*np.square(ker1[k,l])/2.0)
        ker2[k, k] = 1.0

    ker3 = np.zeros((nt1+1, nt1+1))
    for k in np.arange(0, nt1+1):
        for l in np.arange(0, k+1):
            ker3[k,l] = ker1[k, l]*ker2[k, l]
    ker3 = ker3 / np.sqrt(2.0*np.pi)

    pp = np.zeros((nt1+1, nt1+1))
    for k in np.arange(1, nt1+1):
        for l in np.arange(1, k+1):
            pp[k, l] = (tt1[l]-tt1[l-1])/(np.sqrt(tt1[k]-tt1[l])+np.sqrt(tt1[k]-tt1[l-1]))

    f = np.zeros(nt1+1)
    f[1] = 0.0
    for k in np.arange(1, nt1+1):
        f[k] = np.exp(-0.5*np.square(xi-bb1[k])/tt1[k])/np.sqrt(2.0*np.pi*tt1[k])

    phi = np.zeros(nt1+1)
    phi[0] = 0.0
    phi[1] = f[1]/(1+pp[1, 1]*ker3[1, 1])
    for k in np.arange(2, nt1+1):
       aux = 0.0
       for l in np.arange(1, k):
          aux = aux + pp[k, l]*(ker3[k, l]*phi[l]+ker3[k, l-1]*phi[l-1])
       phi[k] = (f[k]-pp[k, k]*ker3[k, k-1]*phi[k-1]-aux)/(1+pp[k, k]*ker3[k, k])

    nx = gridx.shape[0]-1
    grn = np.zeros(nx + 1)
    for l in np.arange(1, nx + 1):
        z = gridx[l] - xi
        g = np.zeros(nt1 + 1)

        if z < 0.1:
            aux1 = np.exp(-z * ker1[nt1, nt1]) * phi[nt1]
            for k in np.arange(0, nt1):
                g[k] = z * np.exp(-0.5*np.square(z) / (tt1[nt1]-tt1[k])) * \
                       (np.exp(-z * ker1[nt1, k]) * ker2[nt1, k] * phi[k]- aux1) / (tt1[nt1] - tt1[k])
                g[k] = g[k] + ker1[nt1, k] * np.exp(-0.5*np.square(z - bb1[nt1] + bb1[k]) / (tt1[nt1] - tt1[k])) * phi[k]
            g = g / np.sqrt(2.0 * np.pi)
            g[nt1] = 0
            grn[l] = npdf((gridx[l]-bb1[nt1])/np.sqrt(tt1[nt1]))/np.sqrt(tt1[nt1])-2.0*aux1*ncdf(-z/np.sqrt(tt1[nt1]))
        else:
            for k in np.arange(0, nt1):
                g[k] = (z-bb1[nt1]+bb1[k])*np.exp(-0.5*np.square(z-bb1[nt1]+bb1[k])/(tt1[nt1]-tt1[k]))*phi[k]/(tt1[nt1]-tt1[k])
            g = g / np.sqrt(2.0 * np.pi)
            g[nt1] = 0
            grn[l] = npdf((gridx[l]-bb1[nt1])/np.sqrt(tt1[nt1]))/np.sqrt(tt1[nt1])

        for k in np.arange(1, nt1+1):
            grn[l] = grn[l]-pp[nt1, k]*(g[k]+g[k-1])

    return grn


@njit
def compute_volterra_survival_prob(maturity: float,
                                   mt: int,
                                   xi: float,
                                   gridx: np.ndarray,
                                   advection: np.ndarray,
                                   diffusion: np.ndarray
                                   ) -> np.ndarray:
    """
    %This function calculates survival probability of xi and x by solving the linear system of Volterra
    %equations for the Heston or any other model.
    %xi is the location of the boundary
    %advection defines the shape of the boundary, diffusion maps
    %calendar time into stretched time, hence also changing the shape of the boundary
    """
    nt = advection.shape[0] - 1
    dt = maturity/nt
    tt = np.zeros(nt+1)
    bb = np.zeros(nt+1)
    tt[0] = 0.0
    bb[0] = 0.0
    for k in np.arange(1, nt+1):
        tt[k] = tt[k-1] + dt*diffusion[k]
        bb[k] = bb[k-1] + dt*advection[k]

    uu = np.zeros(nt+1)
    cc = np.zeros(nt+1)
    for k in np.arange(0, nt+1):
        uu[k] = tt[nt]-tt[nt-k]
        cc[k] = bb[nt-k]

    nt1 = (mt + 1) * nt  # for python array
    uu1 = np.zeros(nt1 + 1)
    cc1 = np.zeros(nt1 + 1)
    for k in np.arange(0, nt):
        for l in np.arange(0, mt+1):  # linear interp
            uu1[k + k*mt + l] = ((mt+1-l)*uu[k]+l*uu[k+1])/(mt+1)
            cc1[k + k*mt + l] = ((mt+1-l)*cc[k]+l*cc[k+1])/(mt+1)
    uu1[nt1] = uu[nt]
    cc1[nt1] = cc[nt]

    ker1 = np.zeros((nt1+1, nt1+1))
    for k in np.arange(1, nt1+1):
        for l in np.arange(0, k):
            ker1[k, l] = -(cc1[k]-cc1[l])/(uu1[k]-uu1[l])
    for k in np.arange(1, nt1+1):
        ker1[k, k] = -(cc1[k]-cc1[k-1])/(uu1[k]-uu1[k-1])
    ker1[0, 0] = ker1[1, 1]

    ker2 = np.zeros((nt1+1, nt1+1))
    ker2[0, 0] = 1.0
    for k in np.arange(1, nt1+1):
        for l in np.arange(0, k):
            ker2[k, l] = np.exp(-(uu1[k]-uu1[l])*np.square(ker1[k,l])/2.0)
        ker2[k, k] = 1.0

    ker3 = np.zeros((nt1+1, nt1+1))
    for k in np.arange(0, nt1+1):
        for l in np.arange(0, k+1):
            ker3[k, l] = ker1[k, l]*ker2[k, l]
    ker3 = ker3 / np.sqrt(2.0*np.pi)

    pp = np.zeros((nt1+1, nt1+1))
    for k in np.arange(1, nt1+1):
        for l in np.arange(1, k+1):
            pp[k, l] = (uu1[l]-uu1[l-1])/(np.sqrt(uu1[k]-uu1[l])+np.sqrt(uu1[k]-uu1[l-1]))

    f = np.ones(nt1+1)
    phi = np.zeros(nt1+1)
    phi[0] = f[0]
    phi[1] = f[1]/(1+pp[1, 1]*ker3[1, 1])
    for k in np.arange(2, nt1+1):
       aux = 0.0
       for l in np.arange(1, k):
            aux = aux + pp[k, l]*(ker3[k, l]*phi[l]+ker3[k, l-1]*phi[l-1])
       phi[k] = (f[k]-pp[k, k]*ker3[k, k-1]*phi[k-1]-aux)/(1.0+pp[k, k]*ker3[k, k])

    nx = gridx.shape[0]-1
    price = np.ones(nx + 1)
    price[0] = 0.0
    for l in np.arange(1, nx + 1):
        g = np.zeros(nt1 + 1)
        z = gridx[l] - xi

        if z <= 0.2:
            aux = np.exp(-z * ker1[nt1, nt1]) * phi[nt1]
            for k in np.arange(0, nt1):
                g[k] = z*np.exp(-0.5 * np.square(z) / (uu1[nt1] - uu1[k]))*(np.exp(-z*ker1[nt1, k])*ker2[nt1, k]*phi[k]-aux) / (uu1[nt1] - uu1[k])
                g[k] = g[k] + ker1[nt1, k]*np.exp(-0.5*np.square(z-cc1[nt1]+cc1[k])/(uu1[nt1]-uu1[k]))*phi[k]
            g = g / np.sqrt(2.0 * np.pi)
            g[nt1] = 0

            price[l] = 1 - 2.0* aux * ncdf(-z / np.sqrt(uu1[nt1]))
            for k in np.arange(1, nt1+1):
                price[l] = price[l]-pp[nt1, k]*(g[k]+g[k-1])
        else:
            for k in np.arange(0, nt1):
                g[k] = (z-cc1[nt1]+cc1[k])*(np.exp(-0.5 * np.square(z-cc1[nt1]+cc1[k]) / (uu1[nt1] - uu1[k])) / (uu1[nt1] - uu1[k]))*phi[k]
            g = g / np.sqrt(2.0 * np.pi)

            price[l] = 1.0
            for k in np.arange(1, nt1+1):
                price[l] = price[l]-pp[nt1, k]*(g[k]+g[k-1])

    return price


@njit
def compute_pde_green(maturity: float,
                      mt: int,
                      gridx: np.ndarray,
                      advection: np.ndarray,
                      diffusion: np.ndarray
                      ) -> np.ndarray:
    nt = advection.shape[0] - 1
    dt = maturity / nt
    tt = np.zeros(nt + 1)
    for k in np.arange(1, nt + 1):
        tt[k] = tt[k - 1] + dt * diffusion[k]

    dx = gridx[1] - gridx[0]
    y = pde.set_one_to_nearest(a=gridx, x0=0.0) / dx
    const_one = np.ones_like(gridx)

    for k in np.arange(0, nt):
        for j in np.arange(0, mt + 1):
            dtau = (tt[k + 1] - tt[k]) / (mt + 1)
            mu = advection[k + 1] / diffusion[k + 1]
            alpha = mu * dtau / 4.0 / dx
            beta = dtau / 4.0 / np.square(dx)

            a1, a2, a3 = (-alpha - beta) * const_one, (1.0 + 2.0 * beta) * const_one, (alpha - beta) * const_one
            b1, b2, b3 = (alpha + beta) * const_one, (1.0 - 2.0 * beta) * const_one, (-alpha + beta) * const_one
            a2[0], b2[0] = 1.0, 1.0
            a2[-1], b2[-1] = 1.0, 1.0
            a3[0], b3[0] = 0.0, 0.0
            a1[-1], b1[-1] = 0.0, 0.0

            u1 = pde.tridag_mult(a=b1, b=b2, c=b3, x=y)
            y = pde.tridag_solve(a=a1, b=a2, c=a3, r=u1)

    grn = y

    return grn


@njit
def compute_pde_survival_prob(maturity: float,
                              mt: int,
                              gridx: np.ndarray,
                              advection: np.ndarray,
                              diffusion: np.ndarray
                              ) -> np.ndarray:
    nt = advection.shape[0] - 1
    dt = maturity / nt
    tt = np.zeros(nt + 1)
    for k in np.arange(1, nt + 1):
        tt[k] = tt[k - 1] + dt * diffusion[k]

    dx = gridx[1] - gridx[0]
    y = np.ones_like(gridx)
    y[0] = 0.0
    const_one = np.ones_like(gridx)

    for k in np.arange(1, nt):
        for j in np.arange(0, mt+1):
            dtau = (tt[nt - k + 1] - tt[nt-k]) / (mt + 1)
            mu = advection[nt - k + 1] / diffusion[nt - k + 1]
            alpha = mu * dtau / 4.0 / dx
            beta = dtau / 4.0 / np.square(dx)

            a1, a2, a3 = (alpha - beta) * const_one, (1.0 + 2.0 * beta) * const_one, (-alpha - beta) * const_one
            b1, b2, b3 = (-alpha + beta) * const_one, (1.0 - 2.0 * beta) * const_one, (alpha + beta) * const_one

            a2[0], b2[0] = 1.0, 1.0
            a2[-1], b2[-1] = 1.0, 1.0
            a3[0], b3[0] = 0.0, 0.0
            a1[-1], b1[-1] = 0.0, 0.0

            u1 = pde.tridag_mult(a=b1, b=b2, c=b3, x=y)
            y = pde.tridag_solve(a=a1, b=a2, c=a3, r=u1)

    grn = y

    return grn
