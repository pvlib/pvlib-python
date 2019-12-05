import numpy as np
from pvlib.ivtools.lambertw import lambertw


def calc_theta_phi_exact(imp, il, vmp, io, nnsvth, rs, rsh):
    """
    CALC_THETA_PHI_EXACT computes Lambert W values appearing in the analytic
    solutions to the single diode equation for the max power point.

    Syntax
    ------
    theta, phi = calc_theta_phi_exact(imp, il, vmp, io, nnsvth, rs, rsh)

    Description
    -----------
    calc_theta_phi_exact calculates values for the Lambert W function which
    are used in the analytic solutions for the single diode equation at the
    maximum power point. For V=V(I),
    phi = W(Io*Rsh/n*Vth * exp((IL + Io - Imp)*Rsh/n*Vth)). For I=I(V),
    theta = W(Rs*Io/n*Vth *
    Rsh/ (Rsh+Rs) * exp(Rsh/ (Rsh+Rs)*((Rs(IL+Io) + V)/n*Vth))

    Parameters
    ----------
    imp: a numpy array of length N of values for Imp (A)
    il: a numpy array of length N of values for the light current IL (A)
    vmp: a numpy array of length N of values for Vmp (V)
    io: a numpy array of length N of values for Io (A)
    nnsvth: a numpy array of length N of values for the diode factor x
            thermal voltage for the module, equal to Ns
            (number of cells in series) x Vth
            (thermal voltage per cell).
    rs: a numpy array of length N of values for the series resistance (ohm)
    rsh: a numpy array of length N of values for the shunt resistance (ohm)

    Returns
    -------
    theta: a numpy array of values for the Lamber W function for solving
           I = I(V)
    phi: a numpy array of values for the Lambert W function for solving
         V = V(I)

    References
    ----------
    [1] PVLib MATLAB
    [2] C. Hansen, Parameter Estimation for Single Diode Models of Photovoltaic
        Modules, Sandia National Laboratories Report SAND2015-XXXX
    [3] A. Jain, A. Kapoor, "Exact analytical solutions of the parameters of
        real solar cells using Lambert W-function", Solar Energy Materials and
        Solar Cells, 81 (2004) 269-277.
    """

    # Argument for Lambert W function involved in V = V(I) [2] Eq. 12; [3]
    # Eq. 3
    # TODO: delete this, b/c replaced with np.where() below
    # if any(nnsvth == 0.):
    #     argw = []
    #     for i in nnsvth:
    #         if i == 0.:
    #             argw.append(float("Inf"))
    #         else:
    #             argw.append(rsh * io / i * np.exp(rsh * (il + io - imp) / i))
    #     argw = np.array(argw)
    # else:
    #     argw = rsh * io / nnsvth * np.exp(rsh * (il + io - imp) / nnsvth)
    argw = np.where(
        nnsvth == 0,
        np.inf,
        rsh * io / nnsvth * np.exp(rsh * (il + io - imp) / nnsvth))
    u = argw > 0
    w = np.zeros(len(u))
    w[~u] = float("Nan")
    if any(argw[u] == float("Inf")):
        tmp = []
        for i in argw[u]:
            if i == float("Inf"):
                tmp.append(float("Nan"))
            else:
                tmp.append(lambertw(i).real)
        tmp = np.array(tmp, dtype=float)
    else:
        tmp = lambertw(argw[u]).real
    ff = np.isnan(tmp)

    # NaN where argw overflows. Switch to log space to evaluate
    if any(ff):
        logargw = np.log(rsh[u]) + np.log(io[u]) - np.log(nnsvth[u]) + \
                  rsh[u] * (il[u] + io[u] - imp[u]) / nnsvth[u]
        # Three iterations of Newton-Raphson method to solve w+log(w)=logargW.
        # The initial guess is w=logargW. Where direct evaluation (above)
        # results in NaN from overflow, 3 iterations of Newton's method gives
        # approximately 8 digits of precision.
        x = logargw
        for i in range(3):
            x *= ((1. - np.log(x) + logargw) / (1. + x))
        tmp[ff] = x[ff]
    w[u] = tmp
    phi = np.transpose(w)

    # Argument for Lambert W function involved in I = I(V) [2] Eq. 11; [3]
    # E1. 2
    # TODO: delete this, b/c replaced with np.where() below
    # if any(nnsvth == 0.):
    #     argw = []
    #     for i in nnsvth:
    #         if i == 0.:
    #             argw.append(float("Inf"))
    #         else:
    #             argw.append(rsh / (rsh + rs) * rs * io / i *
    #                         np.exp(rsh / (rsh + rs) * (rs * (il + io) + vmp) /
    #                                i))
    #     argw = np.array(argw)
    # else:
    #     argw = rsh / (rsh + rs) * rs * io / nnsvth * \
    #            np.exp(rsh / (rsh + rs) * (rs * (il + io) + vmp) / nnsvth)
    argw = np.where(
        nnsvth == 0,
        np.inf,
        rsh / (rsh + rs) * rs * io / nnsvth * np.exp(
            rsh / (rsh + rs) * (rs * (il + io) + vmp) / nnsvth))
    u = argw > 0
    w = np.zeros(len(u))
    w[~u] = float("Nan")
    if any(argw[u] == float("Inf")):
        tmp = []
        for i in argw[u]:
            if i == float("Inf"):
                tmp.append(float("Nan"))
            else:
                tmp.append(lambertw(i).real)
        tmp = np.array(tmp, dtype=float)
    else:
        tmp = lambertw(argw[u]).real
    ff = np.isnan(tmp)

    # NaN where argw overflows. Switch to log space to evaluate
    if any(ff):
        logargw = np.log(rsh[u]) / (rsh[u] + rs[u]) + np.log(rs[u]) + \
                  np.log(io[u]) - np.log(nnsvth[u]) + \
                  (rsh[u] / (rsh[u] + rs[u])) * \
                  (rs[u] * (il[u] + io[u]) + vmp[u]) / nnsvth[u]
        # Three iterations of Newton-Raphson method to solve w+log(w)=logargW.
        # The initial guess is w=logargW. Where direct evaluation (above)
        # results in NaN from overflow, 3 iterations of Newton's method gives
        # approximately 8 digits of precision.
        x = logargw
        for i in range(3):
            x *= ((1. - np.log(x) + logargw) / (1. + x))
        tmp[ff] = x[ff]
    w[u] = tmp
    theta = np.transpose(w)
    return theta, phi
