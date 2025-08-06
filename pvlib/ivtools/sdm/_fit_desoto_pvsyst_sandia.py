"""
helper functions used in desoto.fit_desoto_sandia and pvsyst.fit_pvsyst_sandia
"""

import numpy as np

from scipy import optimize
from scipy.special import lambertw

from pvlib.pvsystem import singlediode, v_from_i
from pvlib.ivtools.utils import rectify_iv_curve, _numdiff
from pvlib.pvsystem import _pvsyst_Rsh


def _initial_iv_params(ivcurves, ee, voc, isc, rsh, nnsvth):
    # sets initial values for iph, io, rs and quality filter u.
    # Helper function for fit_<model>_sandia.
    n = len(ivcurves['v_oc'])
    io = np.ones(n)
    iph = np.ones(n)
    rs = np.ones(n)

    for j in range(n):

        if rsh[j] > 0:
            volt, curr = rectify_iv_curve(ivcurves['v'][j],
                                          ivcurves['i'][j])
            # Initial estimate of Io, evaluate the single diode model at
            # voc and approximate Iph + Io = Isc [5] Step 3a; [6] Step 3b
            io[j] = (isc[j] - voc[j] / rsh[j]) * np.exp(-voc[j] /
                                                        nnsvth[j])

            # initial estimate of rs from dI/dV near Voc
            # [5] Step 3a; [6] Step 3c
            [didv, d2id2v] = _numdiff(volt, curr)
            t3 = volt > .5 * voc[j]
            t4 = volt < .9 * voc[j]
            tmp = -rsh[j] * didv - 1.
            with np.errstate(invalid="ignore"):  # expect nan in didv
                v = np.logical_and.reduce(np.array([t3, t4, ~np.isnan(tmp),
                                                    np.greater(tmp, 0)]))
            if np.any(v):
                vtrs = (nnsvth[j] / isc[j] * (
                    np.log(tmp[v] * nnsvth[j] / (rsh[j] * io[j]))
                    - volt[v] / nnsvth[j]))
                rs[j] = np.mean(vtrs[vtrs > 0], axis=0)
            else:
                rs[j] = 0.

            # Initial estimate of Iph, evaluate the single diode model at
            # Isc [5] Step 3a; [6] Step 3d
            iph[j] = isc[j] + io[j] * np.expm1(isc[j] / nnsvth[j]) \
                + isc[j] * rs[j] / rsh[j]

        else:
            io[j] = np.nan
            rs[j] = np.nan
            iph[j] = np.nan

        # Filter IV curves for good initial values
        # [5] Step 3b
        u = _filter_params(ee, isc, io, rs, rsh)

        # [5] Step 3c
        # Refine Io to match Voc
        io[u] = _update_io(voc[u], iph[u], io[u], rs[u], rsh[u], nnsvth[u])

        # parameters [6], Step 3c
        # Calculate Iph to be consistent with Isc and current values of other
        iph = isc + io * np.expm1(rs * isc / nnsvth) + isc * rs / rsh

    return iph, io, rs, u


def _update_iv_params(voc, isc, vmp, imp, ee, iph, io, rs, rsh, nnsvth, u,
                      maxiter, eps1):
    # Refine Rsh, Rs, Io and Iph in that order.
    # Helper function for fit_<model>_sandia.
    counter = 1.  # counter variable for parameter updating while loop,
    # counts iterations
    prevconvergeparams = {}
    prevconvergeparams['state'] = 0.0

    not_converged = np.array([True])

    while not_converged.any() and counter <= maxiter:
        # update rsh to match max power point using a fixed point method.
        rsh[u] = _update_rsh_fixed_pt(vmp[u], imp[u], iph[u], io[u], rs[u],
                                      rsh[u], nnsvth[u])

        # Calculate Rs to be consistent with Rsh and maximum power point
        _, phi = _calc_theta_phi_exact(vmp[u], imp[u], iph[u], io[u],
                                       rs[u], rsh[u], nnsvth[u])
        rs[u] = (iph[u] + io[u] - imp[u]) * rsh[u] / imp[u] - \
            nnsvth[u] * phi / imp[u] - vmp[u] / imp[u]

        # Update filter for good parameters
        u = _filter_params(ee, isc, io, rs, rsh)

        # Update value for io to match voc
        io[u] = _update_io(voc[u], iph[u], io[u], rs[u], rsh[u], nnsvth[u])

        # Calculate Iph to be consistent with Isc and other parameters
        iph = isc + io * np.expm1(rs * isc / nnsvth) + isc * rs / rsh

        # update filter for good parameters
        u = _filter_params(ee, isc, io, rs, rsh)

        # compute the IV curve from the current parameter values
        result = singlediode(iph[u], io[u], rs[u], rsh[u], nnsvth[u])

        # check convergence criteria
        # [5] Step 3d
        convergeparams = _check_converge(
            prevconvergeparams, result, vmp[u], imp[u], counter)

        prevconvergeparams = convergeparams
        counter += 1.
        t5 = prevconvergeparams['vmperrmeanchange'] >= eps1
        t6 = prevconvergeparams['imperrmeanchange'] >= eps1
        t7 = prevconvergeparams['pmperrmeanchange'] >= eps1
        t8 = prevconvergeparams['vmperrstdchange'] >= eps1
        t9 = prevconvergeparams['imperrstdchange'] >= eps1
        t10 = prevconvergeparams['pmperrstdchange'] >= eps1
        t11 = prevconvergeparams['vmperrabsmaxchange'] >= eps1
        t12 = prevconvergeparams['imperrabsmaxchange'] >= eps1
        t13 = prevconvergeparams['pmperrabsmaxchange'] >= eps1
        not_converged = np.logical_or.reduce(np.array([t5, t6, t7, t8, t9,
                                                       t10, t11, t12, t13]))

    return iph, io, rs, rsh, u


def _extract_sdm_params(ee, tc, iph, io, rs, rsh, n, u, specs, const,
                        model):
    # Get single diode model parameters from five parameters iph, io, rs, rsh
    # and n vs. effective irradiance and temperature
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            'Parameter extraction using Sandia method requires statsmodels')

    tck = tc + 273.15
    tok = const['T0'] + 273.15  # convert to to K

    params = {}

    if model == 'pvsyst':
        # Estimate I_o_ref and EgRef
        x_for_io = const['q'] / const['k'] * (1. / tok - 1. / tck[u]) / n[u]

        # Estimate R_sh_0, R_sh_ref and R_sh_exp
        # Initial guesses. R_sh_0 is value at ee=0.
        nans = np.isnan(rsh)
        if any(ee < 400):
            grsh0 = np.mean(rsh[np.logical_and(~nans, ee < 400)])
        else:
            grsh0 = np.max(rsh)
        # Rsh_ref is value at Ee = 1000
        if any(ee > 400):
            grshref = np.mean(rsh[np.logical_and(~nans, ee > 400)])
        else:
            grshref = np.min(rsh)
        # PVsyst default for Rshexp is 5.5
        R_sh_exp = 5.5

        # Find parameters for Rsh equation

        def fun_rsh(x, rshexp, ee, e0, rsh):
            tf = (
                np.log10(_pvsyst_Rsh(ee, x[1], x[0], R_sh_exp, e0))
                - np.log10(rsh)
            )
            return tf

        x0 = np.array([grsh0, grshref])
        beta = optimize.least_squares(
            fun_rsh, x0, args=(R_sh_exp, ee[u], const['E0'], rsh[u]),
            bounds=np.array([[1., 1.], [1.e7, 1.e6]]), verbose=2)
        # Extract PVsyst parameter values
        R_sh_0 = beta.x[0]
        R_sh_ref = beta.x[1]

        # parameters unique to PVsyst
        params['R_sh_0'] = R_sh_0
        params['R_sh_exp'] = R_sh_exp

    elif model == 'desoto':
        dEgdT = -0.0002677
        x_for_io = const['q'] / const['k'] * (
            1. / tok - 1. / tck[u] + dEgdT * (tc[u] - const['T0']) / tck[u])

        # Estimate R_sh_ref
        nans = np.isnan(rsh)
        x = const['E0'] / ee[np.logical_and(u, ee > 400, ~nans)]
        y = rsh[np.logical_and(u, ee > 400, ~nans)]
        new_x = sm.add_constant(x)
        beta = sm.RLM(y, new_x).fit()
        R_sh_ref = beta.params[1]

        params['dEgdT'] = dEgdT

    # Estimate I_o_ref and EgRef
    y = np.log(io[u]) - 3. * np.log(tck[u] / tok)
    new_x = sm.add_constant(x_for_io)
    res = sm.RLM(y, new_x).fit()
    beta = res.params
    I_o_ref = np.exp(beta[0])
    EgRef = beta[1]

    # Estimate I_L_ref
    x = tc[u] - const['T0']
    y = iph[u] * (const['E0'] / ee[u])
    # average over non-NaN values of Y and X
    nans = np.isnan(y - specs['alpha_sc'] * x)
    I_L_ref = np.mean(y[~nans] - specs['alpha_sc'] * x[~nans])

    # Estimate R_s
    nans = np.isnan(rs)
    R_s = np.mean(rs[np.logical_and(u, ee > 400, ~nans)])

    params['I_L_ref'] = I_L_ref
    params['I_o_ref'] = I_o_ref
    params['EgRef'] = EgRef
    params['R_sh_ref'] = R_sh_ref
    params['R_s'] = R_s
    # save values for each IV curve
    params['iph'] = iph
    params['io'] = io
    params['rsh'] = rsh
    params['rs'] = rs
    params['u'] = u

    return params


def _update_io(voc, iph, io, rs, rsh, nnsvth):
    """
    Adjusts Io to match Voc using other parameter values.

    Helper function for fit_pvsyst_sandia, fit_desoto_sandia

    Description
    -----------
    Io is updated iteratively 10 times or until successive
    values are less than 0.000001 % different. The updating is similar to
    Newton's method.

    Parameters
    ----------
    voc: a numpy array of length N of values for Voc (V)
    iph: a numpy array of length N of values for lighbt current IL (A)
    io: a numpy array of length N of initial values for Io (A)
    rs: a numpy array of length N of values for the series resistance (ohm)
    rsh: a numpy array of length N of values for the shunt resistance (ohm)
    nnsvth: a numpy array of length N of values for the diode factor x thermal
            voltage for the module, equal to Ns (number of cells in series) x
            Vth (thermal voltage per cell).

    Returns
    -------
    new_io - a numpy array of length N of updated values for io

    References
    ----------
    .. [1] PVLib MATLAB https://github.com/sandialabs/MATLAB_PV_LIB
    .. [2] C. Hansen, Parameter Estimation for Single Diode Models of
       Photovoltaic Modules, Sandia National Laboratories Report SAND2015-2065
    .. [3] C. Hansen, Estimation of Parameters for Single Diode Models using
       Measured IV Curves, Proc. of the 39th IEEE PVSC, June 2013.
    """

    eps = 1e-6
    niter = 10
    k = 1
    maxerr = 1

    tio = io  # Current Estimate of Io

    while maxerr > eps and k < niter:
        # Predict Voc
        pvoc = v_from_i(0., iph, tio, rs, rsh, nnsvth)

        # Difference in Voc
        dvoc = pvoc - voc

        # Update Io
        with np.errstate(invalid="ignore", divide="ignore"):
            new_io = tio * (1. + (2. * dvoc) / (2. * nnsvth - dvoc))
            # Calculate Maximum Percent Difference
            maxerr = np.max(np.abs(new_io - tio) / tio) * 100.

        tio = new_io
        k += 1.

    return new_io


def _filter_params(ee, isc, io, rs, rsh):
    # Function _filter_params identifies bad parameter sets. A bad set contains
    # Nan, non-positive or imaginary values for parameters; Rs > Rsh; or data
    # where effective irradiance Ee differs by more than 5% from a linear fit
    # to Isc vs. Ee

    badrsh = np.logical_or(rsh < 0., np.isnan(rsh))
    negrs = rs < 0.
    badrs = np.logical_or(rs > rsh, np.isnan(rs))
    imagrs = ~(np.isreal(rs))
    badio = np.logical_or(np.logical_or(~(np.isreal(rs)), io <= 0),
                          np.isnan(io))
    goodr = np.logical_and(~badrsh, ~imagrs)
    goodr = np.logical_and(goodr, ~negrs)
    goodr = np.logical_and(goodr, ~badrs)
    goodr = np.logical_and(goodr, ~badio)

    matrix = np.vstack((ee / 1000., np.zeros(len(ee)))).T
    eff = np.linalg.lstsq(matrix, isc, rcond=None)[0][0]
    pisc = eff * ee / 1000
    pisc_error = np.abs(pisc - isc) / isc
    # check for departure from linear relation between Isc and Ee
    badiph = pisc_error > .05

    u = np.logical_and(goodr, ~badiph)
    return u


def _check_converge(prevparams, result, vmp, imp, i):
    """
    Function _check_converge computes convergence metrics for all IV curves.

    Helper function for fit_pvsyst_sandia, fit_desoto_sandia

    Parameters
    ----------
    prevparams: Convergence Parameters from the previous Iteration (used to
                determine Percent Change in values between iterations)
    result: performacne parameters of the (predicted) single diode fitting,
            which includes Voc, Vmp, Imp, Pmp and Isc
    vmp: measured values for each IV curve
    imp: measured values for each IV curve
    i: Index of current iteration in cec_parameter_estimation

    Returns
    -------
    convergeparam: dict containing the following for Imp, Vmp and Pmp:
        - maximum percent difference between measured and modeled values
        - minimum percent difference between measured and modeled values
        - maximum absolute percent difference between measured and modeled
          values
        - mean percent difference between measured and modeled values
        - standard deviation of percent difference between measured and modeled
          values
        - absolute difference for previous and current values of maximum
          absolute percent difference (measured vs. modeled)
        - absolute difference for previous and current values of mean percent
          difference (measured vs. modeled)
        - absolute difference for previous and current values of standard
          deviation of percent difference (measured vs. modeled)
    """

    convergeparam = {}

    imperror = (result['i_mp'] - imp) / imp * 100.
    vmperror = (result['v_mp'] - vmp) / vmp * 100.
    pmperror = (result['p_mp'] - (imp * vmp)) / (imp * vmp) * 100.

    convergeparam['imperrmax'] = max(imperror)  # max of the error in Imp
    convergeparam['imperrmin'] = min(imperror)  # min of the error in Imp
    # max of the absolute error in Imp
    convergeparam['imperrabsmax'] = max(abs(imperror))
    # mean of the error in Imp
    convergeparam['imperrmean'] = np.mean(imperror, axis=0)
    # std of the error in Imp
    convergeparam['imperrstd'] = np.std(imperror, axis=0, ddof=1)

    convergeparam['vmperrmax'] = max(vmperror)  # max of the error in Vmp
    convergeparam['vmperrmin'] = min(vmperror)  # min of the error in Vmp
    # max of the absolute error in Vmp
    convergeparam['vmperrabsmax'] = max(abs(vmperror))
    # mean of the error in Vmp
    convergeparam['vmperrmean'] = np.mean(vmperror, axis=0)
    # std of the error in Vmp
    convergeparam['vmperrstd'] = np.std(vmperror, axis=0, ddof=1)

    convergeparam['pmperrmax'] = max(pmperror)  # max of the error in Pmp
    convergeparam['pmperrmin'] = min(pmperror)  # min of the error in Pmp
    # max of the abs err. in Pmp
    convergeparam['pmperrabsmax'] = max(abs(pmperror))
    # mean error in Pmp
    convergeparam['pmperrmean'] = np.mean(pmperror, axis=0)
    # std error Pmp
    convergeparam['pmperrstd'] = np.std(pmperror, axis=0, ddof=1)

    if prevparams['state'] != 0.0:
        convergeparam['imperrstdchange'] = np.abs(
            convergeparam['imperrstd'] / prevparams['imperrstd'] - 1.)
        convergeparam['vmperrstdchange'] = np.abs(
            convergeparam['vmperrstd'] / prevparams['vmperrstd'] - 1.)
        convergeparam['pmperrstdchange'] = np.abs(
            convergeparam['pmperrstd'] / prevparams['pmperrstd'] - 1.)
        convergeparam['imperrmeanchange'] = np.abs(
            convergeparam['imperrmean'] / prevparams['imperrmean'] - 1.)
        convergeparam['vmperrmeanchange'] = np.abs(
            convergeparam['vmperrmean'] / prevparams['vmperrmean'] - 1.)
        convergeparam['pmperrmeanchange'] = np.abs(
            convergeparam['pmperrmean'] / prevparams['pmperrmean'] - 1.)
        convergeparam['imperrabsmaxchange'] = np.abs(
            convergeparam['imperrabsmax'] / prevparams['imperrabsmax'] - 1.)
        convergeparam['vmperrabsmaxchange'] = np.abs(
            convergeparam['vmperrabsmax'] / prevparams['vmperrabsmax'] - 1.)
        convergeparam['pmperrabsmaxchange'] = np.abs(
            convergeparam['pmperrabsmax'] / prevparams['pmperrabsmax'] - 1.)
        convergeparam['state'] = 1.0
    else:
        convergeparam['imperrstdchange'] = float("Inf")
        convergeparam['vmperrstdchange'] = float("Inf")
        convergeparam['pmperrstdchange'] = float("Inf")
        convergeparam['imperrmeanchange'] = float("Inf")
        convergeparam['vmperrmeanchange'] = float("Inf")
        convergeparam['pmperrmeanchange'] = float("Inf")
        convergeparam['imperrabsmaxchange'] = float("Inf")
        convergeparam['vmperrabsmaxchange'] = float("Inf")
        convergeparam['pmperrabsmaxchange'] = float("Inf")
        convergeparam['state'] = 1.
    return convergeparam


def _update_rsh_fixed_pt(vmp, imp, iph, io, rs, rsh, nnsvth):
    """
    Adjust Rsh to match Vmp using other parameter values

    Helper function for fit_pvsyst_sandia, fit_desoto_sandia

    Description
    -----------
    Rsh is updated iteratively using a fixed point expression
    obtained from combining Vmp = Vmp(Imp) (using the analytic solution to the
    single diode equation) and dP / dI = 0 at Imp. 500 iterations are performed
    because convergence can be very slow.

    Parameters
    ----------
    vmp: a numpy array of length N of values for Vmp (V)
    imp: a numpy array of length N of values for Imp (A)
    iph: a numpy array of length N of values for light current IL (A)
    io: a numpy array of length N of values for Io (A)
    rs: a numpy array of length N of values for series resistance (ohm)
    rsh: a numpy array of length N of initial values for shunt resistance (ohm)
    nnsvth: a numpy array length N of values for the diode factor x thermal
            voltage for the module, equal to Ns (number of cells in series) x
            Vth (thermal voltage per cell).

    Returns
    -------
    numpy array of length N of updated values for Rsh

    References
    ----------
    .. [1] PVLib for MATLAB https://github.com/sandialabs/MATLAB_PV_LIB
    .. [2] C. Hansen, Parameter Estimation for Single Diode Models of
       Photovoltaic Modules, Sandia National Laboratories Report SAND2015-2065
    """
    niter = 500
    x1 = rsh

    for i in range(niter):
        _, z = _calc_theta_phi_exact(vmp, imp, iph, io, rs, x1, nnsvth)
        with np.errstate(divide="ignore"):
            next_x1 = (1 + z) / z * ((iph + io) * x1 / imp - nnsvth * z / imp
                                     - 2 * vmp / imp)
        x1 = next_x1

    return x1


def _calc_theta_phi_exact(vmp, imp, iph, io, rs, rsh, nnsvth):
    """
    _calc_theta_phi_exact computes Lambert W values appearing in the analytic
    solutions to the single diode equation for the max power point.

    Helper function for fit_pvsyst_sandia

    Parameters
    ----------
    vmp: a numpy array of length N of values for Vmp (V)
    imp: a numpy array of length N of values for Imp (A)
    iph: a numpy array of length N of values for the light current IL (A)
    io: a numpy array of length N of values for Io (A)
    rs: a numpy array of length N of values for the series resistance (ohm)
    rsh: a numpy array of length N of values for the shunt resistance (ohm)
    nnsvth: a numpy array of length N of values for the diode factor x
            thermal voltage for the module, equal to Ns
            (number of cells in series) x Vth
            (thermal voltage per cell).

    Returns
    -------
    theta: a numpy array of values for the Lamber W function for solving
           I = I(V)
    phi: a numpy array of values for the Lambert W function for solving
         V = V(I)

    Notes
    -----
    _calc_theta_phi_exact calculates values for the Lambert W function which
    are used in the analytic solutions for the single diode equation at the
    maximum power point. For V=V(I),
    phi = W(Io*Rsh/n*Vth * exp((IL + Io - Imp)*Rsh/n*Vth)). For I=I(V),
    theta = W(Rs*Io/n*Vth *
    Rsh/ (Rsh+Rs) * exp(Rsh/ (Rsh+Rs)*((Rs(IL+Io) + V)/n*Vth))

    References
    ----------
    .. [1] PVL MATLAB 2065 https://github.com/sandialabs/MATLAB_PV_LIB
    .. [2] C. Hansen, Parameter Estimation for Single Diode Models of
       Photovoltaic Modules, Sandia National Laboratories Report SAND2015-2065
    .. [3] A. Jain, A. Kapoor, "Exact analytical solutions of the parameters of
       real solar cells using Lambert W-function", Solar Energy Materials and
       Solar Cells, 81 (2004) 269-277.
    """
    # handle singleton inputs
    vmp = np.asarray(vmp)
    imp = np.asarray(imp)
    iph = np.asarray(iph)
    io = np.asarray(io)
    rs = np.asarray(rs)
    rsh = np.asarray(rsh)
    nnsvth = np.asarray(nnsvth)

    # Argument for Lambert W function involved in V = V(I) [2] Eq. 12; [3]
    # Eq. 3
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        argw = np.where(
            nnsvth == 0,
            np.nan,
            rsh * io / nnsvth * np.exp(rsh * (iph + io - imp) / nnsvth))
        phi = np.where(argw > 0, lambertw(argw).real, np.nan)

    # NaN where argw overflows. Switch to log space to evaluate
    u = np.isinf(argw)
    if np.any(u):
        logargw = (
            np.log(rsh[u]) + np.log(io[u]) - np.log(nnsvth[u])
            + rsh[u] * (iph[u] + io[u] - imp[u]) / nnsvth[u])
        # Three iterations of Newton-Raphson method to solve w+log(w)=logargW.
        # The initial guess is w=logargW. Where direct evaluation (above)
        # results in NaN from overflow, 3 iterations of Newton's method gives
        # approximately 8 digits of precision.
        x = logargw
        for i in range(3):
            x *= ((1. - np.log(x) + logargw) / (1. + x))
        phi[u] = x
    phi = np.transpose(phi)

    # Argument for Lambert W function involved in I = I(V) [2] Eq. 11; [3]
    # E1. 2
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        argw = np.where(
            nnsvth == 0,
            np.nan,
            rsh / (rsh + rs) * rs * io / nnsvth * np.exp(
                rsh / (rsh + rs) * (rs * (iph + io) + vmp) / nnsvth))
        theta = np.where(argw > 0, lambertw(argw).real, np.nan)

    # NaN where argw overflows. Switch to log space to evaluate
    u = np.isinf(argw)
    if np.any(u):
        with np.errstate(divide="ignore"):
            logargw = (
                np.log(rsh[u]) - np.log(rsh[u] + rs[u]) + np.log(rs[u])
                + np.log(io[u]) - np.log(nnsvth[u])
                + (rsh[u] / (rsh[u] + rs[u]))
                * (rs[u] * (iph[u] + io[u]) + vmp[u]) / nnsvth[u])
        # Three iterations of Newton-Raphson method to solve w+log(w)=logargW.
        # The initial guess is w=logargW. Where direct evaluation (above)
        # results in NaN from overflow, 3 iterations of Newton's method gives
        # approximately 8 digits of precision.
        x = logargw
        for i in range(3):
            x *= ((1. - np.log(x) + logargw) / (1. + x))
        theta[u] = x
    theta = np.transpose(theta)

    return theta, phi
