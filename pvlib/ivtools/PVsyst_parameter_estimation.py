import logging
import warnings
import numpy as np
from collections import OrderedDict
from pvlib.ivtools.est_single_diode_param import estimate_parameters
from pvlib.ivtools.update_io_known_n import update_io_known_n
from pvlib.ivtools.update_rsh_fixed_pt import update_rsh_fixed_pt
from pvlib.ivtools.calc_theta_phi_exact import calc_theta_phi_exact
from pvlib.pvsystem import singlediode

# optional imports used by pvsyst_parameter_estimation
# from scipy import optimize
# import statsmodels.api as sm
warnings.warn('pvsyst_parameter_estimation requires scipy and statsmodels')

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def numdiff(x, f):
    """
    NUMDIFF computes first and second order derivative using possibly unequally
    spaced data.

    Parameters
    ----------
    x : numeric
        a numpy array of values of x
    f : numeric
        a numpy array of values of the function f for which derivatives are to
        be computed. Must be the same length as x.

    Returns
    -------
    df : numeric
        a numpy array of len(x) containing the first derivative of f at each
        point x except at the first 2 and last 2 points
    df2 : numeric
        a numpy array of len(x) containing the second derivative of f at each
        point x except at the first 2 and last 2 points.

    Description
    -----------
    numdiff computes first and second order derivatives using a 5th order
    formula that accounts for possibly unequally spaced data. Because a 5th
    order centered difference formula is used, numdiff returns NaNs for the
    first 2 and last 2 points in the input vector for x.

    References
    ----------
    [1] PVLib MATLAB
    [2] M. K. Bowen, R. Smith, "Derivative formulae and errors for
        non-uniformly spaced points", Proceedings of the Royal Society A, vol.
        461 pp 1975 - 1997, July 2005. DOI: 10.1098/rpsa.2004.1430
    """

    n = len(f)

    df = np.zeros(n)
    df2 = np.zeros(n)

    # first two points are special
    df[:2] = float("Nan")
    df2[:2] = float("Nan")

    # Last two points are special
    df[-2:] = float("Nan")
    df2[-2:] = float("Nan")

    # Rest of points. Take reference point to be the middle of each group of 5
    # points. Calculate displacements
    ff = np.vstack((f[:-4], f[1:-3], f[2:-2], f[3:-1], f[4:])).T

    a0 = (np.vstack((x[:-4], x[1:-3], x[2:-2], x[3:-1], x[4:])).T
          - np.tile(x[2:-2], [5, 1]).T)

    u1 = np.zeros(a0.shape)
    left = np.zeros(a0.shape)
    u2 = np.zeros(a0.shape)

    u1[:, 0] = (
        a0[:, 1] * a0[:, 2] * a0[:, 3] + a0[:, 1] * a0[:, 2] * a0[:, 4]
        + a0[:, 1] * a0[:, 3] * a0[:, 4] + a0[:, 2] * a0[:, 3] * a0[:, 4])
    u1[:, 1] = (
        a0[:, 0] * a0[:, 2] * a0[:, 3] + a0[:, 0] * a0[:, 2] * a0[:, 4]
        + a0[:, 0] * a0[:, 3] * a0[:, 4] + a0[:, 2] * a0[:, 3] * a0[:, 4])
    u1[:, 2] = (
        a0[:, 0] * a0[:, 1] * a0[:, 3] + a0[:, 0] * a0[:, 1] * a0[:, 4]
        + a0[:, 0] * a0[:, 3] * a0[:, 4] + a0[:, 1] * a0[:, 3] * a0[:, 4])
    u1[:, 3] = (
        a0[:, 0] * a0[:, 1] * a0[:, 2] + a0[:, 0] * a0[:, 1] * a0[:, 4]
        + a0[:, 0] * a0[:, 2] * a0[:, 4] + a0[:, 1] * a0[:, 2] * a0[:, 4])
    u1[:, 4] = (
        a0[:, 0] * a0[:, 1] * a0[:, 2] + a0[:, 0] * a0[:, 1] * a0[:, 3]
        + a0[:, 0] * a0[:, 2] * a0[:, 3] + a0[:, 1] * a0[:, 2] * a0[:, 3])

    left[:, 0] = (a0[:, 0] - a0[:, 1]) * (a0[:, 0] - a0[:, 2]) * \
        (a0[:, 0] - a0[:, 3]) * (a0[:, 0] - a0[:, 4])
    left[:, 1] = (a0[:, 1] - a0[:, 0]) * (a0[:, 1] - a0[:, 2]) * \
        (a0[:, 1] - a0[:, 3]) * (a0[:, 1] - a0[:, 4])
    left[:, 2] = (a0[:, 2] - a0[:, 0]) * (a0[:, 2] - a0[:, 1]) * \
        (a0[:, 2] - a0[:, 3]) * (a0[:, 2] - a0[:, 4])
    left[:, 3] = (a0[:, 3] - a0[:, 0]) * (a0[:, 3] - a0[:, 1]) * \
        (a0[:, 3] - a0[:, 2]) * (a0[:, 3] - a0[:, 4])
    left[:, 4] = (a0[:, 4] - a0[:, 0]) * (a0[:, 4] - a0[:, 1]) * \
        (a0[:, 4] - a0[:, 2]) * (a0[:, 4] - a0[:, 3])

    df[2:-2] = np.sum(-(u1 / left) * ff, axis=1)

    # second derivative
    u2[:, 0] = (
        a0[:, 1] * a0[:, 2] + a0[:, 1] * a0[:, 3] + a0[:, 1] * a0[:, 4]
        + a0[:, 2] * a0[:, 3] + a0[:, 2] * a0[:, 4] + a0[:, 3] * a0[:, 4])
    u2[:, 1] = (
        a0[:, 0] * a0[:, 2] + a0[:, 0] * a0[:, 3] + a0[:, 0] * a0[:, 4]
        + a0[:, 2] * a0[:, 3] + a0[:, 2] * a0[:, 4] + a0[:, 3] * a0[:, 4])
    u2[:, 2] = (
        a0[:, 0] * a0[:, 1] + a0[:, 0] * a0[:, 3] + a0[:, 0] * a0[:, 4]
        + a0[:, 1] * a0[:, 3] + a0[:, 1] * a0[:, 3] + a0[:, 3] * a0[:, 4])
    u2[:, 3] = (
        a0[:, 0] * a0[:, 1] + a0[:, 0] * a0[:, 2] + a0[:, 0] * a0[:, 4]
        + a0[:, 1] * a0[:, 2] + a0[:, 1] * a0[:, 4] + a0[:, 2] * a0[:, 4])
    u2[:, 4] = (
        a0[:, 0] * a0[:, 1] + a0[:, 0] * a0[:, 2] + a0[:, 0] * a0[:, 3]
        + a0[:, 1] * a0[:, 2] + a0[:, 1] * a0[:, 4] + a0[:, 2] * a0[:, 3])

    df2[2:-2] = 2. * np.sum(u2 * ff, axis=1)
    return df, df2


def rectify_iv_curve(ti, tv, voc, isc):
    """
    ``rectify_IV_curve`` ensures that Isc and Voc are included in a IV curve
    and removes duplicate voltage and current points.

    Parameters
    ----------
    ti : numeric
        test currents [A]
    tv : numeric
        test voltages [V]
    voc : numeric
        open circuit voltages [V]
    isc : numeric
        short circuit current [A]

    Returns
    -------
    current (I [A]), voltage (V [V])

    Description
    -----------
    ``rectify_IV_curve`` ensures that the IV curve data

    * increases in voltage
    * contain no negative current or voltage values
    * have the first data point as (0, Isc)
    * have the last data point as (Voc, 0)
    * contain no duplicate voltage values. Where voltage values are
      repeated, a single data point is substituted with current equal to
      the average of current at each repeated voltage.
    """
    # Filter out negative voltage and current values
    data_filter = []
    for i, v in zip(ti, tv):
        if i < 0:
            continue
        if v > voc:
            continue
        if v < 0:
            continue
        if np.isnan(i) or np.isnan(v):
            continue
        data_filter.append((i, v))

    current = np.array([isc])
    voltage = np.array([0.])

    for i, v in data_filter:
        current = np.append(current, i)
        voltage = np.append(voltage, v)

    # Add in Voc and Isc
    current = np.append(current, 0.)
    voltage = np.append(voltage, voc)

    # Remove duplicate Voltage and Current points
    u, index, inverse = np.unique(voltage, return_index=True,
                                  return_inverse=True)
    if len(u) != len(voltage):
        v = []
        for i in u:
            fil = []
            for n, j in enumerate(voltage):
                if i == j:
                    fil.append(n)
            t = current[fil]
            v.append(np.average(t))
        voltage = u
        current = np.array(v)
    return current, voltage


def estrsh(x, rshexp, g, go):
    # computes rsh for PVsyst model where the parameters are in vector xL
    # x[0] = Rsh0
    # x[1] = Rshref

    rsho = x[0]
    rshref = x[1]

    rshb = np.maximum(
        (rshref - rsho * np.exp(-rshexp)) / (1. - np.exp(-rshexp)), 0.)
    prsh = rshb + (rsho - rshb) * np.exp(-rshexp * g / go)
    return prsh


def filter_params(io, rsh, rs, ee, isc):
    # Function filter_params identifies bad parameter sets. A bad set contains
    # Nan, non-positive or imaginary values for parameters; Rs > Rsh; or data
    # where effective irradiance Ee differs by more than 5% from a linear fit
    # to Isc vs. Ee

    badrsh = np.logical_or(rsh < 0., np.isnan(rsh))
    negrs = rs < 0.
    badrs = np.logical_or(rs > rsh, np.isnan(rs))
    imagrs = ~(np.isreal(rs))
    badio = np.logical_or(~(np.isreal(rs)), io <= 0)
    goodr = np.logical_and(~badrsh, ~imagrs)
    goodr = np.logical_and(goodr, ~negrs)
    goodr = np.logical_and(goodr, ~badrs)
    goodr = np.logical_and(goodr, ~badio)

    matrix = np.vstack((ee / 1000., np.zeros(len(ee)))).T
    eff = np.linalg.lstsq(matrix, isc)[0][0]
    pisc = eff * ee / 1000
    pisc_error = np.abs(pisc - isc) / isc
    # check for departure from linear relation between Isc and Ee
    badiph = pisc_error > .05

    u = np.logical_and(goodr, ~badiph)
    return u


def check_converge(prevparams, result, vmp, imp, i):
    """
    Function check_converge computes convergence metrics for all IV curves.

    Parameters
    ----------
    prevparams: Convergence Parameters from the previous Iteration (used to
                determine Percent Change in values between iterations)
    result: performacne paramters of the (predicted) single diode fitting,
            which includes Voc, Vmp, Imp, Pmp and Isc
    vmp: measured values for each IV curve
    imp: measured values for each IV curve
    i: Index of current iteration in cec_parameter_estimation

    Returns
    -------
    convergeparam: a class containing the following for Imp, Vmp and Pmp.
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
    convergeparam = OrderedDict()

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

    # TODO: use abs(x/y - 1) instead of abs(x-y)/y
    if prevparams['state'] != 0.0:
        convergeparam['imperrstdchange'] = np.abs(
            (convergeparam['imperrstd'] - prevparams['imperrstd'])
            / prevparams['imperrstd'])
        convergeparam['vmperrstdchange'] = np.abs(
            (convergeparam['vmperrstd'] - prevparams['vmperrstd'])
            / prevparams['vmperrstd'])
        convergeparam['pmperrstdchange'] = np.abs(
            (convergeparam['pmperrstd'] - prevparams['pmperrstd'])
            / prevparams['pmperrstd'])
        convergeparam['imperrmeanchange'] = np.abs(
            (convergeparam['imperrmean'] - prevparams['imperrmean'])
            / prevparams['imperrmean'])
        convergeparam['vmperrmeanchange'] = np.abs(
            (convergeparam['vmperrmean'] - prevparams['vmperrmean'])
            / prevparams['vmperrmean'])
        convergeparam['pmperrmeanchange'] = np.abs(
            (convergeparam['pmperrmean'] - prevparams['pmperrmean'])
            / prevparams['pmperrmean'])
        convergeparam['imperrabsmaxchange'] = np.abs(
            (convergeparam['imperrabsmax'] - prevparams['imperrabsmax'])
            / prevparams['imperrabsmax'])
        convergeparam['vmperrabsmaxchange'] = np.abs(
            (convergeparam['vmperrabsmax'] - prevparams['vmperrabsmax'])
            / prevparams['vmperrabsmax'])
        convergeparam['pmperrabsmaxchange'] = np.abs(
            (convergeparam['pmperrabsmax'] - prevparams['pmperrabsmax'])
            / prevparams['pmperrabsmax'])
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


const_default = OrderedDict()
const_default['E0'] = 1000.0
const_default['T0'] = 25.0
const_default['k'] = 1.38066e-23
const_default['q'] = 1.60218e-19


def fun_rsh(x, rshexp, ee, e0, rsh):
    tf = np.log10(estrsh(x, rshexp, ee, e0)) - np.log10(rsh)
    return tf


def pvsyst_parameter_estimation(ivcurves, specs, const=const_default,
                                maxiter=5, eps1=1.e-3):
    """
    pvsyst_parameter_estimation estimates parameters fro the PVsyst module
    performance model

    Parameters
    ----------
    ivcurves : a dict
        containing IV curve data in the following fields where j
    denotes the jth data set
        ivcurves['i'][j] - a numpy array of current (A) (same length as v)
        ivcurves['v'][j] - a numpy array of voltage (V) (same length as i)
        ivcurves['ee'][j] - effective irradiance (W / m^2), i.e., POA broadband
                            irradiance adjusted by solar spectrum modifier
        ivcurves['tc'][j] - cell temperature (C)
        ivcurves['isc'][j] - short circuit current of IV curve (A)
        ivcurves['voc'][j] - open circuit voltage of IV curve (V)
        ivcurves['imp'][j] - current at max power point of IV curve (A)
        ivcurves['vmp'][j] - voltage at max power point of IV curve (V)

    specs : a dict
        containing module-level values
        specs['ns'] - number of cells in series
        specs['aisc'] - temperature coefficeint of isc (A/C)

    const : an optional OrderedDict
        containing physical and other constants
        const['E0'] - effective irradiance at STC, normally 1000 W/m2
        constp['T0'] - cell temperature at STC, normally 25 C
        const['k'] - 1.38066E-23 J/K (Boltzmann's constant)
        const['q'] - 1.60218E-19 Coulomb (elementary charge)

    maxiter : an optional numpy array
        input that sets the maximum number of
        iterations for the parameter updating part of the algorithm.
        Default value is 5.

    eps1: the desired tolerance for the IV curve fitting. The iterative
          parameter updating stops when absolute values of the percent change
          in mean, max and standard deviation of Imp, Vmp and Pmp between
          iterations are all less than eps1, or when the number of iterations
          exceeds maxiter. Default value is 1e-3 (.0001%).

    Returns
    -------
    pvsyst: a OrderedDict containing the model parameters
        pvsyst['IL_ref'] - light current (A) at STC
        pvsyst['Io_ref'] - dark current (A) at STC
        pvsyst['eG'] - effective band gap (eV) at STC
        pvsyst['Rsh_ref'] - shunt resistance (ohms) at STC
        pvsyst['Rsh0'] - shunt resistance (ohms) at zero irradiance
        pvsyst['Rshexp'] - exponential factor defining decrease in rsh with
                           increasing effective irradiance
        pvsyst['Rs_ref'] - series resistance (ohms) at STC
        pvsyst['gamma_ref'] - diode (ideality) factor at STC
        pvsyst['mugamma'] - temperature coefficient for diode (ideality) factor
        pvsyst['Iph'] - numpy array of values of light current Iph estimated
                        for each IV curve
        pvsyst['Io'] - numpy array of values of dark current Io estimated for
                       each IV curve
        pvsyst['Rsh'] - numpy array of values of shunt resistance Rsh estimated
                        for each IV curve
        pvsyst['Rs'] - numpy array of values of series resistance Rs estimated
                       for each IV curve
        pvsyst.u - filter indicating IV curves with parameter values deemed
                   reasonable by the private function filter_params

    Description
    -----------
    pvsyst_paramter_estimation estimates parameters for the PVsyst module
    performance model [2,3,4]. Estimation methods are documented in [5,6,7].

    References
    ----------
    [1] PVLib MATLAB
    [2] K. Sauer, T. Roessler, C. W. Hansen, Modeling the Irradiance and
        Temperature Dependence of Photovoltaic Modules in PVsyst, IEEE Journal
        of Photovoltaics v5(1), January 2015.
    [3] A. Mermoud, PV Modules modeling, Presentation at the 2nd PV Performance
        Modeling Workshop, Santa Clara, CA, May 2013
    [4] A. Mermoud, T. Lejeuene, Performance Assessment of a Simulation Model
        for PV modules of any available technology, 25th European Photovoltaic
        Solar Energy Conference, Valencia, Spain, Sept. 2010
    [5] C. Hansen, Estimating Parameters for the PVsyst Version 6 Photovoltaic
        Module Performance Model, Sandia National Laboratories Report
        SAND2015-8598
    [6] C. Hansen, Parameter Estimation for Single Diode Models of Photovoltaic
        Modules, Sandia National Laboratories Report SAND2015-2065
    [7] C. Hansen, Estimation of Parameters for Single Diode Models using
        Measured IV Curves, Proc. of the 39th IEEE PVSC, June 2013.
    """
    from scipy import optimize
    import statsmodels.api as sm
    ee = ivcurves['ee']
    tc = ivcurves['tc']
    tck = tc + 273.15
    isc = ivcurves['isc']
    voc = ivcurves['voc']
    imp = ivcurves['imp']
    vmp = ivcurves['vmp']

    # Cell Thermal Voltage
    vth = const['k'] / const['q'] * tck

    n = len(ivcurves['voc'])

    # Initial estimate of Rsh used to obtain the diode factor gamma0 and diode
    # temperature coefficient mugamma. Rsh is estimated using the co-content
    # integral method.

    pio = np.ones(n)
    piph = np.ones(n)
    prsh = np.ones(n)
    prs = np.ones(n)
    pn = np.ones(n)

    for j in range(n):
        current, voltage = rectify_iv_curve(ivcurves['i'][j], ivcurves['v'][j],
                                            voc[j], isc[j])
        # initial estimate of Rsh, from integral over voltage regression
        # [5] Step 3a; [6] Step 3a
        pio[j], piph[j], prs[j], prsh[j], pn[j] = \
            estimate_parameters(current, voltage, vth[j] * specs['ns'])

    # Estimate the diode factor gamma from Isc-Voc data. Method incorporates
    # temperature dependence by means of the equation for Io

    y = np.log(isc - voc / prsh) - 3. * np.log(tck / (const['T0'] + 273.15))
    x1 = const['q'] / const['k'] * (1. / (const['T0'] + 273.15) - 1. / tck)
    x2 = voc / (vth * specs['ns'])
    t0 = np.isnan(y)
    t1 = np.isnan(x1)
    t2 = np.isnan(x2)
    uu = np.logical_or(t0, t1)
    uu = np.logical_or(uu, t2)

    x = np.vstack((np.ones(len(x1[~uu])), x1[~uu], -x1[~uu] *
                   (tck[~uu] - (const['T0'] + 273.15)), x2[~uu],
                   -x2[~uu] * (tck[~uu] - (const['T0'] + 273.15)))).T
    alpha = np.linalg.lstsq(x, y[~uu])[0]

    gamma_ref = 1. / alpha[3]
    mugamma = alpha[4] / alpha[3] ** 2

    if np.isnan(gamma_ref) or np.isnan(mugamma) or not np.isreal(gamma_ref) \
            or not np.isreal(mugamma):
        badgamma = True
    else:
        badgamma = False

    pvsyst = OrderedDict()

    if ~badgamma:
        gamma = gamma_ref + mugamma * (tc - const['T0'])

        nnsvth = gamma * (vth * specs['ns'])

        # For each IV curve, sequentially determine initial values for Io, Rs,
        # and Iph [5] Step 3a; [6] Step 3

        io = np.ones(n)
        iph = np.ones(n)
        rs = np.ones(n)
        rsh = prsh

        for j in range(n):
            curr, volt = rectify_iv_curve(ivcurves['i'][j], ivcurves['v'][j],
                                          voc[j], isc[j])

            if rsh[j] > 0:
                # Initial estimate of Io, evaluate the single diode model at
                # voc and approximate Iph + Io = Isc [5] Step 3a; [6] Step 3b
                io[j] = (isc[j] - voc[j] / rsh[j]) * np.exp(-voc[j] /
                                                            nnsvth[j])

                # initial estimate of rs from dI/dV near Voc
                # [5] Step 3a; [6] Step 3c
                [didv, d2id2v] = numdiff(volt, curr)
                t3 = volt > .5 * voc[j]
                t4 = volt < .9 * voc[j]
                u = np.logical_and(t3, t4)
                tmp = -rsh[j] * didv - 1.
                v = np.logical_and(u, tmp > 0)
                if np.sum(v) > 0:
                    vtrs = (nnsvth[j] / isc[j] * (
                        np.log(tmp[v] * nnsvth[j] / (rsh[j] * io[j]))
                        - volt[v] / nnsvth[j]))
                    rs[j] = np.mean(vtrs[vtrs > 0], axis=0)
                else:
                    rs[j] = 0.

                # Initial estimate of Iph, evaluate the single diode model at
                # Isc [5] Step 3a; [6] Step 3d

                iph[j] = isc[j] - io[j] + io[j] * np.exp(isc[j] / nnsvth[j]) \
                    + isc[j] * rs[j] / rsh[j]
            else:
                io[j] = float("Nan")
                rs[j] = float("Nan")
                iph[j] = float("Nan")

        # Filter IV curves for good initial values
        LOGGER.debug('filtering params ... may take awhile')
        # [5] Step 3b
        u = filter_params(io, rsh, rs, ee, isc)

        # Refine Io to match Voc
        LOGGER.debug('Refine Io to match Voc')
        # [5] Step 3c
        tmpiph = iph
        tmpio = update_io_known_n(rsh[u], rs[u], nnsvth[u], io[u], tmpiph[u],
                                  voc[u])
        io[u] = tmpio

        # Calculate Iph to be consistent with Isc and current values of other
        LOGGER.debug('Calculate Iph to be consistent with Isc, ...')
        # parameters [6], Step 3c
        iph = isc - io + io * np.exp(rs * isc / nnsvth) + isc * rs / rsh

        # Refine Rsh, Rs, Io and Iph in that order.
        counter = 1.  # counter variable for parameter updating while loop,
        # counts iterations
        prevconvergeparams = OrderedDict()
        prevconvergeparams['state'] = 0.0

        t14 = np.array([True])

        while t14.any() and counter <= maxiter:
            # update rsh to match max power point using a fixed point method.
            tmprsh = update_rsh_fixed_pt(rsh[u], rs[u], io[u], iph[u],
                                         nnsvth[u], imp[u], vmp[u])

            rsh[u] = tmprsh

            # Calculate Rs to be consistent with Rsh and maximum power point
            LOGGER.debug('step %d: calculate Rs', counter)
            [a, phi] = calc_theta_phi_exact(imp[u], iph[u], vmp[u], io[u],
                                            nnsvth[u], rs[u], rsh[u])
            rs[u] = (iph[u] + io[u] - imp[u]) * rsh[u] / imp[u] - \
                nnsvth[u] * phi / imp[u] - vmp[u] / imp[u]

            # Update filter for good parameters
            u = filter_params(io, rsh, rs, ee, isc)

            # Update value for io to match voc
            LOGGER.debug('step %d: calculate dark/saturation current (Io)',
                         counter)
            tmpio = update_io_known_n(rsh[u], rs[u], nnsvth[u], io[u], iph[u],
                                      voc[u])
            io[u] = tmpio

            # Calculate Iph to be consistent with Isc and other parameters
            LOGGER.debug('step %d: calculate Iph', counter)
            iph = isc - io + io * np.exp(rs * isc / nnsvth) + isc * rs / rsh

            # update filter for good parameters
            u = filter_params(io, rsh, rs, ee, isc)

            # compute the IV curve from the current parameter values
            LOGGER.debug('step %d: compute IV curve', counter)
            result = singlediode(iph[u], io[u], rs[u], rsh[u], nnsvth[u])

            # check convergence criteria
            LOGGER.debug('step %d: checking convergence', counter)
            # [5] Step 3d
            convergeparams = check_converge(
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
            t14 = np.logical_or(t5, t6)
            t14 = np.logical_or(t14, t7)
            t14 = np.logical_or(t14, t8)
            t14 = np.logical_or(t14, t9)
            t14 = np.logical_or(t14, t10)
            t14 = np.logical_or(t14, t11)
            t14 = np.logical_or(t14, t12)
            t14 = np.logical_or(t14, t13)

        # Extract coefficients for auxillary equations
        # Estimate Io0 and eG
        tok = const['T0'] + 273.15  # convert to to K
        x = const['q'] / const['k'] * (1. / tok - 1. / tck[u]) / gamma[u]
        y = np.log(io[u]) - 3. * np.log(tck[u] / tok)
        new_x = sm.add_constant(x)
        res = sm.RLM(y, new_x).fit()
        beta = res.params
        io0 = np.exp(beta[0])
        eg = beta[1]

        # Estimate Iph0
        x = tc[u] - const['T0']
        y = iph[u] * (const['E0'] / ee[u])
        # average over non-NaN values of Y and X
        nans = np.isnan(y - specs['aisc'] * x)
        iph0 = np.mean(y[~nans] - specs['aisc'] * x[~nans])

        # Additional filter for Rsh and Rs; Restrict effective irradiance to be
        # greater than 400 W/m^2
        vfil = ee > 400

        # Estimate Rsh0, Rsh_ref and Rshexp

        # Initial Guesses. Rsh0 is value at Ee=0.
        nans = np.isnan(rsh)
        if any(ee < 400):
            grsh0 = np.mean(rsh[np.logical_and(~nans, ee < 400)])
        else:
            grsh0 = np.max(rsh)

        # Rsh_ref is value at Ee = 1000
        if any(vfil):
            grshref = np.mean(rsh[np.logical_and(~nans, vfil)])
        else:
            grshref = np.min(rsh)

        # PVsyst default for Rshexp is 5.5
        rshexp = 5.5

        # Here we use a nonlinear least squares technique. Lsqnonlin minimizes
        # the sum of squares of the objective function (here, tf).
        x0 = np.array([grsh0, grshref])
        beta = optimize.least_squares(
            fun_rsh, x0, args=(rshexp, ee[u], const['E0'], rsh[u]),
            bounds=np.array([[1., 1.], [1.e7, 1.e6]]), verbose=2)

        # Extract PVsyst parameter values
        rsh0 = beta.x[0]
        rshref = beta.x[1]

        # Estimate Rs0
        t15 = np.logical_and(u, vfil)
        rs0 = np.mean(rs[t15])

        # Save parameter estimates in output structure
        pvsyst['IL_ref'] = iph0
        pvsyst['Io_ref'] = io0
        pvsyst['eG'] = eg
        pvsyst['Rs_ref'] = rs0
        pvsyst['gamma_ref'] = gamma_ref
        pvsyst['mugamma'] = mugamma
        pvsyst['Iph'] = iph
        pvsyst['Io'] = io
        pvsyst['Rsh0'] = rsh0
        pvsyst['Rsh_ref'] = rshref
        pvsyst['Rshexp'] = rshexp
        pvsyst['Rs'] = rs
        pvsyst['Rsh'] = rsh
        pvsyst['Ns'] = specs['ns']
        pvsyst['u'] = u

    else:
        raise RuntimeError(
            "Failed to estimate the diode (ideality) factor parameter.")
    return pvsyst
