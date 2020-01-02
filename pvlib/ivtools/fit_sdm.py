"""
The ``fit_sdm`` module contains functions to fit single diode models.

Function names should follow the pattern "fit_" + name of model + "_" +
 fitting method.

"""

import numpy as np
from collections import OrderedDict

import logging

from pvlib.pvsystem import singlediode, v_from_i

from pvlib.ivtools.utility import constants, rectify_iv_curve, numdiff
from pvlib.ivtools.fit_sde import fit_cocontent


def fit_cec_sam(celltype, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc,
                gamma_pmp, cells_in_series, temp_ref=25):
    """
    Estimates parameters for the CEC single diode model (SDM) using the SAM
    SDK. Uses the method described in [Dobo12]_.

    Parameters
    ----------
    celltype : str
        Value is one of 'monoSi', 'multiSi', 'polySi', 'cis', 'cigs', 'cdte',
        'amorphous'
    v_mp : float
        Voltage at maximum power point [V]
    i_mp : float
        Current at maximum power point [A]
    v_oc : float
        Open circuit voltage [V]
    i_sc : float
        Short circuit current [A]
    alpha_sc : float
        Temperature coefficient of short circuit current [A/C]
    beta_voc : float
        Temperature coefficient of open circuit voltage [V/C]
    gamma_pmp : float
        Temperature coefficient of power at maximum point point [%/C]
    cells_in_series : int
        Number of cells in series
    temp_ref : float, default 25
        Reference temperature condition [C]

    Returns
    -------
    Tuple of the following elements:

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions
        [A]

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions
        [A]

    R_sh_ref : float
        The shunt resistance at reference conditions [ohm]

    R_s : float
        The series resistance at reference conditions [ohm]

    a_ref : float
        The product of the usual diode ideality factor ``n`` (unitless),
        number of cells in series ``Ns``, and cell thermal voltage at
        reference conditions [V]

    Adjust : float
        The adjustment to the temperature coefficient for short circuit
        current [%]

    Raises
    ------
        ImportError if NREL-PySAM is not installed.

        RuntimeError if parameter extraction is not successful.

    Notes
    -----
    Inputs ``v_mp``, ``v_oc``, ``i_mp`` and ``i_sc`` are assumed to be from a
    single IV curve at constant irradiance and cell temperature. Irradiance is
    not explicitly used by the fitting procedure. The irradiance level at which
    the input IV curve is determined and the specified cell temperature
    ``temp_ref`` are the reference conditions for the output parameters
    ``I_L_ref``, ``I_o_ref``, ``R_sh_ref``, ``R_s``, ``a_ref`` and ``Adjust``.

    References
    ----------
    .. [Dobo12] A. Dobos, "An Improved Coefficient Calculator for the
       California Energy Commission 6 Parameter Photovoltaic Module Model",
       Journal of Solar Energy Engineering, vol 134, 2012.
    """

    try:
        from PySAM import PySSC
    except ImportError:
        raise ImportError("Requires NREL's PySAM package at "
                          "https://pypi.org/project/NREL-PySAM/.")

    datadict = {'tech_model': '6parsolve', 'financial_model': 'none',
                'celltype': celltype, 'Vmp': v_mp,
                'Imp': i_mp, 'Voc': v_oc, 'Isc': i_sc, 'alpha_isc': alpha_sc,
                'beta_voc': beta_voc, 'gamma_pmp': gamma_pmp,
                'Nser': cells_in_series, 'Tref': temp_ref}

    result = PySSC.ssc_sim_from_dict(datadict)
    if result['cmod_success'] == 1:
        return tuple([result[k] for k in ['Il', 'Io', 'Rsh', 'Rs', 'a',
                      'Adj']])
    else:
        raise RuntimeError('Parameter estimation failed')


def fit_desoto(v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc,
               cells_in_series, EgRef=1.121, dEgdT=-0.0002677,
               temp_ref=25, irrad_ref=1000, root_kwargs={}):
    """
    Calculates the parameters for the De Soto single diode model [DeSo06]_
    using the procedure described in [Duff13]_. This procedure has the
    advantage of using common specifications given by manufacturers in the
    datasheets of PV modules.

    The solution is found using ``scipy.optimize.root``,
    with the corresponding default solver method 'hybr'.
    No restriction is put on the fit variables, i.e. series
    or shunt resistance could go negative. Nevertheless, if it happens,
    check carefully the inputs and their units; `alpha_sc` and `beta_voc` are
    often given in %/K in manufacturer datasheets and should be given
    in A/K and V/K here.

    The parameters returned by this function can be used by
    ``pvsystem.calcparams_desoto`` to calculate the values at different
    irradiance and cell temperature.

    Parameters
    ----------
    v_mp: float
        Module voltage at the maximum-power point at reference conditions [V].
    i_mp: float
        Module current at the maximum-power point at reference conditions [A].
    v_oc: float
        Open-circuit voltage at reference conditions [V].
    i_sc: float
        Short-circuit current at reference conditions [A].
    alpha_sc: float
        The short-circuit current (i_sc) temperature coefficient of the
        module [A/K].
    beta_voc: float
        The open-circuit voltage (v_oc) temperature coefficient of the
        module [V/K].
    cells_in_series: integer
        Number of cell in the module.
    EgRef: float, default 1.121 eV - value for silicon
        Energy of bandgap of semi-conductor used [eV]
    dEgdT: float, default -0.0002677 - value for silicon
        Variation of bandgap according to temperature [eV/K]
    temp_ref: float, default 25
        Reference temperature condition [C]
    irrad_ref: float, default 1000
        Reference irradiance condition [W/m2]
    root_kwargs: dictionary, default None
        Dictionary of arguments to pass onto scipy.optimize.root()

    Returns
    -------
    Tuple of the following elements:

        Dictionary with the following elements:

            * I_L_ref : light-generated current at reference conditions [A]
            * I_o_ref : diode saturation current at reference conditions [A]
            * R_s : series resistance [ohms]
            * R_sh_ref : shunt resistance at reference conditions [ohms].
            * a_ref : modified ideality factor at reference conditions.
                 The product of the usual diode ideality factor (n, unitless),
                 number of cells in series (Ns), and cell thermal voltage at
                 specified effective irradiance and cell temperature.
            * alpha_sc : temperature coefficient of short-circuit current [A/K]
            * EgRef : bandgap energy of semi-conductor used [eV]
            * dEgdT : variation of bandgap according to temperature [eV/K]
            * irrad_ref : reference irradiance condition [W/m^2]
            * temp_ref : reference temperature condition [C]

        OptimizeResult

            Optimization result of ``scipy.optimize.root``.
            See ``scipy.optimize.OptimizeResult`` for more details.

    References
    ----------
    .. [DeSo06] W. De Soto et al., "Improvement and validation of a model for
       photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
       2006.

    .. [Duff13] John A Dufﬁe, William A Beckman, "Solar Engineering of Thermal
       Processes", Wiley, 2013
    """

    try:
        from scipy.optimize import root
        import scipy.constants
    except ImportError:
        raise ImportError("The fit_desoto function requires scipy.")

    # Constants
    k = scipy.constants.value('Boltzmann constant in eV/K')
    Tref = temp_ref + 273.15  # [K]

    # initial guesses of variables for computing convergence:
    # Values are taken from [2], p753
    Rsh_0 = 100.0
    a_0 = 1.5*k*Tref*cells_in_series
    IL_0 = i_sc
    Io_0 = i_sc * np.exp(-v_oc/a_0)
    Rs_0 = (a_0*np.log1p((IL_0-i_mp)/Io_0) - v_mp)/i_mp
    # params_i : initial values vector
    params_i = np.array([IL_0, Io_0, a_0, Rsh_0, Rs_0])

    # specs of module
    specs = (i_sc, v_oc, i_mp, v_mp, beta_voc, alpha_sc, EgRef, dEgdT,
             Tref, k)

    # computing with system of equations described in [1]
    optimize_result = root(_system_of_equations_desoto, x0=params_i,
                           args=(specs,), **root_kwargs)

    if optimize_result.success:
        sdm_params = optimize_result.x
    else:
        raise RuntimeError(
            'Parameter estimation failed:\n' + optimize_result.message)

    # results
    return ({'I_L_ref': sdm_params[0],
             'I_o_ref': sdm_params[1],
             'a_ref': sdm_params[2],
             'R_sh_ref': sdm_params[3],
             'R_s': sdm_params[4],
             'alpha_sc': alpha_sc,
             'EgRef': EgRef,
             'dEgdT': dEgdT,
             'irrad_ref': irrad_ref,
             'temp_ref': temp_ref},
            optimize_result)


def _system_of_equations_desoto(params, specs):
    """Evaluates the systems of equations used to solve for the single
    diode equation parameters. Function designed to be used by
    scipy.optimize.root() in fit_desoto().

    Parameters
    ----------
    params: ndarray
        Array with parameters of the De Soto single diode model. Must be
        given in the following order: IL, Io, a, Rsh, Rs
    specs: tuple
        Specifications of pv module given by manufacturer. Must be given
        in the following order: Isc, Voc, Imp, Vmp, beta_oc, alpha_sc

    Returns
    -------
    system of equations to solve with scipy.optimize.root().


    References
    ----------
    .. [1] W. De Soto et al., "Improvement and validation of a model for
       photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
       2006.

    .. [2] John A Dufﬁe, William A Beckman, "Solar Engineering of Thermal
       Processes", Wiley, 2013
    """

    # six input known variables
    Isc, Voc, Imp, Vmp, beta_oc, alpha_sc, EgRef, dEgdT, Tref, k = specs

    # five parameters vector to find
    IL, Io, a, Rsh, Rs = params

    # five equation vector
    y = [0, 0, 0, 0, 0]

    # 1st equation - short-circuit - eq(3) in [1]
    y[0] = Isc - IL + Io * np.expm1(Isc * Rs / a) + Isc * Rs / Rsh

    # 2nd equation - open-circuit Tref - eq(4) in [1]
    y[1] = -IL + Io * np.expm1(Voc / a) + Voc / Rsh

    # 3rd equation - Imp & Vmp - eq(5) in [1]
    y[2] = Imp - IL + Io * np.expm1((Vmp + Imp * Rs) / a) \
        + (Vmp + Imp * Rs) / Rsh

    # 4th equation - Pmp derivated=0 - eq23.2.6 in [2]
    # caution: eq(6) in [1] has a sign error
    y[3] = Imp \
        - Vmp * ((Io / a) * np.exp((Vmp + Imp * Rs) / a) + 1.0 / Rsh) \
        / (1.0 + (Io * Rs / a) * np.exp((Vmp + Imp * Rs) / a) + Rs / Rsh)

    # 5th equation - open-circuit T2 - eq (4) at temperature T2 in [1]
    T2 = Tref + 2
    Voc2 = (T2 - Tref) * beta_oc + Voc  # eq (7) in [1]
    a2 = a * T2 / Tref  # eq (8) in [1]
    IL2 = IL + alpha_sc * (T2 - Tref)  # eq (11) in [1]
    Eg2 = EgRef * (1 + dEgdT * (T2 - Tref))  # eq (10) in [1]
    Io2 = Io * (T2 / Tref)**3 * np.exp(1 / k * (EgRef/Tref - Eg2/T2))  # eq (9)
    y[4] = -IL2 + Io2 * np.expm1(Voc2 / a2) + Voc2 / Rsh  # eq (4) at T2

    return y


def fit_pvsyst_sandia(ivcurves, specs, const=constants, maxiter=5,
                      eps1=1.e-3):
    """
    Estimate parameters for the PVsyst module performance model.

    Parameters
    ----------
    ivcurves : dict
        Contains IV curve data as arrays indexed by the IV curves

        * `i` - array of current (same length as `v`) [A]

        * `v` - array of voltage (same length as `i`) [V]

        * `ee` - float, effective irradiance, i.e., POA broadband
          irradiance adjusted by solar spectrum modifier [W/m^2]

        * `tc` - cell temperature [C]

        * `isc` - short circuit current [A]

        * `voc` - open circuit voltage [V]

        * `imp` - current at max power point [A]

        * `vmp` - voltage at max power point [V]

    specs : dict

        * `ns` - number of cells in series

        * `aisc` - module temperature coefficient for isc [A/C]

    const : OrderedDict, optional

        * `E0` - reference effective irradiance, default 1000 W/m2

        * `T0` - reference cell temperature, default 25C

        * `k` - 1.38066E-23 J/K (Boltzmann's constant)

        * `q` - 1.60218E-19 Coulomb (elementary charge)

    maxiter : int, default 5
        maximum number of iterations for the parameter updating part of the
        algorithm.

    eps1 : float, default 1e-3
        the desired tolerance for the IV curve fitting. Fitting stops when
        absolute values of the percent change in mean, max and standard
        deviation of Imp, Vmp and Pmp between iterations are all less than
        `eps1`, or when the number of iterations exceeds maxiter.

    Returns
    -------
    pvsyst : OrderedDict
        * `IL_ref` - light current at reference condition [A]
        * `Io_ref` - dark current at reference condition [A]
        * `eG` - effective band gap at reference condition [eV]
        * `Rsh_ref` - shunt resistance at reference condition [ohm]
        * `Rsh0` - shunt resistance at zero irradiance [ohm]
        * `Rshexp` - exponential factor defining decrease in Rsh with
          increasing effective irradiance
        * `Rs_ref` - series resistance at reference condition [ohm]
        * `gamma_ref` - diode (ideality) factor at reference condition
          [unitless]
        * `mugamma` - temperature coefficient for diode (ideality) factor [1/C]
        * `Iph` - array of values of light current Iph estimated for each IV
          curve [A]
        * `Io` - array of values of dark current Io estimated for each IV
          curve [A]
        * `Rsh` - array of values of shunt resistance Rsh estimated for each
          IV curve [ohm]
        * `Rs` - array of values of series resistance Rs estimated for each IV
          curve [ohm]
        * `u` - array of boolean indicating IV curves with parameter values
          deemed reasonable by the private function ``_filter_params``

    Notes
    -----
    Estimates parameters for the PVsyst module performance model [1]_, [2]_,
    [3]_.using methods documented in [4]_, [5]_, [6]_.
    Ported from PVLib Matlab [7]_.

    References
    ----------
    .. [1] K. Sauer, T. Roessler, C. W. Hansen, Modeling the Irradiance and
       Temperature Dependence of Photovoltaic Modules in PVsyst, IEEE Journal
       of Photovoltaics v5(1), January 2015.
    .. [2] A. Mermoud, PV Modules modeling, Presentation at the 2nd PV
       Performance Modeling Workshop, Santa Clara, CA, May 2013
    .. [3] A. Mermoud, T. Lejeuene, Performance Assessment of a Simulation
       Model for PV modules of any available technology, 25th European
       Photovoltaic Solar Energy Conference, Valencia, Spain, Sept. 2010
    .. [4] C. Hansen, Estimating Parameters for the PVsyst Version 6
       Photovoltaic Module Performance Model, Sandia National Laboratories
       Report SAND2015-8598
    .. [5] C. Hansen, Parameter Estimation for Single Diode Models of
       Photovoltaic Modules, Sandia National Laboratories Report SAND2015-2065
    .. [6] C. Hansen, Estimation of Parameters for Single Diode Models using
       Measured IV Curves, Proc. of the 39th IEEE PVSC, June 2013.
    .. [7] PVLib MATLAB https://github.com/sandialabs/MATLAB_PV_LIB
    """

    logging.basicConfig()
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)

    try:
        from scipy import optimize
        import statsmodels.api as sm
    except ImportError:
        raise ImportError('fit_pvsyst_sandia requires scipy and statsmodels')

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

    rsh = np.ones(n)
    for j in range(n):
        voltage, current = rectify_iv_curve(ivcurves['v'][j], ivcurves['i'][j])
        # initial estimate of Rsh, from integral over voltage regression
        # [5] Step 3a; [6] Step 3a
        _, _, _, rsh[j], _ = \
            fit_cocontent(voltage, current, vth[j] * specs['ns'])

    gamma_ref, mugamma = _fit_pvsyst_sandia_gamma(isc, voc, rsh, vth, tck,
                                                  const, specs)

    badgamma = np.isnan(gamma_ref) or np.isnan(mugamma) \
        or not np.isreal(gamma_ref) or not np.isreal(mugamma)

    if badgamma:
        raise RuntimeError(
            "Failed to estimate the diode (ideality) factor parameter.")

    else:
        pvsyst = OrderedDict()

        gamma = gamma_ref + mugamma * (tc - const['T0'])

        nnsvth = gamma * (vth * specs['ns'])

        # For each IV curve, sequentially determine initial values for Io, Rs,
        # and Iph [5] Step 3a; [6] Step 3

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

                iph[j] = isc[j] + io[j] * np.expm1(isc[j] / nnsvth[j]) \
                    + isc[j] * rs[j] / rsh[j]
            else:
                io[j] = np.nan
                rs[j] = np.nan
                iph[j] = np.nan

        # Filter IV curves for good initial values
        LOGGER.debug('filtering params ... may take awhile')
        # [5] Step 3b
        u = _filter_params(io, rsh, rs, ee, isc)

        # Refine Io to match Voc
        LOGGER.debug('Refine Io to match Voc')
        # [5] Step 3c
        tmpio = _update_io(rsh[u], rs[u], nnsvth[u], io[u], iph[u], voc[u])
        io[u] = tmpio

        # Calculate Iph to be consistent with Isc and current values of other
        LOGGER.debug('Calculate Iph to be consistent with Isc, ...')
        # parameters [6], Step 3c
        iph = isc + io * np.expm1(rs * isc / nnsvth) + isc * rs / rsh

        # Refine Rsh, Rs, Io and Iph in that order.
        counter = 1.  # counter variable for parameter updating while loop,
        # counts iterations
        prevconvergeparams = OrderedDict()
        prevconvergeparams['state'] = 0.0

        t14 = np.array([True])

        while t14.any() and counter <= maxiter:
            # update rsh to match max power point using a fixed point method.
            tmprsh = _update_rsh_fixed_pt(iph[u], io[u], rsh[u], rs[u],
                                          nnsvth[u],  vmp[u], imp[u])

            rsh[u] = tmprsh

            # Calculate Rs to be consistent with Rsh and maximum power point
            LOGGER.debug('step %d: calculate Rs', counter)
            phi = _calc_phi_exact(imp[u], iph[u], io[u], rsh[u], nnsvth[u])
            rs[u] = (iph[u] + io[u] - imp[u]) * rsh[u] / imp[u] - \
                nnsvth[u] * phi / imp[u] - vmp[u] / imp[u]

            # Update filter for good parameters
            u = _filter_params(io, rsh, rs, ee, isc)

            # Update value for io to match voc
            LOGGER.debug('step %d: calculate dark/saturation current (Io)',
                         counter)
            tmpio = _update_io(rsh[u], rs[u], nnsvth[u], io[u], iph[u], voc[u])
            io[u] = tmpio

            # Calculate Iph to be consistent with Isc and other parameters
            LOGGER.debug('step %d: calculate Iph', counter)
            iph = isc + io * np.expm1(rs * isc / nnsvth) + isc * rs / rsh

            # update filter for good parameters
            u = _filter_params(io, rsh, rs, ee, isc)

            # compute the IV curve from the current parameter values
            LOGGER.debug('step %d: compute IV curve', counter)
            result = singlediode(iph[u], io[u], rs[u], rsh[u], nnsvth[u])

            # check convergence criteria
            LOGGER.debug('step %d: checking convergence', counter)
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

        # Estimate Rsh0, Rsh_ref and Rshexp
        # Initial Guesses. Rsh0 is value at Ee=0.
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
        rshexp = 5.5

        # Find parameters for Rsh equation
        def fun_rsh(x, rshexp, ee, e0, rsh):
            tf = np.log10(_rsh_pvsyst(x, rshexp, ee, e0)) - np.log10(rsh)
            return tf
        x0 = np.array([grsh0, grshref])
        beta = optimize.least_squares(
            fun_rsh, x0, args=(rshexp, ee[u], const['E0'], rsh[u]),
            bounds=np.array([[1., 1.], [1.e7, 1.e6]]), verbose=2)

        # Extract PVsyst parameter values
        rsh0 = beta.x[0]
        rshref = beta.x[1]

        # Estimate Rs0
        t15 = np.logical_and(u, ee > 400)
        rs0 = np.mean(rs[t15])

        # Save parameter estimates in output structure
        pvsyst['IL_ref'] = iph0
        pvsyst['Io_ref'] = io0
        pvsyst['eG'] = eg
        pvsyst['Rs_ref'] = rs0
        pvsyst['gamma_ref'] = gamma_ref
        pvsyst['mugamma'] = mugamma
        pvsyst['Rsh0'] = rsh0
        pvsyst['Rsh_ref'] = rshref
        pvsyst['Rshexp'] = rshexp
        pvsyst['Iph'] = iph
        pvsyst['Io'] = io
        pvsyst['Rsh'] = rsh
        pvsyst['Rs'] = rs
        pvsyst['Ns'] = specs['ns']
        pvsyst['u'] = u

    return pvsyst


def _fit_pvsyst_sandia_gamma(isc, voc, rsh, vth, tck, const, specs):
    # Estimate the diode factor gamma from Isc-Voc data. Method incorporates
    # temperature dependence by means of the equation for Io

    # Helper function for fit_pvsyst_sandia

    y = np.log(isc - voc / rsh) - 3. * np.log(tck / (const['T0'] + 273.15))
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
    alpha = np.linalg.lstsq(x, y[~uu], rcond=None)[0]

    gamma_ref = 1. / alpha[3]
    mugamma = alpha[4] / alpha[3] ** 2
    return gamma_ref, mugamma


def _update_io(rsh, rs, nnsvth, io, il, voc):
    """
    _update_io adjusts io to match voc using other parameter values.

    Helper function for fit_pvsyst_sandia

    Description
    -----------
    _update_io adjusts io to match voc using other parameter values,
    i.e., Rsh (shunt resistance), Rs (Series Resistance), n (diode factor), and
    IL (Light Current). Io is updated iteratively 10 times or until successive
    values are less than 0.000001 % different. The updating is similar to
    Newton's method.

    Parameters
    ----------
    rsh: a numpy array of length N of values for the shunt resistance (ohm)
    rs: a numpy array of length N of values for the series resistance (ohm)
    nnsvth: a numpy array of length N of values for the diode factor x thermal
            voltage for the module, equal to Ns (number of cells in series) x
            Vth (thermal voltage per cell).
    io: a numpy array of length N of initial values for Io (A)
    il: a numpy array of length N of values for lighbt current IL (A)
    voc: a numpy array of length N of values for Voc (V)

    Returns
    -------
    next_io: a numpy array of lenght N of updated values for Io

    References
    ----------
    .. [1] C. Hansen, Parameter Estimation for Single Diode Models of
       Photovoltaic Modules, Sandia National Laboratories Report SAND2015-2065
    .. [2] C. Hansen, Estimation of Parameters for Single Diode Models using
       Measured IV Curves, Proc. of the 39th IEEE PVSC, June 2013.
    .. [3] PVLib MATLAB https://github.com/sandialabs/MATLAB_PV_LIB
    """

    eps = 1e-6
    niter = 10
    k = 1
    maxerr = 1

    tio = io  # Current Estimate of Io

    while maxerr > eps and k < niter:
        # Predict Voc
        pvoc = v_from_i(rsh, rs, nnsvth, 0., tio, il)

        # Difference in Voc
        dvoc = pvoc - voc

        # Update Io
        next_io = tio * (1. + (2. * dvoc) / (2. * nnsvth - dvoc))

        # Calculate Maximum Percent Difference
        maxerr = np.max(np.abs(next_io - tio) / tio) * 100.
        tio = next_io
        k += 1

    return next_io


def _rsh_pvsyst(x, rshexp, g, go):
    # computes rsh for PVsyst model where the parameters are in vector xL
    # x[0] = Rsh0
    # x[1] = Rshref

    rsho = x[0]
    rshref = x[1]

    rshb = np.maximum(
        (rshref - rsho * np.exp(-rshexp)) / (1. - np.exp(-rshexp)), 0.)
    rsh = rshb + (rsho - rshb) * np.exp(-rshexp * g / go)
    return rsh


def _filter_params(io, rsh, rs, ee, isc):
    # Function _filter_params identifies bad parameter sets. A bad set contains
    # Nan, non-positive or imaginary values for parameters; Rs > Rsh; or data
    # where effective irradiance Ee differs by more than 5% from a linear fit
    # to Isc vs. Ee

    # Helper function for fit_pvsyst_sandia

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

    Helper function for fit_pvsyst_sandia

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
    convergeparam: OrderedDict containing the following for Imp, Vmp and Pmp:
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


def _update_rsh_fixed_pt(il, io, rsh, rs, nnsvth, vmp, imp):
    """
    Adjust Rsh to match Vmp using other paramter values

    Helper function for fit_pvsyst_sandia

    Description
    -----------
    _update_rsh_fixed_pt adjusts rsh to match vmp using other parameter values,
    i.e., Rs (series resistance), n (diode factor), Io (dark current), and IL
    (light current). Rsh is updated iteratively using a fixed point expression
    obtained from combining Vmp = Vmp(Imp) (using the analytic solution to the
    single diode equation) and dP / dI = 0 at Imp. 500 iterations are performed
    because convergence can be very slow. [1]_, [2]_. Ported from PVLib Matlab
    [3]_.

    Parameters
    ----------
    il: a numpy array of length N of values for light current IL (A)
    io: a numpy array of length N of values for Io (A)
    rsh: a numpy array of length N of initial values for shunt resistance (ohm)
    rs: a numpy array of length N of values for series resistance (ohm)
    nnsvth: a numpy array length N of values for the diode factor x thermal
            voltage for the module, equal to Ns (number of cells in series) x
            Vth (thermal voltage per cell).
    vmp: a numpy array of length N of values for Vmp (V)
    imp: a numpy array of length N of values for Imp (A)

    Returns
    -------
    next_rsh: a numpy array of length N of updated values for Rsh

    References
    ----------
    .. [1] C. Hansen, Parameter Estimation for Single Diode Models of
       Photovoltaic Modules, Sandia National Laboratories Report SAND2015-2065
    .. [2] C. Hansen, Estimation of Parameters for Single Diode Models using
       Measured IV Curves, Proc. of the 39th IEEE PVSC, June 2013.
    .. [3] PVLib MATLAB https://github.com/sandialabs/MATLAB_PV_LIB
    """
    maxiter = 500
    z = 0.
    i = 1
    rsh_diff = 999.

    def _done(i, z, maxiter, rsh_diff):
        with np.errstate(invalid='ignore'):
            return (i > maxiter) or np.all((rsh_diff < 1e-4) | np.isnan(z) |
                                           np.isinf(z))

    while not _done(i, z, maxiter, rsh_diff):
        z = _calc_phi_exact(imp, il, io, rsh, nnsvth)
        next_rsh = (1 + z) / z * ((il + io) * rsh / imp - nnsvth * z / imp -
                                  2 * vmp / imp)
        rsh_diff = abs(next_rsh - rsh)
        rsh = next_rsh
        i += 1
    return next_rsh


def _calc_phi_exact(imp, il, io, rsh, nnsvth):
    """
    Computes Lambert W value appearing in the analytic
    solution V=V(I) of the single diode equation at the max power point.

    Helper function for fit_pvsyst_sandia

    Parameters
    ----------
    imp: a numpy array of length N of values for Imp (A)
    il: a numpy array of length N of values for the light current IL (A)
    io: a numpy array of length N of values for Io (A)
    rsh: a numpy array of length N of values for the shunt resistance (ohm)
    nnsvth: a numpy array of length N of values for the diode factor x
            thermal voltage for the module, equal to Ns
            (number of cells in series) x Vth
            (thermal voltage per cell).

    Returns
    -------
    w: a numpy array of values for the Lambert W function for solving
       V = V(I)

    Description
    -----------
    Calculates values for the Lambert W function which
    are used in the analytic solution V=V(I) for the single diode equation at
    the maximum power point.
    For V=V(I), phi = W(Io*Rsh/n*Vth * exp((IL + Io - Imp)*Rsh/n*Vth)).
    If the argument x of W(x) is negative, returns NaN.

    References
    ----------
    .. [1] C. Hansen, Parameter Estimation for Single Diode Models of
       Photovoltaic Modules, Sandia National Laboratories Report SAND2015-2065
    .. [2] PVLib MATLAB https://github.com/sandialabs/MATLAB_PV_LIB
    .. [3] A. Jain, A. Kapoor, "Exact analytical solutions of the parameters of
       real solar cells using Lambert W-function", Solar Energy Materials and
       Solar Cells, 81 (2004) 269-277.
    """

    try:
        from scipy.special import lambertw
    except ImportError:
        raise ImportError('Requires lambertw from scipy')

    def _logw_fn_for_v(rsh, io, inv_nnsvth, il, imp, numiter=8):
        # evaluates W in log space for V=V(I)
        with np.errstate(invalid='ignore'):
            logargw = np.log(rsh) + np.log(io) + np.log(inv_nnsvth) \
                + rsh * (il + io - imp) * inv_nnsvth
        return _converge_w(logargw, numiter)

    # Argument for Lambert W function involved in V = V(I) [2] Eq. 12; [3]
    # Eq. 3
    # avoid numpy divide by zero warning
    with np.errstate(divide='ignore'):
        inv_nnsvth = 1. / nnsvth
    # avoid 0. * np.inf warning
    with np.errstate(invalid='ignore'):
        argw = np.where(
            np.isinf(inv_nnsvth),
            np.nan,
            rsh * io * inv_nnsvth * np.exp(rsh * (il + io - imp) * inv_nnsvth))
        w = np.where(argw > 0, lambertw(argw).real, 0.0)  # 0. is a placeholder
    # avoid divide by zero, log(negative) warning
    with np.errstate(divide='ignore', invalid='ignore'):
        # switch to log if argw blew up
        w = np.where(np.isposinf(argw),
                     _logw_fn_for_v(rsh, io, inv_nnsvth, il, imp), w)
    w = np.where(w == 0.0, np.nan, w)
    return w


def _converge_w(logargw, numiter):
    # Newton-Raphson method to solve w+log(w)=logargW.
    # 3 iterations gives approximately 8 digits of precision.
    x = logargw.copy()
    for i in range(numiter):
        x *= (1. - np.log(x) + logargw) / (1. + x)
    return x


def _calc_theta_exact(vmp, il, io, rsh, rs, nnsvth):
    """
    Computes Lambert W value appearing in the analytic
    solution I=I(V) of the single diode equation at the max power point.

    Helper function for fit_pvsyst_sandia

    Parameters
    ----------
    vmp: a numpy array of length N of values for Vmp (V)
    il: a numpy array of length N of values for the light current IL (A)
    io: a numpy array of length N of values for Io (A)
    rsh: a numpy array of length N of values for the shunt resistance (ohm)
    rs: a numpy array of length N of values for the series resistance (ohm)
    nnsvth: a numpy array of length N of values for the diode factor x
            thermal voltage for the module, equal to Ns
            (number of cells in series) x Vth
            (thermal voltage per cell).

    Returns
    -------
    w: a numpy array of values for the Lambert W function for solving
       I = I(V)

    Description
    -----------
    Calculates values for the Lambert W function which
    are used in the analytic solution I=I(V) for the single diode equation at
    the maximum power point.
    For I=I(V), theta = W(Rs*Io/n*Vth * Rsh/ (Rsh+Rs) *
        exp(Rsh / (Rsh+Rs)*((Rs(IL+Io) + V)/n*Vth))
    If the argument x of W(x) is negative, returns NaN.

    References
    ----------
    .. [1] C. Hansen, Parameter Estimation for Single Diode Models of
       Photovoltaic Modules, Sandia National Laboratories Report SAND2015-2065
    .. [2] PVLib MATLAB https://github.com/sandialabs/MATLAB_PV_LIB
    .. [3] A. Jain, A. Kapoor, "Exact analytical solutions of the parameters of
       real solar cells using Lambert W-function", Solar Energy Materials and
       Solar Cells, 81 (2004) 269-277.
    """

    try:
        from scipy.special import lambertw
    except ImportError:
        raise ImportError('Requires lambertw from scipy')

    def _logw_fn_for_i(rsh, io, inv_nnsvth, inv_rsh_rs, il, vmp, rs,
                       numiter=3):
        # evaluates W in log space for I=I(V)
        logargw = (
            np.log(rsh) + np.log(inv_rsh_rs) + np.log(rs) + np.log(io)
            + np.log(inv_nnsvth) + rsh * inv_rsh_rs * (rs * (il + io) + vmp)
            * inv_nnsvth)
        return _converge_w(logargw, numiter)

    with np.errstate(divide='ignore'):
        inv_nnsvth = 1. / nnsvth
        inv_rsh_rs = 1. / (rsh + rs)
    # Argument for Lambert W function involved in I = I(V) [2] Eq. 11; [3]
    # E1. 2
    # avoid 0. * np.inf warning
    with np.errstate(invalid='ignore'):
        argw = np.where(
            nnsvth == 0,
            np.nan,
            rsh * inv_rsh_rs * rs * io * inv_nnsvth * np.exp(
                rsh * inv_rsh_rs * (rs * (il + io) + vmp) * inv_nnsvth))
        w = np.where(argw > 0, lambertw(argw).real, 0.0)
    # avoid divide by zero, log(negative) warning
    with np.errstate(divide='ignore', invalid='ignore'):
        # switch to log if argw blew up
        w = np.where(np.isposinf(argw),
                     _logw_fn_for_i(rsh, io, inv_nnsvth, inv_rsh_rs, il, vmp,
                                    rs), w)
    w = np.where(w == 0.0, np.nan, w)
    return w
