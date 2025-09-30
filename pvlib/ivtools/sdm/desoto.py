import numpy as np

from scipy import constants
from scipy import optimize

from pvlib.ivtools.utils import rectify_iv_curve
from pvlib.ivtools.sde import _fit_sandia_cocontent

from pvlib.ivtools.sdm._fit_desoto_pvsyst_sandia import (
    _extract_sdm_params, _initial_iv_params, _update_iv_params
)

CONSTANTS = {'E0': 1000.0, 'T0': 25.0, 'k': constants.k, 'q': constants.e}


def fit_desoto(v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, cells_in_series,
               EgRef=1.121, dEgdT=-0.0002677, temp_ref=25, irrad_ref=1000,
               init_guess={}, root_kwargs={}):
    """
    Calculates the parameters for the De Soto single diode model.

    This procedure (described in [1]_) fits the De Soto model [2]_ using
    common specifications given by manufacturers in the
    datasheets of PV modules.

    The solution is found using :py:func:`scipy.optimize.root`,
    with the default solver method 'hybr'.
    No restriction is put on the fit variables, e.g. series
    or shunt resistance could go negative. Nevertheless, if it happens,
    check carefully the inputs and their units. For example, ``alpha_sc`` and
    ``beta_voc`` are often given in %/K in manufacturers datasheets but should
    be given in A/K and V/K here.

    The parameters returned by this function can be used by
    :py:func:`pvlib.pvsystem.calcparams_desoto` to calculate single diode
    equation parameters at different irradiance and cell temperature.

    Parameters
    ----------
    v_mp: float
        Module voltage at the maximum-power point at reference conditions. [V]
    i_mp: float
        Module current at the maximum-power point at reference conditions. [A]
    v_oc: float
        Open-circuit voltage at reference conditions. [V]
    i_sc: float
        Short-circuit current at reference conditions. [A]
    alpha_sc: float
        The short-circuit current (``i_sc``) temperature coefficient of the
        module. [A/K]
    beta_voc: float
        The open-circuit voltage (``v_oc``) temperature coefficient of the
        module. [V/K]
    cells_in_series: integer
        Number of cell in the module.
    EgRef: float, default 1.121 eV - value for silicon
        Energy of bandgap of semi-conductor used. [eV]
    dEgdT: float, default -0.0002677 - value for silicon
        Variation of bandgap according to temperature. [1/K]
    temp_ref: float, default 25
        Reference temperature condition. [°C]
    irrad_ref: float, default 1000
        Reference irradiance condition. [Wm⁻²]
    init_guess: dict, optional
        Initial values for optimization. Keys can be `'Rsh_0'`, `'a_0'`,
        `'IL_0'`, `'Io_0'`, `'Rs_0'`.
    root_kwargs : dictionary, optional
        Dictionary of arguments to pass onto scipy.optimize.root()

    Returns
    -------
    dict with the following elements:
        I_L_ref: float
            Light-generated current at reference conditions. [A]
        I_o_ref: float
            Diode saturation current at reference conditions. [A]
        R_s: float
            Series resistance. [ohm]
        R_sh_ref: float
            Shunt resistance at reference conditions. [ohm].
        a_ref: float
            Modified ideality factor at reference conditions.
            The product of the usual diode ideality factor (n, unitless),
            number of cells in series (Ns), and cell thermal voltage at
            specified effective irradiance and cell temperature.
        alpha_sc: float
            The short-circuit current (i_sc) temperature coefficient of the
            module. [A/K]
        EgRef: float
            Energy of bandgap of semi-conductor used. [eV]
        dEgdT: float
            Variation of bandgap according to temperature. [1/K]
        irrad_ref: float
            Reference irradiance condition. [Wm⁻²]
        temp_ref: float
            Reference temperature condition. [°C]

    scipy.optimize.OptimizeResult
        Optimization result of scipy.optimize.root().
        See scipy.optimize.OptimizeResult for more details.

    References
    ----------
    .. [1] J. A Duffie, W. A Beckman, "Solar Engineering of Thermal Processes",
       4th ed., Wiley, 2013. :doi:`10.1002/9781118671603`
    .. [2] W. De Soto et al., "Improvement and validation of a model for
       photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
       2006. :doi:`10.1016/j.solener.2005.06.010`

    """

    # Constants
    k = constants.value('Boltzmann constant in eV/K')
    Tref = temp_ref + 273.15  # [K]

    # initial guesses of variables for computing convergence:
    # Default values are taken from [1], p753
    init_guess_keys = ['IL_0', 'Io_0', 'Rs_0', 'Rsh_0', 'a_0']  # order matters
    init = {key: None for key in init_guess_keys}
    init['IL_0'] = i_sc
    init['a_0'] = 1.5*k*Tref*cells_in_series
    init['Io_0'] = i_sc * np.exp(-v_oc/init['a_0'])
    init['Rs_0'] = (init['a_0']*np.log1p((init['IL_0'] - i_mp)/init['Io_0'])
                    - v_mp) / i_mp
    init['Rsh_0'] = 100.0
    # overwrite if optional init_guess is provided
    for key in init_guess:
        if key in init_guess_keys:
            init[key] = init_guess[key]
        else:
            raise ValueError(f"'{key}' is not a valid name;"
                             f" allowed values are {init_guess_keys}")
    # params_i : initial values vector
    params_i = np.array([init[k] for k in init_guess_keys])

    # specs of module
    specs = (i_sc, v_oc, i_mp, v_mp, beta_voc, alpha_sc, EgRef, dEgdT,
             Tref, k)

    # computing with system of equations described in [1]
    optimize_result = optimize.root(_system_of_equations_desoto, x0=params_i,
                                    args=(specs,), **root_kwargs)

    if optimize_result.success:
        sdm_params = optimize_result.x
    else:
        raise RuntimeError(
            'Parameter estimation failed:\n' + optimize_result.message)

    # results
    return ({'I_L_ref': sdm_params[0],
             'I_o_ref': sdm_params[1],
             'R_s': sdm_params[2],
             'R_sh_ref': sdm_params[3],
             'a_ref': sdm_params[4],
             'alpha_sc': alpha_sc,
             'EgRef': EgRef,
             'dEgdT': dEgdT,
             'irrad_ref': irrad_ref,
             'temp_ref': temp_ref},
            optimize_result)


def _system_of_equations_desoto(params, specs):
    """Evaluates the systems of equations used to solve for the single
    diode equation parameters. Function designed to be used by
    scipy.optimize.root in fit_desoto.

    Parameters
    ----------
    params: ndarray
        Array with parameters of the De Soto single diode model. Must be
        given in the following order: IL, Io, a, Rs, Rsh
    specs: tuple
        Specifications of pv module given by manufacturer. Must be given
        in the following order: Isc, Voc, Imp, Vmp, beta_oc, alpha_sc

    Returns
    -------
    value of the system of equations to solve with scipy.optimize.root().
    """

    # six input known variables
    Isc, Voc, Imp, Vmp, beta_oc, alpha_sc, EgRef, dEgdT, Tref, k = specs

    # five parameters vector to find
    IL, Io, Rs, Rsh, a = params

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


def fit_desoto_sandia(ivcurves, specs, const=None, maxiter=5, eps1=1.e-3):
    """
    Estimate parameters for the De Soto module performance model.

    Parameters
    ----------
    ivcurves : dict
        i : array
            One array element for each IV curve. The jth element is itself an
            array of current for jth IV curve (same length as v[j]) [A]
        v : array
            One array element for each IV curve. The jth element is itself an
            array of voltage for jth IV curve  (same length as i[j]) [V]
        ee : array
            effective irradiance for each IV curve, i.e., POA broadband
            irradiance adjusted by solar spectrum modifier [W / m^2]
        tc : array
            cell temperature for each IV curve. [°C]
        i_sc : array
            short circuit current for each IV curve [A]
        v_oc : array
            open circuit voltage for each IV curve [V]
        i_mp : array
            current at max power point for each IV curve [A]
        v_mp : array
            voltage at max power point for each IV curve [V]

    specs : dict
        cells_in_series : int
            number of cells in series
        alpha_sc : float
            temperature coefficient of Isc [A/C]
        beta_voc : float
            temperature coefficient of Voc [V/C]

    const : dict
        E0 : float
            effective irradiance at STC, default 1000 [Wm⁻²]
        T0 : float
            cell temperature at STC, default 25°C. [°C]
        k : float
            Boltzmann's constant [J/K]
        q : float
            elementary charge [Coulomb]

    maxiter : int, default 5
        input that sets the maximum number of iterations for the parameter
        updating part of the algorithm.

    eps1: float, default 1e-3
        Tolerance for the IV curve fitting. The parameter updating stops when
        absolute values of the percent change in mean, max and standard
        deviation of Imp, Vmp and Pmp between iterations are all less than
        eps1, or when the number of iterations exceeds maxiter.

    Returns
    -------
    dict
        I_L_ref : float
            Light current at STC [A]
        I_o_ref : float
            Dark current at STC [A]
        EgRef : float
            Effective band gap at STC [eV]
        R_s : float
            Series resistance at STC [ohm]
        R_sh_ref : float
            Shunt resistance at STC [ohm]
        cells_in_series : int
            Number of cells in series
        iph : array
            Light current for each IV curve [A]
        io : array
            Dark current for each IV curve [A]
        rs : array
            Series resistance for each IV curve [ohm]
        rsh : array
            Shunt resistance for each IV curve [ohm]
        a_ref : float
            The product of the usual diode ideality factor (n, unitless),
            number of cells in series (Ns), and cell thermal voltage at
            reference conditions, in units of V.
        dEgdT : float
            The temperature dependence of the energy bandgap (Eg) at reference
            conditions [1/K].
        u : array
            Boolean for each IV curve indicating that the parameter values
            are deemed reasonable by the private function ``_filter_params``

    Notes
    -----
    The De Soto module performance model is described in [1]_. The fitting
    method is documented in [2]_, [3]_. Ported from PVLib Matlab [4]_.

    References
    ----------
    .. [1] W. De Soto et al., "Improvement and validation of a model for
       photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
       2006. :doi:`10.1016/j.solener.2005.06.010`
    .. [2] C. Hansen, Parameter Estimation for Single Diode Models of
       Photovoltaic Modules, Sandia National Laboratories Report SAND2015-2065.
       :doi:`10.2172/1177157`
    .. [3] C. Hansen, Estimation of Parameters for Single Diode Models using
        Measured IV Curves, Proc. of the 39th IEEE PVSC, June 2013.
        :doi:`10.1109/PVSC.2013.6744135`
    .. [4] PVLib MATLAB https://github.com/sandialabs/MATLAB_PV_LIB
    """

    if const is None:
        const = CONSTANTS

    ee = ivcurves['ee']
    tc = ivcurves['tc']
    tck = tc + 273.15
    isc = ivcurves['i_sc']
    voc = ivcurves['v_oc']
    imp = ivcurves['i_mp']
    vmp = ivcurves['v_mp']

    # Cell Thermal Voltage
    vth = const['k'] / const['q'] * tck

    n = len(voc)

    # Initial estimate of Rsh used to obtain the diode factor gamma0 and diode
    # temperature coefficient mu_gamma. Rsh is estimated using the co-content
    # integral method.

    rsh = np.ones(n)
    for j in range(n):
        voltage, current = rectify_iv_curve(ivcurves['v'][j], ivcurves['i'][j])
        # initial estimate of Rsh, from integral over voltage regression
        # [5] Step 3a; [6] Step 3a
        _, _, _, rsh[j], _ = _fit_sandia_cocontent(
            voltage, current, vth[j] * specs['cells_in_series'])

    n0 = _fit_desoto_sandia_diode(ee, voc, vth, tc, specs, const)

    bad_n = np.isnan(n0) or not np.isreal(n0)

    if bad_n:
        raise RuntimeError(
            "Failed to estimate the diode (ideality) factor parameter;"
            " aborting parameter estimation.")

    nnsvth = n0 * specs['cells_in_series'] * vth

    # For each IV curve, sequentially determine initial values for Io, Rs,
    # and Iph [5] Step 3a; [6] Step 3
    iph, io, rs, u = _initial_iv_params(ivcurves, ee, voc, isc, rsh,
                                        nnsvth)

    # Update values for each IV curve to converge at vmp, imp, voc and isc
    iph, io, rs, rsh, u = _update_iv_params(voc, isc, vmp, imp, ee,
                                            iph, io, rs, rsh, nnsvth, u,
                                            maxiter, eps1)

    # get single diode models from converged values for each IV curve
    desoto = _extract_sdm_params(ee, tc, iph, io, rs, rsh, n0, u,
                                 specs, const, model='desoto')
    # Add parameters estimated in this function
    desoto['a_ref'] = n0 * specs['cells_in_series'] * const['k'] / \
        const['q'] * (const['T0'] + 273.15)
    desoto['cells_in_series'] = specs['cells_in_series']

    return desoto


def _fit_desoto_sandia_diode(ee, voc, vth, tc, specs, const):
    # estimates the diode factor for the De Soto model.
    # Helper function for fit_desoto_sandia
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            'Parameter extraction using Sandia method requires statsmodels')

    x = specs['cells_in_series'] * vth * np.log(ee / const['E0'])
    y = voc - specs['beta_voc'] * (tc - const['T0'])
    new_x = sm.add_constant(x)
    res = sm.RLM(y, new_x).fit()
    return np.array(res.params)[1]
