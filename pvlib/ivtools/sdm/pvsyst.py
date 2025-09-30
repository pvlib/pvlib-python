import numpy as np

from scipy import constants

from pvlib.pvsystem import calcparams_pvsyst
from pvlib.singlediode import bishop88_mpp

from pvlib.ivtools.utils import rectify_iv_curve
from pvlib.ivtools.sde import _fit_sandia_cocontent

from pvlib.tools import _first_order_centered_difference

from pvlib.ivtools.sdm._fit_desoto_pvsyst_sandia import (
    _extract_sdm_params, _initial_iv_params, _update_iv_params
)

from pvlib.pvsystem import (
    _pvsyst_Rsh, _pvsyst_IL, _pvsyst_Io, _pvsyst_nNsVth, _pvsyst_gamma
)

CONSTANTS = {'E0': 1000.0, 'T0': 25.0, 'k': constants.k, 'q': constants.e}


def fit_pvsyst_sandia(ivcurves, specs, const=None, maxiter=5, eps1=1.e-3):
    """
    Estimate parameters for the PVsyst module performance model.

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
            cell temperature for each IV curve [°C]
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
            temperature coefficient of isc [A/C]

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
            light current at STC [A]
        I_o_ref : float
            dark current at STC [A]
        EgRef : float
            effective band gap at STC [eV]
        R_s : float
            series resistance at STC [ohm]
        R_sh_ref : float
            shunt resistance at STC [ohm]
        R_sh_0 : float
            shunt resistance at zero irradiance [ohm]
        R_sh_exp : float
            exponential factor defining decrease in shunt resistance with
            increasing effective irradiance
        gamma_ref : float
            diode (ideality) factor at STC [unitless]
        mu_gamma : float
            temperature coefficient for diode (ideality) factor [1/K]
        cells_in_series : int
            number of cells in series
        iph : array
            light current for each IV curve [A]
        io : array
            dark current for each IV curve [A]
        rs : array
            series resistance for each IV curve [ohm]
        rsh : array
            shunt resistance for each IV curve [ohm]
        u : array
            boolean for each IV curve indicating that the parameter values
            are deemed reasonable by the private function ``_filter_params``

    Notes
    -----
    The PVsyst module performance model is described in [1]_, [2]_, and [3]_.
    The fitting method is documented in [4]_, [5]_, and [6]_.
    Ported from PVLib Matlab [7]_.

    References
    ----------
    .. [1] K. Sauer, T. Roessler, C. W. Hansen, Modeling the Irradiance and
       Temperature Dependence of Photovoltaic Modules in PVsyst, IEEE Journal
       of Photovoltaics v5(1), January 2015.
       :doi:`10.1109/JPHOTOV.2014.2364133`
    .. [2] A. Mermoud, PV Modules modeling, Presentation at the 2nd PV
       Performance Modeling Workshop, Santa Clara, CA, May 2013
    .. [3] A. Mermoud, T. Lejeuene, Performance Assessment of a Simulation
       Model for PV modules of any available technology, 25th European
       Photovoltaic Solar Energy Conference, Valencia, Spain, Sept. 2010
    .. [4] C. Hansen, Estimating Parameters for the PVsyst Version 6
       Photovoltaic Module Performance Model, Sandia National Laboratories
       Report SAND2015-8598. :doi:`10.2172/1223058`
    .. [5] C. Hansen, Parameter Estimation for Single Diode Models of
       Photovoltaic Modules, Sandia National Laboratories Report SAND2015-2065.
       :doi:`10.2172/1177157`
    .. [6] C. Hansen, Estimation of Parameters for Single Diode Models using
        Measured IV Curves, Proc. of the 39th IEEE PVSC, June 2013.
        :doi:`10.1109/PVSC.2013.6744135`
    .. [7] PVLib MATLAB https://github.com/sandialabs/MATLAB_PV_LIB
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

    n = len(ivcurves['v_oc'])

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

    gamma_ref, mu_gamma = _fit_pvsyst_sandia_gamma(voc, isc, rsh, vth, tck,
                                                   specs, const)

    badgamma = np.isnan(gamma_ref) or np.isnan(mu_gamma) \
        or not np.isreal(gamma_ref) or not np.isreal(mu_gamma)

    if badgamma:
        raise RuntimeError(
            "Failed to estimate the diode (ideality) factor parameter;"
            " aborting parameter estimation.")

    gamma = gamma_ref + mu_gamma * (tc - const['T0'])
    nnsvth = gamma * (vth * specs['cells_in_series'])

    # For each IV curve, sequentially determine initial values for Io, Rs,
    # and Iph [5] Step 3a; [6] Step 3
    iph, io, rs, u = _initial_iv_params(ivcurves, ee, voc, isc, rsh,
                                        nnsvth)

    # Update values for each IV curve to converge at vmp, imp, voc and isc
    iph, io, rs, rsh, u = _update_iv_params(voc, isc, vmp, imp, ee,
                                            iph, io, rs, rsh, nnsvth, u,
                                            maxiter, eps1)

    # get single diode models from converged values for each IV curve
    pvsyst = _extract_sdm_params(ee, tc, iph, io, rs, rsh, gamma, u,
                                 specs, const, model='pvsyst')
    # Add parameters estimated in this function
    pvsyst['gamma_ref'] = gamma_ref
    pvsyst['mu_gamma'] = mu_gamma
    pvsyst['cells_in_series'] = specs['cells_in_series']

    return pvsyst


def _fit_pvsyst_sandia_gamma(voc, isc, rsh, vth, tck, specs, const):
    # Estimate the diode factor gamma from Isc-Voc data. Method incorporates
    # temperature dependence by means of the equation for Io

    y = np.log(isc - voc / rsh) - 3. * np.log(tck / (const['T0'] + 273.15))
    x1 = const['q'] / const['k'] * (1. / (const['T0'] + 273.15) - 1. / tck)
    x2 = voc / (vth * specs['cells_in_series'])
    uu = np.logical_or(np.isnan(y), np.isnan(x1), np.isnan(x2))

    x = np.vstack((np.ones(len(x1[~uu])), x1[~uu], -x1[~uu] *
                   (tck[~uu] - (const['T0'] + 273.15)), x2[~uu],
                   -x2[~uu] * (tck[~uu] - (const['T0'] + 273.15)))).T
    alpha = np.linalg.lstsq(x, y[~uu], rcond=None)[0]

    gamma_ref = 1. / alpha[3]
    mu_gamma = alpha[4] / alpha[3] ** 2
    return gamma_ref, mu_gamma


def pvsyst_temperature_coeff(alpha_sc, gamma_ref, mu_gamma, I_L_ref, I_o_ref,
                             R_sh_ref, R_sh_0, R_s, cells_in_series,
                             R_sh_exp=5.5, EgRef=1.121, irrad_ref=1000,
                             temp_ref=25):
    r"""
    Calculates the temperature coefficient of power for a pvsyst single
    diode model.

    The temperature coefficient is determined as the numerical derivative
    :math:`\frac{dP}{dT}` at the maximum power point at reference conditions
    [1]_.

    Parameters
    ----------
    alpha_sc : float
        The short-circuit current temperature coefficient of the module. [A/C]

    gamma_ref : float
        The diode ideality factor. [unitless]

    mu_gamma : float
        The temperature coefficient for the diode ideality factor. [1/K]

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions.
        [A]

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions.
        [A]

    R_sh_ref : float
        The shunt resistance at reference conditions. [ohm]

    R_sh_0 : float
        The shunt resistance at zero irradiance conditions. [ohm]

    R_s : float
        The series resistance at reference conditions. [ohm]

    cells_in_series : int
        The number of cells connected in series.

    R_sh_exp : float, default 5.5
        The exponent in the equation for shunt resistance. [unitless]

    EgRef : float, default 1.121
        The energy bandgap of the module's cells at reference temperature.
        Default of 1.121 eV is for crystalline silicon. Must be positive. [eV]

    irrad_ref : float, default 1000
        Reference irradiance. [Wm⁻²].

    temp_ref : float, default 25
        Reference cell temperature. [°C]


    Returns
    -------
    gamma_pdc : float
        Temperature coefficient of power at maximum power point at reference
        conditions. [1/C]

    References
    ----------
    .. [1] K. Sauer, T. Roessler, C. W. Hansen, Modeling the Irradiance and
       Temperature Dependence of Photovoltaic Modules in PVsyst, IEEE Journal
       of Photovoltaics v5(1), January 2015.
    """

    def maxp(temp_cell, irrad_ref, alpha_sc, gamma_ref, mu_gamma, I_L_ref,
             I_o_ref, R_sh_ref, R_sh_0, R_s, cells_in_series, R_sh_exp, EgRef,
             temp_ref):
        params = calcparams_pvsyst(
            irrad_ref, temp_cell, alpha_sc, gamma_ref, mu_gamma, I_L_ref,
            I_o_ref, R_sh_ref, R_sh_0, R_s, cells_in_series, R_sh_exp, EgRef,
            irrad_ref, temp_ref)
        res = bishop88_mpp(*params)
        return res[2]

    args = (irrad_ref, alpha_sc, gamma_ref, mu_gamma, I_L_ref,
            I_o_ref, R_sh_ref, R_sh_0, R_s, cells_in_series, R_sh_exp, EgRef,
            temp_ref)
    pmp = maxp(temp_ref, *args)
    gamma_pdc = _first_order_centered_difference(maxp, x0=temp_ref, args=args)

    return gamma_pdc / pmp


def fit_pvsyst_iec61853_sandia_2025(effective_irradiance, temp_cell,
                                    i_sc, v_oc, i_mp, v_mp,
                                    cells_in_series, EgRef=1.121,
                                    alpha_sc=None, beta_mp=None,
                                    R_s=None, r_sh_coeff=0.12,
                                    min_Rsh_irradiance=None,
                                    irradiance_tolerance=20,
                                    temperature_tolerance=1):
    """
    Estimate parameters for the PVsyst module performance model using
    IEC 61853-1 matrix measurements.

    Parameters
    ----------
    effective_irradiance : array
        Effective irradiance for each test condition [W/m²]
    temp_cell : array
        Cell temperature for each test condition. [°C]
    i_sc : array
        Short circuit current for each test condition [A]
    v_oc : array
        Open circuit voltage for each test condition [V]
    i_mp : array
        Current at maximum power point for each test condition [A]
    v_mp : array
        Voltage at maximum power point for each test condition [V]
    cells_in_series : int
        The number of cells connected in series.
    EgRef : float, optional
        The energy bandgap at reference temperature in units of eV.
        1.121 eV for crystalline silicon. EgRef must be >0.
    alpha_sc : float, optional
        Temperature coefficient of short circuit current.  If not specified,
        it will be estimated using the ``i_sc`` values at irradiance of
        1000 W/m2. [A/K]
    beta_mp : float, optional
        Temperature coefficient of maximum power voltage.  If not specified,
        it will be estimated using the ``v_mp`` values at irradiance of
        1000 W/m2. [1/K]
    R_s : float, optional
        Series resistance value.  If not provided, a value will be estimated
        from the input measurements. [ohm]
    r_sh_coeff : float, default 0.12
        Shunt resistance fitting coefficient.  The default value is taken
        from [1]_.
    min_Rsh_irradiance : float, optional
        Irradiance threshold below which values are excluded when estimating
        shunt resistance parameter values.  May be useful for modules
        with problematic low-light measurements. [W/m²]
    irradiance_tolerance : float, default 20
        Tolerance for irradiance variation around the STC value.
        The default value corresponds to a +/- 2% interval around the STC
        value of 1000 W/m². [W/m²]
    temperature_tolerance : float, default 1
        Tolerance for temperature variation around the STC value.
        The default value corresponds to a +/- 1 degree interval around the STC
        value of 25°C. [°C]

    Returns
    -------
    dict
        alpha_sc : float
            short circuit current temperature coefficient [A/K]
        gamma_ref : float
            diode (ideality) factor at STC [unitless]
        mu_gamma : float
            temperature coefficient for diode (ideality) factor [1/K]
        I_L_ref : float
            light current at STC [A]
        I_o_ref : float
            dark current at STC [A]
        R_sh_ref : float
            shunt resistance at STC [ohm]
        R_sh_0 : float
            shunt resistance at zero irradiance [ohm]
        R_sh_exp : float
            exponential factor defining decrease in shunt resistance with
            increasing effective irradiance
        R_s : float
            series resistance at STC [ohm]
        cells_in_series : int
            number of cells in series
        EgRef : float
            effective band gap at STC [eV]

    See also
    --------
    pvlib.pvsystem.calcparams_pvsyst
    pvlib.ivtools.sdm.fit_pvsyst_sandia

    Notes
    -----
    Input arrays of operating conditions and electrical measurements must be
    1-D with equal lengths.

    Values supplied for ``alpha_sc``, ``beta_mp``, and ``R_s`` must be
    consistent with the matrix data, as these values are used when estimating
    other model parameters.

    This method is non-iterative.  In some cases, it may be desirable to
    refine the estimated parameter values using a numerical optimizer such as
    the default method in ``scipy.optimize.minimize``.

    References
    ----------
    .. [1] K. S. Anderson, C. W. Hansen, and M. Theristis, "A Noniterative
       Method of Estimating Parameter Values for the PVsyst Version 6
       Single-Diode Model From IEC 61853-1 Matrix Measurements," IEEE Journal
       of Photovoltaics, vol. 15, 3, 2025. :doi:`10.1109/JPHOTOV.2025.3554338`
    """

    is_g_stc = np.isclose(effective_irradiance, 1000, rtol=0,
                          atol=irradiance_tolerance)
    is_t_stc = np.isclose(temp_cell, 25, rtol=0,
                          atol=temperature_tolerance)

    if alpha_sc is None:
        mu_i_sc = _fit_tempco_pvsyst_iec61853_sandia_2025(i_sc[is_g_stc],
                                                          temp_cell[is_g_stc])
        i_sc_ref = float(i_sc[is_g_stc & is_t_stc].item())
        alpha_sc = mu_i_sc * i_sc_ref

    if beta_mp is None:
        beta_mp = _fit_tempco_pvsyst_iec61853_sandia_2025(v_mp[is_g_stc],
                                                          temp_cell[is_g_stc])

    R_sh_ref, R_sh_0, R_sh_exp = \
        _fit_shunt_resistances_pvsyst_iec61853_sandia_2025(
            i_sc, i_mp, v_mp, effective_irradiance, temp_cell, beta_mp,
            coeff=r_sh_coeff, min_irradiance=min_Rsh_irradiance)

    if R_s is None:
        R_s = _fit_series_resistance_pvsyst_iec61853_sandia_2025(v_oc, i_mp,
                                                                 v_mp)

    gamma_ref, mu_gamma = \
        _fit_diode_ideality_factor_pvsyst_iec61853_sandia_2025(
            i_sc[is_t_stc], v_oc[is_t_stc], i_mp[is_t_stc], v_mp[is_t_stc],
            effective_irradiance[is_t_stc], temp_cell[is_t_stc],
            R_sh_ref, R_sh_0, R_sh_exp, R_s, cells_in_series)

    I_o_ref = _fit_saturation_current_pvsyst_iec61853_sandia_2025(
        i_sc, v_oc, effective_irradiance, temp_cell, gamma_ref, mu_gamma,
        R_sh_ref, R_sh_0, R_sh_exp, R_s, cells_in_series, EgRef
    )

    I_L_ref = _fit_photocurrent_pvsyst_iec61853_sandia_2025(
        i_sc, effective_irradiance, temp_cell, alpha_sc,
        gamma_ref, mu_gamma,
        I_o_ref, R_sh_ref, R_sh_0, R_sh_exp, R_s, cells_in_series, EgRef
    )

    gamma_ref, mu_gamma = \
        _fit_diode_ideality_factor_post_pvsyst_iec61853_sandia_2025(
            i_mp, v_mp, effective_irradiance, temp_cell, alpha_sc, I_L_ref,
            I_o_ref, R_sh_ref, R_sh_0, R_sh_exp, R_s, cells_in_series, EgRef)

    fitted_params = dict(
        alpha_sc=alpha_sc,
        gamma_ref=gamma_ref,
        mu_gamma=mu_gamma,
        I_L_ref=I_L_ref,
        I_o_ref=I_o_ref,
        R_sh_ref=R_sh_ref,
        R_sh_0=R_sh_0,
        R_sh_exp=R_sh_exp,
        R_s=R_s,
        cells_in_series=cells_in_series,
        EgRef=EgRef,
    )
    return fitted_params


def _fit_tempco_pvsyst_iec61853_sandia_2025(values, temp_cell,
                                            temp_cell_ref=25):
    fit = np.polynomial.polynomial.Polynomial.fit(temp_cell, values, deg=1)
    intercept, slope = fit.convert().coef
    value_ref = intercept + slope*temp_cell_ref
    return slope / value_ref


def _fit_shunt_resistances_pvsyst_iec61853_sandia_2025(
        i_sc, i_mp, v_mp, effective_irradiance, temp_cell,
        beta_v_mp, coeff=0.2, min_irradiance=None):
    if min_irradiance is None:
        min_irradiance = 0

    mask = effective_irradiance >= min_irradiance
    i_sc = i_sc[mask]
    i_mp = i_mp[mask]
    v_mp = v_mp[mask]
    effective_irradiance = effective_irradiance[mask]
    temp_cell = temp_cell[mask]

    # Equation 10
    Rsh_est = (
        (v_mp / (1 + beta_v_mp * (temp_cell - 25)))
        / (coeff * (i_sc - i_mp))
    )
    Rshexp = 5.5

    # Eq 11
    y = Rsh_est
    x = np.exp(-Rshexp * effective_irradiance / 1000)

    fit = np.polynomial.polynomial.Polynomial.fit(x, y, deg=1)
    intercept, slope = fit.convert().coef
    Rshbase = intercept
    Rsh0 = slope + Rshbase

    # Eq 12
    expRshexp = np.exp(-Rshexp)
    Rshref = Rshbase * (1 - expRshexp) + Rsh0 * expRshexp

    return Rshref, Rsh0, Rshexp


def _fit_series_resistance_pvsyst_iec61853_sandia_2025(v_oc, i_mp, v_mp):
    # Stein et al 2014, https://doi.org/10.1109/PVSC.2014.6925326

    # Eq 13
    x = np.array([np.ones(len(i_mp)), i_mp, np.log(i_mp), v_mp]).T
    y = v_oc

    coeff, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    R_s = coeff[1]
    return R_s


def _fit_diode_ideality_factor_pvsyst_iec61853_sandia_2025(
        i_sc, v_oc, i_mp, v_mp, effective_irradiance, temp_cell,
        R_sh_ref, R_sh_0, R_sh_exp, R_s, cells_in_series):

    NsVth = _pvsyst_nNsVth(temp_cell, gamma=1, cells_in_series=cells_in_series)
    Rsh = _pvsyst_Rsh(effective_irradiance, R_sh_ref, R_sh_0, R_sh_exp)
    term1 = (i_sc * (1 + R_s/Rsh) - v_oc / Rsh)  # Eq 15
    term2 = (i_sc - i_mp) * (1 + R_s/Rsh) - v_mp / Rsh  # Eq 16

    # Eq 14
    x1 = NsVth * np.log(term2 / term1)

    x = np.array([x1]).T
    y = v_mp + i_mp*R_s - v_oc

    coeff, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    gamma_ref = coeff[0]
    return gamma_ref, 0


def _fit_saturation_current_pvsyst_iec61853_sandia_2025(
        i_sc, v_oc, effective_irradiance, temp_cell, gamma_ref, mu_gamma,
        R_sh_ref, R_sh_0, R_sh_exp, R_s, cells_in_series, EgRef):
    R_sh = _pvsyst_Rsh(effective_irradiance, R_sh_ref, R_sh_0, R_sh_exp)
    gamma = _pvsyst_gamma(temp_cell, gamma_ref, mu_gamma)
    nNsVth = _pvsyst_nNsVth(temp_cell, gamma, cells_in_series)

    # Eq 17
    I_o_est = (i_sc * (1 + R_s/R_sh) - v_oc/R_sh) / (np.expm1(v_oc / nNsVth))
    x = _pvsyst_Io(temp_cell, gamma, I_o_ref=1, EgRef=EgRef)

    # Eq 18
    log_I_o_ref = np.mean(np.log(I_o_est) - np.log(x))
    I_o_ref = np.exp(log_I_o_ref)

    return I_o_ref


def _fit_photocurrent_pvsyst_iec61853_sandia_2025(
        i_sc, effective_irradiance, temp_cell, alpha_sc, gamma_ref,
        mu_gamma, I_o_ref, R_sh_ref, R_sh_0, R_sh_exp, R_s, cells_in_series,
        EgRef):
    R_sh = _pvsyst_Rsh(effective_irradiance, R_sh_ref, R_sh_0, R_sh_exp)
    gamma = _pvsyst_gamma(temp_cell, gamma_ref, mu_gamma)
    I_o = _pvsyst_Io(temp_cell, gamma, I_o_ref, EgRef)
    nNsVth = _pvsyst_nNsVth(temp_cell, gamma, cells_in_series)

    # Eq 19
    I_L_est = i_sc + I_o * (np.expm1(i_sc * R_s / nNsVth)) + i_sc * R_s / R_sh

    # Eq 20
    x = np.array([effective_irradiance / 1000]).T
    y = I_L_est - effective_irradiance / 1000 * alpha_sc * (temp_cell - 25)
    coeff, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    I_L_ref = coeff[0]
    return I_L_ref


def _fit_diode_ideality_factor_post_pvsyst_iec61853_sandia_2025(
        i_mp, v_mp, effective_irradiance, temp_cell, alpha_sc, I_L_ref,
        I_o_ref, R_sh_ref, R_sh_0, R_sh_exp, R_s, cells_in_series, EgRef):

    Rsh = _pvsyst_Rsh(effective_irradiance, R_sh_ref, R_sh_0, R_sh_exp)
    I_L = _pvsyst_IL(effective_irradiance, temp_cell, I_L_ref, alpha_sc)
    NsVth = _pvsyst_nNsVth(temp_cell, gamma=1, cells_in_series=cells_in_series)

    Tref_K = 25 + 273.15
    Tcell_K = temp_cell + 273.15

    # Eq 21
    k = constants.k  # Boltzmann constant in J/K
    q = constants.e  # elementary charge in coulomb
    numerator = (
        (q * EgRef / k) * (1/Tref_K - 1/Tcell_K)
        + (v_mp + i_mp*R_s) / NsVth
    )
    denominator = (
        np.log((I_L - i_mp - (v_mp+i_mp*R_s) / Rsh) / I_o_ref)
        - 3 * np.log(Tcell_K / Tref_K)
    )
    gamma_est = numerator / denominator

    # Eq 22
    x = np.array([np.ones(len(i_mp)), temp_cell - 25]).T
    y = gamma_est

    coeff, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    gamma_ref, mu_gamma = coeff
    return gamma_ref, mu_gamma
