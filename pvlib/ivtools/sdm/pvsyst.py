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
            cell temperature for each IV curve [C]
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
            effective irradiance at STC, default 1000 [W/m^2]
        T0 : float
            cell temperature at STC, default 25 [C]
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
        Reference irradiance. [W/m^2].

    temp_ref : float, default 25
        Reference cell temperature. [C]


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
