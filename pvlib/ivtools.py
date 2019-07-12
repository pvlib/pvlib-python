# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:34:10 2019

@author: cwhanse
"""

import numpy as np


def fit_cec_sam(celltype, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc,
                gamma_pmp, cells_in_series, temp_ref=25):
    '''
    Estimates parameters for the CEC single diode model [1] using the SAM SDK.

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
    temp_ref : float, default 25C
        Reference temperature condition

    Returns
    -------
    tuple of the following elements:

        a_ref : float
            The product of the usual diode ideality factor ``n`` (unitless),
            number of cells in series ``Ns``, and cell thermal voltage at
            reference conditions [V]

        I_L_ref : float
            The light-generated current (or photocurrent) at reference
            conditions [A]

        I_o_ref : float
            The dark or diode reverse saturation current at reference
            conditions [A]

        R_sh_ref : float
            The shunt resistance at reference conditions, in ohms.

        R_s : float
            The series resistance at reference conditions, in ohms.

        Adjust : float
            The adjustment to the temperature coefficient for short circuit
            current, in percent.

    Raises:
        ImportError if NREL-PySAM is not installed.
        RuntimeError if parameter extraction is not successful.

    Notes
    -----
    Inputs ``v_mp``, ``v_oc``, ``i_mp`` and ``i_sc`` are assumed to be from a
    single IV curve at constant irradiance and cell temperature. Irradiance is
    not explicitly used by the fitting procedure. The irradiance level at which
    the input IV curve is determined and the specified cell temperature
    ``Tref`` are the reference conditions for the output parameters
    ``I_L_ref``, ``I_o_ref``, ``R_sh_ref``, ``R_s`` and ``Adjust``.

    References
    ----------
    [1] A. Dobos, "An Improved Coefficient Calculator for the California
    Energy Commission 6 Parameter Photovoltaic Module Model", Journal of
    Solar Energy Engineering, vol 134, 2012.
    '''

    try:
        from PySAM import PySSC
    except ImportError as e:
        raise("Requires NREL's PySAM package at "
              "https://pypi.org/project/NREL-PySAM/.") from e

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


def fit_sde_sandia(v, i, v_oc=None, i_sc=None, mp=None, vlim=0.2, ilim=0.1):
    """ Fits the single diode equation to an IV curve.

    Parameters
    ----------
    v : ndarray
        1D array of `float` type containing voltage at each point on the IV
        curve, increasing from 0 to v_oc inclusive [V]

    i : ndarray
        1D array of `float` type containing current at each point on the IV
        curve, from i_sc to 0 inclusive, of `float` type [A]

    v_oc : float, default None
        Open circuit voltage [V]. If not provided, ``v_oc`` is taken as
        ``v[-1]``.

    i_sc : float, default None
        Short circuit current [A]. If not provided, ``i_sc`` is taken as
        ``i[0]``.

    mp : tuple of float, default None
        Voltage, current at maximum power point in units of [V], [A].
        If not provided, values are found from inputs ``v`` and ``i``.

    vlim : float, default 0.2
        defines portion of IV curve where the exponential term in the single
        diode equation can be neglected, i.e. V <= vlim * v_oc [V]

    ilim : float, default 0.1
        defines portion of the IV curve where the exponential term in the
        single diode equation is signficant, approximately defined by
        I < (1 - ilim) * i_sc [A]

    Returns
    -------
    tuple of the following elements:

        photocurrent : float
            photocurrent [A]

        saturation_current : float
            dark (saturation) current [A]

        resistance_shunt : float
            shunt (parallel) resistance, ohm

        resistance_series : float
            series resistance, ohm

        nNsVth : float
            product of thermal voltage ``Vth`` [V], diode ideality factor
            ``n``, and number of series cells ``Ns``

    Raises:
        RuntimeError if parameter extraction is not successful.

    Notes
    -----
    Inputs ``v``, ``i``, ``v_mp``, ``v_oc``, ``i_mp`` and ``i_sc`` are assumed
    to be from a single IV curve at constant irradiance and cell temperature.

    :py:func:`fit_sde_sandia` obtains values for the five parameters for the
    single diode equation [1]:

    .. math::

        I = IL - I0*[exp((V+I*Rs)/(nNsVth))-1] - (V + I*Rs)/Rsh

    :py:func:`pvsystem.singlediode` for definition of the parameters.

    The extraction method [2] proceeds in four steps:
        1) In the single diode equation, replace Rsh = 1/Gp and re-arrange

    .. math::

        I = IL - I0*exp((V+I*Rs)/(nNsVth) - 1) - (V + I*Rs)/Rsh

    .. math::

        I = IL/(1+Gp*Rs) - (Gp*V)/(1+Gp*Rs) -
          I0/(1+Gp*Rs)*exp((V+I*Rs)/(nNsVth) - 1)

        2) fit the linear portion of the IV curve defined as V <= vlim * v_oc,
            where I0/(1+Gp*Rs)*exp((V+I*Rs)/(nNsVth) - 1) ≈ 0

    .. math::

        I ~ IL/(1+Gp*Rs) - (Gp*V)/(1+Gp*Rs) = beta0 + beta1*V

        3) fit the exponential portion of the IV curve defined by
            beta0 + beta1*V - I > ilim * i_sc, where
            exp((V+I*Rs)/(nNsVth)) >> 1 so that
            exp((V+I*Rs)/(nNsVth)) - 1 ≈ exp((V+I*Rs)/(nNsVth))

    .. math::

        log(beta0 - beta1*V - I) ~ log((I0)/(1+Gp*Rs)) + (V)/(nNsVth) +
        (Rs*I)/(nNsVth) = beta2 + beta3*V + beta4*I

        4) calculate values for ``IL, I0, Rs, Rsh,`` and ``nNsVth`` from the
            regression coefficents beta0, beta1, beta3 and beta4.


    References
    ----------
    [1] S.R. Wenham, M.A. Green, M.E. Watt, "Applied Photovoltaics" ISBN
    0 86758 909 4
    [2] C. B. Jones, C. W. Hansen, Single Diode Parameter Extraction from
    In-Field Photovoltaic I-V Curves on a Single Board Computer, 46th IEEE
    Photovoltaic Specialist Conference, Chicago, IL, 2019
    """

    # If not provided, extract v_oc, i_sc, v_mp and i_mp from the IV curve data
    if v_oc is None:
        v_oc = v[-1]
    if i_sc is None:
        i_sc = i[0]
    if mp is not None:
        v_mp, i_mp = mp
    else:
        v_mp, i_mp = _find_mp(v, i)

    # Find beta0 and beta1 from linear portion of the IV curve
    beta0, beta1 = _find_beta0_beta1(v, i, vlim, v_oc)

    if not np.isnan(beta0):
        beta3, beta4 = _find_beta3_beta4(v, i, beta0, beta1, ilim, i_sc)

    # calculate single diode parameters from regression coefficients
    result = _calculate_sde_parameters(beta0, beta1, beta3, beta4, v_mp, i_mp,
                                       v_oc)

    if result is None:
        raise RuntimeError("Parameter extraction failed. Try increasing the"
                           "number of data points in the IV curve")
    else:
        IL, I0, Rsh, Rs, nNsVth = result
        return IL, I0, Rsh, Rs, nNsVth


def _find_mp(v, i):
    """
    Finds voltage and current at maximum power point.

    Parameters
    ----------
    v : ndarray
        1D array containing voltage at each point on the IV curve, increasing
        from 0 to v_oc inclusive, of `float` type [V]

    i : ndarray
        1D array containing current at each point on the IV curve, decreasing
        from i_sc to 0 inclusive, of `float` type [A]

    Returns
    -------
    v, i : tuple
        voltage ``v`` and current ``i`` at the maximum power point [V], [A]
    """
    p = v * i
    idx = np.argmax(p)
    return v[idx], i[idx]


def _calc_I0(IL, I, V, Gp, Rs, beta3):
    return (IL - I - Gp * V - Gp * Rs * I) / np.exp(beta3 * (V + Rs * I))


def _find_beta0_beta1(v, i, vlim, v_oc):
    # Get intercept and slope of linear portion of IV curve.
    # Start with V =< vlim * v_oc, extend by adding points until slope is
    # negative (downward).
    beta0 = np.nan
    beta1 = np.nan
    idx = np.searchsorted(v, vlim * v_oc)
    while idx <= len(v):
        coef = np.polyfit(v[:idx], i[:idx], deg=1)
        if coef[0] < 0:
            # intercept term
            beta0 = coef[1].item()
            # sign change of slope to get positive parameter value
            beta1 = -coef[0].item()
            break
        else:
            idx += 1
    return beta0, beta1


def _find_beta3_beta4(v, i, beta0, beta1, ilim, i_sc):
    # Subtract the IV curve from the linear fit.
    y = beta0 - beta1 * v - i
    x = np.array([np.ones_like(v), v, i]).T
    # Select points where y > ilim * i_sc to regress log(y) onto x
    result = np.linalg.lstsq(x[y > ilim * i_sc], np.log(y[y > ilim * i_sc]),
                             rcond=None)
    coef = result[0]
    beta3 = coef[1].item()
    beta4 = coef[2].item()
    return beta3, beta4


def _calculate_sde_parameters(beta0, beta1, beta3, beta4, v_mp, i_mp, v_oc):
    failed = False
    if any(np.isnan([beta0, beta1, beta3, beta4])):
        failed = True
    else:
        nNsVth = 1.0 / beta3
        Rs = beta4 / beta3
        Gp = beta1 / (1.0 - Rs * beta1)
        Rsh = 1.0 / Gp
        IL = (1 + Gp * Rs) * beta0
        # calculate I0
        I0_vmp = _calc_I0(IL, i_mp, v_mp, Gp, Rs, beta3)
        I0_voc = _calc_I0(IL, 0, v_oc, Gp, Rs, beta3)
        if (I0_vmp > 0) and (I0_voc > 0):
            I0 = 0.5 * (I0_vmp + I0_voc)
        elif (I0_vmp > 0):
            I0 = I0_vmp
        elif (I0_voc > 0):
            I0 = I0_voc
        else:
            failed = True
    if failed:
        result = None
    else:
        result = (IL, I0, Rsh, Rs, nNsVth)
    return result
