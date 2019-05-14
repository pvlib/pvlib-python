# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:34:10 2019

@author: cwhanse
"""

import numpy as np


def fit_cec_with_sam(celltype, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc,
                     gamma_pmp, cells_in_series, temp_ref=25):
    '''
    Estimates parameters for the CEC single diode model using the SAM SDK.

    Parameters
    ----------
    celltype : str
        Value is one of 'monoSi', 'multiSi', 'polySi', 'cis', 'cigs', 'cdte',
        'amorphous'
    v_mp : float
        Voltage at maximum power point at standard test condition (STC)
    i_mp : float
        Current at maximum power point at STC
    Voc : float
        Open circuit voltage at STC
    i_sc : float
        Short circuit current at STC
    alpha_sc : float
        Temperature coefficient of short circuit current at STC, A/C
    beta_voc : float
        Temperature coefficient of open circuit voltage at STC, V/C
    gamma_pmp : float
        Temperature coefficient of power at maximum point point at STC, %/C
    cells_in_series : int
        Number of cells in series
    temp_ref : float, default 25
        Reference temperature condition

    Returns
    -------
    a_ref : float
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at reference
        conditions, in units of V.

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.

    Adjust : float
        The adjustment to the temperature coefficient for short circuit
        current, in percent

    Raises:
        ImportError if NREL-PySAM is not installed
    '''

    try:
        from PySAM import PySSC
    except ImportError as e:
        raise(e)

    datadict = {"tech_model": '6parsolve', 'celltype': celltype, 'Vmp': v_mp,
                'Imp': i_mp, 'Voc': v_oc, 'Isc': i_sc, 'alpha_isc': alpha_sc,
                'beta_voc': beta_voc, 'gamma_pmp': gamma_pmp,
                'Nser': cells_in_series, 'Tref': temp_ref}

    result = PySSC.ssc_sim_from_dict(datadict)
    a_ref = result['a']
    I_L_ref = result['Il']
    I_o_ref = result['Io']
    R_s = result['Rs']
    R_sh_ref = result['Rsh']
    Adjust = result['Adj']

    return I_L_ref, I_o_ref, R_sh_ref, R_s, a_ref, Adjust


def fit_sde_sandia(v, i, v_oc, i_sc, v_mp, i_mp, vlim=0.2, ilim=0.1):
    """ Fits the single diode equation to an IV curve.

    If fitting fails, returns NaN in each parameter.

    Parameters
    ----------
    v : numeric
        Voltage at each point on the IV curve, from 0 to v_oc

    i : numeric
        Current at each point on the IV curve, from i_sc to 0

    v_oc : float
        Open circuit voltage

    i_sc : float
        Short circuit current

    v_mp : float
        Voltage at maximum power point

    i_mp : float
        Current at maximum power point

    vlim : float, default 0.2
        defines linear portion of IV curve i.e. V <= vlim * v_oc

    ilim : float, default 0.1
        defines exponential portion of IV curve i.e. I > ilim * i_sc

    Returns
    -------
    tuple of the following elements:

        photocurrent : float
            photocurrent, A

        saturation_current : float
            dark (saturation) current, A

        resistance_shunt : float
            shunt (parallel) resistance, ohm

        resistance_series : float
            series resistance, ohm

        nNsVth : float
            product of diode (ideality) factor n (unitless) x number of
            cells in series Ns (unitless) x cell thermal voltage Vth (V), V

    References
    ----------
    [1] C. B. Jones, C. W. Hansen, Single Diode Parameter Extraction from
    In-Field Photovoltaic I-V Curves on a Single Board Computer, 46th IEEE
    Photovoltaic Specialist Conference, Chicago, IL, 2019
    """
    # Find intercept and slope of linear portion of IV curve.
    # Start with V < vlim * v_oc, extend by adding points until slope is
    # acceptable
    beta = [np.nan for i in range(5)]
    # get index of largest voltage less than/equal to limit
    idx = _max_index(v, vlim * v_oc)
    while np.isnan(beta[1]) and (idx <= len(v)):
        try:
            coef = np.polyfit(v[:idx], i[:idx], deg=1)
            if coef[0] < 0:
                # intercept term
                beta[0] = coef[1].item()
                # sign change of slope to get positive parameter value
                beta[1] = -coef[0].item()
        except Exception as e:
            raise(e)
        if np.isnan(beta[1]):
            idx += 1

    if not np.isnan(beta[0]):
        # Find parameters from exponential portion of IV curve
        y = beta[0] - beta[1] * v - i
        x = np.array([np.ones_like(v), v, i]).T
        idx = _min_index(y, ilim * i_sc)
        try:
            result = np.linalg.lstsq(x[idx:, ], np.log(y[idx:]))
            coef = result[0]
            beta[3] = coef[1].item()
            beta[4] = coef[2].item()
        except Exception as e:
            raise(e)

    if not any([np.isnan(beta[i]) for i in [0, 1, 3, 4]]):
        # calculate parameters
        nNsVth = 1.0 / beta[3]
        Rs = beta[4] / beta[3]
        Gp = beta[1] / (1.0 - Rs * beta[1])
        Rsh = 1.0 / Gp
        IL = (1 + Gp * Rs) * beta[0]
        # calculate I0
        I0_v_mp = _calc_I0(IL, i_mp, v_mp, Gp, Rs, beta[3])
        I0_v_oc = _calc_I0(IL, 0, v_oc, Gp, Rs, beta[3])
        if (I0_v_mp > 0) and (I0_v_oc > 0):
            I0 = 0.5 * (I0_v_mp + I0_v_oc)
        elif (I0_v_mp > 0):
            I0 = I0_v_mp
        elif (I0_v_oc > 0):
            I0 = I0_v_oc
        else:
            I0 = np.nan
    else:
        IL = I0 = Rs = Rsh = nNsVth = np.nan

    return IL, I0, Rsh, Rs, nNsVth


def _calc_I0(IL, I, V, Gp, Rs, beta3):
    return (IL - I - Gp * V - Gp * Rs * I) / np.exp(beta3 * (V + Rs * I))


def _max_index(x, xlim):
    """ Finds maximum index of value of x <= xlim """
    return int(np.argwhere(x <= xlim)[-1])


def _min_index(x, xlim):
    """ Finds minimum index of value of x > xlim """
    return int(np.argwhere(x > xlim)[0])
