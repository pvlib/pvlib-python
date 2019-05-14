# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:34:10 2019

@author: cwhanse
"""

import numpy as np
from PySAM import PySSC


def fit_cec_with_sam(celltype, Vmp, Imp, Voc, Isc, alpha_sc, beta_voc,
                     gamma_pmp, cells_in_series, temp_ref=25):
    '''
    Estimates parameters for the CEC single diode model using the SAM SDK.

    Parameters
    ----------
    celltype : str
        Value is one of 'monoSi', 'multiSi', 'polySi', 'cis', 'cigs', 'cdte',
        'amorphous'
    Vmp : float
        Voltage at maximum power point at standard test condition (STC)
    Imp : float
        Current at maximum power point at STC
    Voc : float
        Open circuit voltage at STC
    Isc : float
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
    '''

    datadict = {"tech_model": '6parsolve', 'celltype': celltype, 'Vmp': Vmp,
                'Imp': Imp, 'Voc': Voc, 'Isc': Isc, 'alpha_isc': alpha_sc,
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


def fit_sde_sandia(V, I, Voc, Isc, Vmp, Imp, vlim=0.2, ilim=0.1):
    """ Fits the single diode equation to an IV curve.

    If fitting fails, returns NaN in each parameter.

    Parameters
    ----------
    V : numeric
        Voltage at each point on the IV curve, from 0 to Voc

    I : numeric
        Current at each point on the IV curve, from Isc to 0

    Voc : float
        Open circuit voltage

    Isc : float
        Short circuit current

    Vmp : float
        Voltage at maximum power point

    Imp : float
        Current at maximum power point

    vlim : float, default 0.2
        defines linear portion of IV curve i.e. V <= vlim * Voc

    ilim : float, default 0.1
        defines exponential portion of IV curve i.e. I > ilim * Isc

    Returns
    -------
    IL : float
        photocurrent, A

    I0 : float
        dark (saturation) current, A

    Rsh : float
        shunt (parallel) resistance, ohm

    Rs : float
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
    # Start with V < vlim * Voc, extend by adding points until slope is
    # acceptable
    beta = [np.nan for i in range(5)]
    # get index of largest voltage less than/equal to limit
    idx = _max_index(V, vlim * Voc)
    while np.isnan(beta[1]) and (idx <= len(V)):
        try:
            coef = np.polyfit(V[:idx], I[:idx], deg=1)
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
        Y = beta[0] - beta[1] * V - I
        X = np.array([np.ones_like(V), V, I]).T
        idx = _min_index(Y, ilim * Isc)
        try:
            result = np.linalg.lstsq(X[idx:, ], np.log(Y[idx:]))
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
        I0_Vmp = _calc_I0(IL, Imp, Vmp, Gp, Rs, beta[3])
        I0_Voc = _calc_I0(IL, 0, Voc, Gp, Rs, beta[3])
        if (I0_Vmp > 0) and (I0_Voc > 0):
            I0 = 0.5 * (I0_Vmp + I0_Voc)
        elif (I0_Vmp > 0):
            I0 = I0_Vmp
        elif (I0_Voc > 0):
            I0 = I0_Voc
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
