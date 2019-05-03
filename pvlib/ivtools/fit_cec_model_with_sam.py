# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:42:24 2019

@author: cwhanse
"""

from PySSC import PySSC

def fit_cec_model_with_sam(sam_dir, celltype, Vmp, Imp, Voc, Isc, alpha_sc,
                           beta_voc, gamma_pmp, cells_in_series, temp_ref=25):
    '''
    Estimates parameters for the CEC single diode model using SAM SDK.

    Parameters
    ----------
    sam_dir : str
        Full path to folder containing the SAM file ssc.dll
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

    try:
        ssc = PySSC(sam_dir)
    except Exception as e:
        raise(e)

    data = ssc.data_create()

    ssc.data_set_string(data, b'celltype', celltype.encode('utf-8'))
    ssc.data_set_number(data, b'Vmp', Vmp)
    ssc.data_set_number(data, b'Imp', Imp)
    ssc.data_set_number(data, b'Voc', Voc)
    ssc.data_set_number(data, b'Isc', Isc)
    ssc.data_set_number(data, b'alpha_isc', alpha_sc)
    ssc.data_set_number(data, b'beta_voc', beta_voc)
    ssc.data_set_number(data, b'gamma_pmp', gamma_pmp)
    ssc.data_set_number(data, b'Nser', cells_in_series)
    ssc.data_set_number(data, b'Tref', temp_ref)

    solver = ssc.module_create(b'6parsolve')
    ssc.module_exec_set_print(0)
    if ssc.module_exec(solver, data) == 0:
        print('IV curve fit error')
        idx = 1
        msg = ssc.module_log(solver, 0)
        while (msg != None):
            print('	: ' + msg.decode("utf - 8"))
            msg = ssc.module_log(solver, idx)
            idx = idx + 1
    ssc.module_free(solver)
    a_ref = ssc.data_get_number(data, b'a')
    I_L_ref = ssc.data_get_number(data, b'Il')
    I_o_ref = ssc.data_get_number(data, b'Io')
    R_s = ssc.data_get_number(data, b'Rs')
    R_sh_ref = ssc.data_get_number(data, b'Rsh')
    Adjust = ssc.data_get_number(data, b'Adj')

    return a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust
