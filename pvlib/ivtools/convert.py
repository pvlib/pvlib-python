"""
The ``convert`` module contains functions to convert between single diode
models.

Function names should follow the pattern "convert_" + name of source model +
 "_" + name of target method + "_" + name of conversion method.

"""

import numpy as np
import pandas as pd

from scipy import constants
from scipy import optimize

from pvlib.pvsystem import (calcparams_pvsyst, calcparams_cec, singlediode)


CONSTANTS = {'E0': 1000.0, 'T0': 25.0, 'k': constants.k, 'q': constants.e}


IEC61853 = pd.DataFrame(
    columns=['effective_irradiance', 'temp_cell'],
    data=np.array(
        [[100, 100, 100, 100, 200, 200, 200, 200, 400, 400, 400, 400,
          600, 600, 600, 600, 800, 800, 800, 800, 1000, 1000, 1000, 1000,
          1100, 1100, 1100, 1100],
         [15, 25, 50, 75, 15, 25, 50, 75, 15, 25, 50, 75, 15, 25, 50, 75,
          15, 25, 50, 75, 15, 25, 50, 75, 15, 25, 50, 75]]).T,
    dtype=np.float64)



def _pvsyst_objfun(pvs_mod, cec_ivs, ee, tc, cs):

    # translate the guess into named args that are used in the functions
    # order : [alpha_sc, gamma_ref, mu_gamma, I_L_ref, I_o_ref,
    # R_sh_mult, R_sh_ref, R_s]
    # cec_ivs : DataFrame with columns i_sc, v_oc, i_mp, v_mp, p_mp
    # ee : effective irradiance
    # tc : cell temperature
    # cs : cells in series
    alpha_sc = pvs_mod[0]
    gamma_ref = pvs_mod[1]
    mu_gamma = pvs_mod[2]
    I_L_ref = pvs_mod[3]
    I_o_ref = pvs_mod[4]
    R_sh_mult = pvs_mod[5]
    R_sh_ref = pvs_mod[6]
    R_s = pvs_mod[7]

    R_sh_0 = R_sh_ref * R_sh_mult

    pvs_params = calcparams_pvsyst(
        ee, tc, alpha_sc, gamma_ref, mu_gamma, I_L_ref, I_o_ref, R_sh_ref,
        R_sh_0, R_s, cs)

    pvsyst_ivs = singlediode(*pvs_params)

    isc_diff = np.abs((pvsyst_ivs['i_sc'] - cec_ivs['i_sc']) /
                      cec_ivs['i_sc']).mean()
    imp_diff = np.abs((pvsyst_ivs['i_mp'] - cec_ivs['i_mp']) /
                      cec_ivs['i_mp']).mean()
    voc_diff = np.abs((pvsyst_ivs['v_oc'] - cec_ivs['v_oc']) /
                      cec_ivs['v_oc']).mean()
    vmp_diff = np.abs((pvsyst_ivs['v_mp'] - cec_ivs['v_mp']) /
                      cec_ivs['v_mp']).mean()
    pmp_diff = np.abs((pvsyst_ivs['p_mp'] - cec_ivs['p_mp']) /
                      cec_ivs['p_mp']).mean()

    mean_abs_diff = (isc_diff + imp_diff + voc_diff + vmp_diff + pmp_diff) / 5

    return mean_abs_diff


def convert_cec_pvsyst(cec_model, cells_in_series, method='Nelder-Mead',
                       options=None):
    r"""
    Convert a set of CEC model parameters to an equivalent set of PVsyst model
    parameters.

    Parameter conversion uses optimization as described in [1]_ to fit the
    PVsyst model to :math:`I_{sc}`, :math:`V_{oc}`, :math:`V_{mp}`,
    :math:`I_{mp}`, and :math:`P_{mp}`, calculated using the input CEC model
    at the IEC 61853-3 conditions [2]_.

    Parameters
    ----------
    cec_model : dict or DataFrame
        Must include keys: 'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_s', 'Adjust'
    cell_in_series : int
        Number of cells in series.
    method : str, default 'Nelder-Mead'
        Method for scipy.optimize.minimize.
    options : dict, optional
        Solver options passed to scipy.optimize.minimize.

    Returns
    -------
    dict with the following elements:
        alpha_sc : float
            Short-circuit current temperature coefficient [A/C] .
        I_L_ref : float
            The light-generated current (or photocurrent) at reference
            conditions [A].
        I_o_ref : float
            The dark or diode reverse saturation current at reference
            conditions [A].
        EgRef : float
            The energy bandgap at reference temperature [eV].
        R_s : float
            The series resistance at reference conditions [ohm].
        R_sh_ref : float
            The shunt resistance at reference conditions [ohm].
        R_sh_0 : float
            Shunt resistance at zero irradiance [ohm].
        R_sh_exp : float
            Exponential factor defining decrease in shunt resistance with
            increasing effective irradiance [unitless].
        gamma_ref : float
            Diode (ideality) factor at reference conditions [unitless].
        mu_gamma : float
            Temperature coefficient for diode (ideality) factor at reference
            conditions [1/K].
        cells_in_series : int
            Number of cells in series.

    Notes
    -----
    Reference conditions are irradiance of 1000 W/m⁻² and cell temperature of
    25 °C.

    See Also
    --------
    pvlib.ivtools.convert.convert_pvsyst_cec

    References
    ----------
    .. [1] L. Deville et al., "Parameter Translation for Photovoltaic Single
       Diode Models", Journal of Photovoltaics, vol. 15(3), pp. 451-457,
       May 2025. :doi:`10.1109/jphotov.2025.3539319`

    .. [2] "IEC 61853-3 Photovoltaic (PV) module performance testing and energy
       rating - Part 3: Energy rating of PV modules". IEC, Geneva, 2018.
    """
    if options is None:
        options = {'maxiter': 5000, 'maxfev': 5000, 'xatol': 0.001}

    # calculate target IV curve values
    cec_params = calcparams_cec(
        IEC61853['effective_irradiance'],
        IEC61853['temp_cell'],
        **cec_model)
    cec_ivs = singlediode(*cec_params)

    # initial guess at PVsyst parameters
    # Order in list is alpha_sc, gamma_ref, mu_gamma, I_L_ref, I_o_ref,
    # Rsh_mult = R_sh_0 / R_sh_ref, R_sh_ref, R_s
    initial = [0, 1.2, 0.001, cec_model['I_L_ref'], cec_model['I_o_ref'],
               12, 1000, cec_model['R_s']]

    # bounds for PVsyst parameters
    b_alpha = (-1, 1)
    b_gamma = (1, 2)
    b_mu = (-1, 1)
    b_IL = (1e-12, 100)
    b_Io = (1e-24, 0.1)
    b_Rmult = (1, 20)
    b_Rsh = (100, 1e6)
    b_Rs = (1e-12, 10)
    bounds = [b_alpha, b_gamma, b_mu, b_IL, b_Io, b_Rmult, b_Rsh, b_Rs]

    # optimization to find PVsyst parameters
    result = optimize.minimize(
        _pvsyst_objfun, initial,
        args=(cec_ivs, IEC61853['effective_irradiance'],
              IEC61853['temp_cell'], cells_in_series),
        method='Nelder-Mead',
        bounds=bounds,
        options=options)

    alpha_sc, gamma, mu_gamma, I_L_ref, I_o_ref, Rsh_mult, R_sh_ref, R_s = \
        result.x

    R_sh_0 = Rsh_mult * R_sh_ref
    R_sh_exp = 5.5
    EgRef = 1.121  # default for all modules in the CEC model
    return {'alpha_sc': alpha_sc,
            'I_L_ref': I_L_ref, 'I_o_ref': I_o_ref, 'EgRef': EgRef, 'R_s': R_s,
            'R_sh_ref': R_sh_ref, 'R_sh_0': R_sh_0, 'R_sh_exp': R_sh_exp,
            'gamma_ref': gamma, 'mu_gamma': mu_gamma,
            'cells_in_series': cells_in_series,
            }


def _cec_objfun(cec_mod, pvs_ivs, ee, tc, alpha_sc):
    # translate the guess into named args that are used in the functions
    # order : [I_L_ref, I_o_ref, a_ref, R_sh_ref, R_s, alpha_sc, Adjust]
    # pvs_ivs : DataFrame with columns i_sc, v_oc, i_mp, v_mp, p_mp
    # ee : effective irradiance
    # tc : cell temperature
    # alpha_sc : temperature coefficient for Isc
    I_L_ref = cec_mod[0]
    I_o_ref = cec_mod[1]
    a_ref = cec_mod[2]
    R_sh_ref = cec_mod[3]
    R_s = cec_mod[4]
    Adjust = cec_mod[5]
    alpha_sc = alpha_sc

    cec_params = calcparams_cec(
        ee, tc, alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust)
    cec_ivs = singlediode(*cec_params)

    isc_rss = np.sqrt(sum((cec_ivs['i_sc'] - pvs_ivs['i_sc'])**2))
    imp_rss = np.sqrt(sum((cec_ivs['i_mp'] - pvs_ivs['i_mp'])**2))
    voc_rss = np.sqrt(sum((cec_ivs['v_oc'] - pvs_ivs['v_oc'])**2))
    vmp_rss = np.sqrt(sum((cec_ivs['v_mp'] - pvs_ivs['v_mp'])**2))
    pmp_rss = np.sqrt(sum((cec_ivs['p_mp'] - pvs_ivs['p_mp'])**2))

    mean_diff = (isc_rss+imp_rss+voc_rss+vmp_rss+pmp_rss) / 5

    return mean_diff


def convert_pvsyst_cec(pvsyst_model, method='Nelder-Mead', options=None):
    r"""
    Convert a set of PVsyst model parameters to an equivalent set of CEC model
    parameters.

    Parameter conversion uses optimization as described in [1]_ to fit the
    CEC model to :math:`I_{sc}`, :math:`V_{oc}`, :math:`V_{mp}`,
    :math:`I_{mp}`, and :math:`P_{mp}`, calculated using the input PVsyst model
    at the IEC 61853-3 conditions [2]_.

    Parameters
    ----------
    pvsyst_model : dict or DataFrame
        Must include keys: 'alpha_sc', 'I_L_ref', 'I_o_ref', 'EgRef', 'R_s',
        'R_sh_ref', 'R_sh_0', 'R_sh_exp', 'gamma_ref', 'mu_gamma',
        'cells_in_series'
    method : str, default 'Nelder-Mead'
        Method for scipy.optimize.minimize.
    options : dict, optional
        Solver options passed to scipy.optimize.minimize.

    Returns
    -------
    dict with the following elements:
        I_L_ref : float
            The light-generated current (or photocurrent) at reference
            conditions [A].
        I_o_ref : float
            The dark or diode reverse saturation current at reference
            conditions [A].
        R_s : float
            The series resistance at reference conditions [ohm].
        R_sh_ref : float
            The shunt resistance at reference conditions [ohm].
        a_ref : float
            The product of the usual diode ideality factor ``n`` (unitless),
            number of cells in series ``Ns``, and cell thermal voltage at
            reference conditions [V].
        Adjust : float
            The adjustment to the temperature coefficient for short circuit
            current, in percent.
        EgRef : float
            The energy bandgap at reference temperature [eV].
        dEgdT : float
            The temperature dependence of the energy bandgap at reference
            conditions [1/K].

    Notes
    -----
    Reference conditions are irradiance of 1000 W/m⁻² and cell temperature of
    25 °C.

    See Also
    --------
    pvlib.ivtools.convert.convert_cec_pvsyst

    References
    ----------
    .. [1] L. Deville et al., "Parameter Translation for Photovoltaic Single
       Diode Models", Journal of Photovoltaics, vol. 15(3), pp. 451-457,
       May 2025. :doi:`10.1109/jphotov.2025.3539319`

    .. [2] "IEC 61853-3 Photovoltaic (PV) module performance testing and energy
       rating - Part 3: Energy rating of PV modules". IEC, Geneva, 2018.
    """

    if options is None:
        options = {'maxiter': 5000, 'maxfev': 5000, 'xatol': 0.001}

    # calculate target IV curve values
    pvs_params = calcparams_pvsyst(
        IEC61853['effective_irradiance'],
        IEC61853['temp_cell'],
        **pvsyst_model)
    pvsyst_ivs = singlediode(*pvs_params)

    # set EgRef and dEgdT to CEC defaults
    EgRef = 1.121
    dEgdT = -0.0002677

    # initial guess
    # order must match _pvsyst_objfun
    # order : [I_L_ref, I_o_ref, a_ref, R_sh_ref, R_s, alpha_sc, Adjust]
    nNsVth = pvsyst_model['gamma_ref'] * pvsyst_model['cells_in_series'] \
        * 0.025
    initial = [pvsyst_model['I_L_ref'], pvsyst_model['I_o_ref'],
               nNsVth, pvsyst_model['R_sh_ref'], pvsyst_model['R_s'],
               0]

    # bounds for PVsyst parameters
    b_IL = (1e-12, 100)
    b_Io = (1e-24, 0.1)
    b_aref = (1e-12, 1000)
    b_Rsh = (100, 1e6)
    b_Rs = (1e-12, 10)
    b_Adjust = (-100, 100)
    bounds = [b_IL, b_Io, b_aref, b_Rsh, b_Rs, b_Adjust]

    result = optimize.minimize(
        _cec_objfun, initial,
        args=(pvsyst_ivs, IEC61853['effective_irradiance'],
              IEC61853['temp_cell'], pvsyst_model['alpha_sc']),
        method='Nelder-Mead',
        bounds=bounds,
        options=options)

    I_L_ref, I_o_ref, a_ref, R_sh_ref, R_s, Adjust = result.x

    return {'alpha_sc': pvsyst_model['alpha_sc'],
            'a_ref': a_ref, 'I_L_ref': I_L_ref, 'I_o_ref': I_o_ref,
            'R_sh_ref': R_sh_ref, 'R_s': R_s, 'Adjust': Adjust,
            'EgRef': EgRef, 'dEgdT': dEgdT
            }
