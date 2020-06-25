import numpy as np
import pandas as pd

from numpy.polynomial.polynomial import polyfit
from numpy.testing import assert_allclose

import os

from pvlib.pvsystem import snlinverter


def _fit_ps0(p_ac, p_dc, p_ac0, p_dc0):
    ''' Determine the ps0 parameter as the intercept at p_ac=0 of a line fit
    to p_ac vs. (p_ac0 * p_dc - p_dc0 * p_ac) / (p_ac0 - p-ac)
    This function assumes that p_dc is paired with p_ac, and that
    p_ac < p_ac0.
    '''
    y = np.array((p_ac0 * p_dc - p_dc0 * p_ac) / (p_ac0 - p_ac), dtype=float)
    x = np.array([np.ones_like(y), p_ac], dtype=float).T
    beta, _, _, _ = np.linalg.lstsq(x, y)
    return beta[0]


def _calc_c0(p_ac, p_dc, p_ac0, p_dc0, p_s0):
    x = p_dc - p_s0
    c0 = (p_ac - x / (p_dc0 - p_s0) * p_ac0) / (x**2. - x * (p_dc0 - p_s0))
    return np.nanmean(c0)


def fit_sandia_datasheet(curves, p_ac_0, dc_voltage, p_nt):
    r'''
    Determine parameters for the Sandia inverter model from a datasheet's
    efficiency curves.

    Parameters
    ----------
    curves : DataFrame
        Columns must be ``'fraction_of_rated_power'``, ``'efficiency'``,
        ``'dc_voltage_level'``. See notes for definition and units for each
        column.
    p_ac_0 : numeric
        Rated AC power of the inverter [W].
    dc_voltage : Dict
        Input DC voltage levels. Keys must be 'Vmin', 'Vnom', and 'Vmax'. [V]
    p_nt : numeric
        Night tare, i.e., power consumed while inverter is not delivering
        AC power. [W]

    Returns
    -------
    Dict with parameters for the Sandia inverter model.

    See :py:func:`snl_inverter` for a description of entries in the returned
    Dict.

    See Also
    --------
    snlinverter

    Notes
    -----
    An inverter efficiency curve comprises a series of pairs
    ('fraction_of_rated_power', 'efficiency'), e.g. (0.1, 0.5), (0.2, 0.7),
    etc. at a specified DC voltage level.The DataFrame `curves` should contain
    multiple efficiency curves for each DC voltage level; at least five curves
    at each level is recommended. Columns in `curves` must be the following:

    ================           ========================================
    Column name                Description
    ================           ========================================
    'fraction_of_rated_power'  Fraction of rated AC power `p_ac_0`. Values must
                               include 1.0. The CEC inverter test protocol
                               specifies values of 0.1, 0.2, 0.3, 0.5, 0.75
                               and 1.0. [unitless]
    'efficiency'               Ratio of measured AC output power to measured
                               DC input power. [unitless]
    'dc_voltage_level'         Values of 'Vmin', 'Vnom', and 'Vmax'.

    The output AC power is calculated as 'fraction_of_rated_power' * 'p_ac_0'.
    The input DC power is calculated as (output AC power) / 'efficiency'.

    References
    ----------
    .. [1] SAND2007-5036, "Performance Model for Grid-Connected
       Photovoltaic Inverters by D. King, S. Gonzalez, G. Galbraith, W.
       Boyson
    .. [2] Sandia Inverter Model page, PV Performance Modeling Collaborative
       https://pvpmc.sandia.gov/modeling-steps/dc-to-ac-conversion/sandia-inverter-model/  # noqa: E501
    '''

    voltage_levels = ['Vmin', 'Vnom', 'Vmax']

    curves['ac_power'] = curves['fraction_of_rated_power'] * p_ac_0
    curves['dc_power'] = curves['ac_power'] / curves['efficiency']

    #Empty dataframe to contain intermediate variables
    coeffs = pd.DataFrame(index=voltage_levels,
                          columns=['p_dc0', 'p_s0', 'c0', 'a', 'b', 'c'],
                          data=np.nan)

    for vl in voltage_levels:
        temp = curves[curves['dc_voltage_level'] == vl]
        # determine p_dc0
        base_eff = temp['efficiency'][temp['fraction_of_rated_power']==1.0]
        p_dc_0 = p_ac_0 / base_eff
        p_dc_0 = float(p_dc_0)

        # determine p_s0
        p_s0 = _fit_ps0(temp['ac_power'][temp['ac_power'] < p_ac_0],
                        temp['dc_power'][temp['ac_power'] < p_ac_0],
                        p_ac_0, p_dc_0)

        # calculate c0
        c0 = _calc_c0(temp['ac_power'], temp['dc_power'], p_ac_0, p_dc_0, p_s0)

        # fit a quadratic to (pac, pdc) at each voltage level, to get c3
        #Get a,b,c values from polyfit
        x = curves['dc_power'][curves['dc_voltage_level']==vl]
        y = curves['ac_power'][curves['dc_voltage_level']==vl]
        c, b, a = polyfit(x, y, 2)
        coeffs['p_dc0'][vl] = p_dc_0
        coeffs['p_s0'][vl] = p_s0
        coeffs['c0'][vl] = c0
        coeffs['a'][vl] = a
        coeffs['b'][vl] = b
        coeffs['c'][vl] = c

    p_dc0 = coeffs['p_dc0']['Vnom']
    p_s0 = coeffs['p_s0']['Vnom']
    c0 = coeffs['c0']['Vnom']
    c1 = (coeffs['p_dc0']['Vmax'] - coeffs['p_dc0']['Vmin']) \
        / (dc_voltage['Vmax'] - dc_voltage['Vmin']) / p_dc0
    c2 = (coeffs['p_s0']['Vmax'] - coeffs['p_s0']['Vmin']) \
        / (dc_voltage['Vmax'] - dc_voltage['Vmin']) / p_s0
    c3 = (coeffs['a']['Vmax'] - coeffs['a']['Vmin']) \
        / ((dc_voltage['Vmax'] - dc_voltage['Vmin']) * c0)

    # prepare dict and return
    return {'Paco': p_ac_0, 'Pdco': p_dc0, 'Vdco': dc_voltage['Vnom'],
            'Pso': p_s0, 'C0': c0, 'C1': c1, 'C2': c2, 'C3': c3, 'Pnt': p_nt}


def fit_sandia_meas(curves, p_ac_0, p_nt):
    r'''
    Determine parameters for the Sandia inverter model from measured efficiency
    curves.

    Parameters
    ----------
    curves : DataFrame
        Columns must be ``'fraction_of_rated_power'``, ``'efficiency'``,
        ``'dc_voltage_level'``, ``'ac_power'``, ``'dc_voltage'``. See notes
        for definition and units for each column.
    p_ac_0 : numeric
        Rated AC power of the inverter [W].
    p_nt : numeric
        Night tare, i.e., power consumed while inverter is not delivering
        AC power. [W]

    Returns
    -------
    Dict with parameters for the Sandia inverter model.

    See :py:func:`snl_inverter` for a description of entries in the returned
    Dict.

    See Also
    --------
    snlinverter

    Notes
    -----
    An inverter efficiency curve comprises a series of pairs
    ('fraction_of_rated_power', 'efficiency'), e.g. (0.1, 0.5), (0.2, 0.7),
    etc. at a specified DC voltage level.The DataFrame `curves` should contain
    multiple efficiency curves for each DC voltage level; at least five curves
    at each level is recommended. Columns in `curves` must be the following:

    ================           ========================================
    Column name                Description
    ================           ========================================
    'fraction_of_rated_power'  Fraction of rated AC power `p_ac_0`. The
                               CEC inverter test protocol specifies values
                               of 0.1, 0.2, 0.3, 0.5, 0.75 and 1.0. [unitless]
    'efficiency'               Ratio of measured AC output power to measured
                               DC input power. [unitless]
    'dc_voltage_level'         Must be one of 'Vmin', 'Vnom', or 'Vmax'.
    'ac_power'                 Measurd output AC power. [W]
    'dc_voltage'               Measured DC input voltage. [V]

    References
    ----------
    .. [1] SAND2007-5036, "Performance Model for Grid-Connected
       Photovoltaic Inverters by D. King, S. Gonzalez, G. Galbraith, W.
       Boyson
    .. [2] Sandia Inverter Model page, PV Performance Modeling Collaborative
       https://pvpmc.sandia.gov/modeling-steps/dc-to-ac-conversion/sandia-inverter-model/  # noqa: E501
    '''

    voltage_levels = ['Vmin', 'Vnom', 'Vmax']

    #Declaring x_d
    v_nom = curves['dc_voltage'][curves['dc_voltage_level']=='Vnom']
    v_nom = v_nom.mean()
    v_d = np.array(
        [curves['dc_voltage'][curves['dc_voltage_level']=='Vmin'].mean(),
         v_nom,
         curves['dc_voltage'][curves['dc_voltage_level']=='Vmax'].mean()])
    x_d = v_d - v_nom

    curves['dc_power'] = curves['ac_power'] / curves['efficiency']

    #Empty dataframe to contain intermediate variables
    coeffs = pd.DataFrame(index=voltage_levels,
                          columns=['a', 'b', 'c', 'p_dc', 'p_s0'], data=np.nan)

    def solve_quad(a, b, c):
        return (-b + (b**2 - 4 * a * c)**.5) / (2 * a)

    #STEP 3E, use np.polyfit to get betas
    def extract_c(x_d, add):
        test = np.polyfit(x_d, add, 1)
        beta1, beta0 = test
        c = beta1 / beta0
        return beta0, beta1, c

    for d in voltage_levels:
        x = curves['dc_power'][curves['dc_voltage_level']==d]
        y = curves['ac_power'][curves['dc_voltage_level']==d]
        #STEP 3B
        #Get a,b,c values from polyfit
        c, b, a = polyfit(x, y, 2)

        #STEP 3D, solve for p_dc and p_s0

        p_dc = solve_quad(a, b, (c - p_ac_0))

        p_s0 = solve_quad(a, b, c)


        #Add values to dataframe at index d
        coeffs['a'][d] = a
        coeffs['b'][d] = b
        coeffs['c'][d] = c
        coeffs['p_dc'][d] = p_dc
        coeffs['p_s0'][d] = p_s0

    b_dc0, b_dc1, c1 = extract_c(x_d, coeffs['p_dc'])
    b_s0, b_s1, c2 = extract_c(x_d, coeffs['p_s0'])
    b_c0, b_c1, c3 = extract_c(x_d, coeffs['a'])


    p_dc0 = b_dc0
    p_s0 = b_s0
    c0 = b_c0

    # prepare dict and return
    return {'Paco': p_ac_0, 'Pdco': p_dc0, 'Vdco': v_nom, 'Pso': p_s0,
            'C0': c0, 'C1': c1, 'C2': c2, 'C3': c3, 'Pnt': p_nt}


DATA_DIR = "C:\\python\\pvlib-remote\\pvlib-python\\pvlib\\data"
def test_fit_sandia_meas():
    inverter_curves = os.path.join(DATA_DIR, 'inverter_fit_snl_meas.csv')
    curves = pd.read_csv(inverter_curves)
    expected = np.array([333000, 343251, 740, 1427.746, -5.768e-08, 3.596e-05,
                         1.038e-03, 2.978e-05, 1.])
    keys = ['Paco', 'Pdco', 'Vdco', 'Pso', 'C0', 'C1', 'C2', 'C3', 'Pnt']
    result_dict = fit_sandia_meas(curves, 333000, 1.)
    result = np.array([result_dict[k] for k in keys])
    assert_allclose(expected, result, rtol=1e-3)


def test_fit_sandia_datasheet():
    inverter_curves = os.path.join(DATA_DIR, 'inverter_fit_snl_datasheet.csv')
    curves = pd.read_csv(inverter_curves)
    dc_voltage_levels = {'Vmin': 220., 'Vnom': 240., 'Vmax': 260.}
    expected_dict = {'Paco': 1000., 'Pdco': 1050., 'Vdco': 240., 'Pso': 10.,
                     'C0': 1e-6, 'C1': 1e-4, 'C2': 1e-2, 'C3': 1e-3, 'Pnt': 1}
    keys = ['Paco', 'Pdco', 'Vdco', 'Pso', 'C0', 'C1', 'C2', 'C3', 'Pnt']
    expected = np.array([expected_dict[k] for k in keys])
    # recover known values within 0.5%
    result_dict = fit_sandia_datasheet(curves, 1000., dc_voltage_levels, 1.)
    result = np.array([result_dict[k] for k in keys])
    assert_allclose(expected, result, rtol=5e-3)
    # calculate efficiency from recovered parameters
    calc_effic = {k: np.nan for k in dc_voltage_levels.keys()}
    for vlev in dc_voltage_levels.keys():
        pdc = curves[curves['dc_voltage_level']==vlev]['pdc']
        calc_effic[vlev] = snlinverter(dc_voltage_levels[vlev], pdc,
                                       result_dict) / pdc
        assert_allclose(calc_effic[vlev],
                        curves[curves['dc_voltage_level']==vlev]['efficiency'],
                        rtol=1e-5)
    return expected, result

exp, res = test_fit_sandia_datasheet()


# calculate efficiency from recovered parameters


