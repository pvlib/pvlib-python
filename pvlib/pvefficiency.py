"""
This module contains implementations of PV module efficiency models.

These models have a common purpose, which is to predict the efficiency at
maximum power point as a function of the main operating conditions:
effective irradiance and module temperature.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import exp10


def power_from_efficiency(efficiency, irradiance, p_mp_ref):
    '''
    Convert normalized or relative efficiency to power.

    If you can't figure out what the parameters mean, don't use this function!
    '''
    G_REF = np.array(1000.)
    g_rel = irradiance / G_REF

    p_rel = g_rel * efficiency
    power = p_rel * p_mp_ref
    return power


def efficiency_from_power(power, irradiance, p_mp_ref):
    '''
    Convert power to normalized or relative efficiency.

    If you can't figure out what the parameters mean, don't use this function!
    '''
    G_REF = np.array(1000.)
    g_rel = irradiance / G_REF

    p_rel = power / np.asanyarray(p_mp_ref, dtype=float)
    eta_rel = p_rel / g_rel
    return eta_rel


def adr(irradiance, temperature, k_a, k_d, tc_d, k_rs, k_rsh):
    '''
    Calculate PV module efficiency using the ADR model.

    The efficiency varies with irradiance and operating temperature
    and is determined by 5 model parameters as described in [1]_.

    Parameters
    ----------
    irradiance : numeric, non-negative
        The effective irradiance incident on the PV module. [W/m²]

    temperature : numeric
        The PV module operating temperature. [°C]

    k_a : numeric
        Absolute scaling factor, which is equal to the efficiency at
        reference conditions. This factor allows the model to be used
        with relative or absolute efficiencies, and to accommodate data sets
        which are not perfectly normalized but have a slight bias at
        the reference conditions. [unitless|%]

    k_d : numeric, negative
        “Dark irradiance” or diode coefficient which influences the voltage
        increase with irradiance. [unitless]

    tc_d : numeric
        Temperature coefficient of the diode coefficient, which indirectly
        influences voltage. Because it is the only temperature coefficient
        in the model, its value will also reflect secondary temperature
        dependencies that are present in the PV module. [unitless]

    k_rs and k_rsh : numeric
        Series and shunt resistance loss factors. Because of the normalization
        they can be read as power loss fractions at reference conditions.
        For example, if k_rs is 0.05, the internal loss assigned to the
        series resistance has a magnitude equal to 5% of the module output.
        [unitless]

    Returns
    -------
    eta : numeric
        The efficiency of the module at the specified irradiance and
        temperature.

    Notes
    -----
    The efficiency values may be absolute or relative, and may be expressed
    as percent or per unit.  This is determined by the efficiency data
    used to derive values for the 5 model parameters.  The first model
    parameter k_a is equal to the efficiency at STC and therefore
    indicates the efficiency scale being used.  k_a can also be changed
    freely to adjust the scale, or to change the module class to a slightly
    higher or lower efficiency.

    All arguments may be scalars or vectors. If multiple arguments
    are vectors they must be the same length.

    See also
    --------
    pvlib.pvefficiency.fit_pvefficiency_adr

    References
    ----------
    .. [1] A. Driesse and J. S. Stein, "From IEC 61853 power measurements
       to PV system simulations", Sandia Report No. SAND2020-3877, 2020.

    .. [2] A. Driesse, M. Theristis and J. S. Stein, "A New Photovoltaic Module
       Efficiency Model for Energy Prediction and Rating," in IEEE Journal
       of Photovoltaics, vol. 11, no. 2, pp. 527-534, March 2021,
       doi: 10.1109/JPHOTOV.2020.3045677.

    Examples
    --------
    >>> adr([1000, 200], 25,
            k_a=100, k_d=-6.0, tc_d=0.02, k_rs=0.05, k_rsh=0.10)
    array([100.        ,  92.79729308])

    >>> adr([1000, 200], 25,
            k_a=1.0, k_d=-6.0, tc_d=0.02, k_rs=0.05, k_rsh=0.10)
    array([1.        , 0.92797293])


    Adapted from https://github.com/adriesse/pvpltools-python
    Copyright (c) 2022, Anton Driesse, PV Performance Labs
    All rights reserved.
    '''
    k_a = np.array(k_a)
    k_d = np.array(k_d)
    tc_d = np.array(tc_d)
    k_rs = np.array(k_rs)
    k_rsh = np.array(k_rsh)

    # normalize the irradiance
    G_REF = np.array(1000.)
    s = irradiance / G_REF

    # obtain the difference from reference temperature
    T_REF = np.array(25.)
    dt = temperature - T_REF

    # equation 29 in JPV
    s_o     = exp10(k_d + (dt * tc_d))                             # noQA: E221
    s_o_ref = exp10(k_d)

    # equation 28 and 30 in JPV
    # the constant k_v does not appear here because it cancels out
    v  = np.log(s / s_o     + 1)                                   # noQA: E221
    v /= np.log(1 / s_o_ref + 1)

    # equation 25 in JPV
    eta = k_a * ((1 + k_rs + k_rsh) * v - k_rs * s - k_rsh * v**2)

    return eta


def fit_pvefficiency_adr(irradiance, temperature, eta, dict_output=True,
                         **kwargs):
    """
    Determine the parameters of the adr module efficiency model by non-linear
    least-squares fit to lab or field measurements.

    Parameters
    ----------
    irradiance : numeric, non-negative
        Effective irradiance incident on the PV module. [W/m²]

    temperature : numeric
        PV module operating temperature. [°C]

    eta : numeric
        Efficiency of the PV module at the specified irradiance and
        temperature(s). [unitless] or [%]

    dict_output : boolean, optional
        When True, return the result as a dictionary; when False, return
        the result as an numpy array.

    kwargs :
        Optional keyword arguments passed to `curve_fit`.

    Returns
    -------
    popt : array
        Optimal values for the parameters.

    pcov : 2-D array
        Estimated covariance of popt. See `curve_fit` for details.

    See also
    --------
    pvlib.pvefficiency.adr


    Adapted from https://github.com/adriesse/pvpltools-python
    Copyright (c) 2022, Anton Driesse, PV Performance Labs
    All rights reserved.
    """
    irradiance = np.asarray(irradiance, dtype=float).reshape(-1)
    temperature = np.asarray(temperature, dtype=float).reshape(-1)
    eta = np.asarray(eta, dtype=float).reshape(-1)

    eta_max = np.max(eta)

    P_NAMES = ['k_a', 'k_d', 'tc_d', 'k_rs', 'k_rsh']
    P_MAX   = [+np.inf,   0, +0.1, 1, 1]                           # noQA: E221
    P_MIN   = [0,       -12, -0.1, 0, 0]                           # noQA: E221
    P0      = [eta_max,  -6,  0.0, 0, 0]                           # noQA: E221
    P_SCALE = [eta_max,  10,  0.1, 1, 1]

    fit_options = dict(p0=P0,
                       bounds=[P_MIN, P_MAX],
                       method='trf',
                       x_scale=P_SCALE,
                       loss='soft_l1',
                       f_scale=eta_max * 0.05,
                       )

    fit_options.update(kwargs)

    def adr_wrapper(xdata, *params):
        return adr(*xdata, *params)

    result = curve_fit(adr_wrapper,
                       xdata=[irradiance, temperature],
                       ydata=eta,
                       **fit_options,
                       )
    popt = result[0]
    if dict_output:
        return dict(zip(P_NAMES, popt))
    else:
        return popt
