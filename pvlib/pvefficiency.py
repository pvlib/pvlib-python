"""
This module contains implementations of PV module efficiency models.

These models have a common purpose, which is to predict the efficiency at
maximum power point as a function of the main operating conditions:
effective irradiance and module temperature.

A function to fit any of these models to measurements is also provided.

Copyright (c) 2019-2020 Anton Driesse, PV Performance Labs.
"""

import inspect

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import exp10


def fit_efficiency_model(irradiance, temperature, eta, model, p0=None,
                         **kwargs):
    """
    Determine the parameters of a module efficiency model by non-linear
    least-squares fit.

    This is a convenience function that calls the scipy curve_fit function
    with suitable parameters and defaults.

    Parameters
    ----------
    irradiance : numeric, non-negative
        Effective irradiance incident on the PV module. [W/m²]

    temperature : numeric
        PV module operating temperature. [°C]

    eta : numeric
        Efficiency of the PV module at the specified irradiance and
        temperature(s). [unitless|%]

    model : function
        A PV module efficiency function such as `adr`.  It must take
        irradiance and temperature as the first two arguments and the
        model-specific parameters as the remaining arguments. [n/a]

    p0 : array_like, optional
        Initial guess for the parameters, which may speed up the fit process.
        Units depend on the model.

    kwargs :
        Optional keyword arguments passed to `curve_fit`.

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared
        residuals ``model(irradiance, temperature, *popt) - eta`` is
        minimized.

    pcov : 2-D array
        Estimated covariance of popt. See `curve_fit` for details.

    See also
    --------
    pvlib.pvefficiency.adr
    scipy.optimize.curve_fit

    Author: Anton Driesse, PV Performance Labs
    """

    if p0 is None:
        # determine number of parameters by inspecting the function
        # and set initial parameters all to 1
        sig = inspect.signature(model)
        p0 = np.zeros(len(sig.parameters) - 2)

    if not 'method' in kwargs:
        kwargs['method'] = 'trf'

    def model_wrapper(xdata, *params):
        return model(*xdata, *params)

    popt, pcov = curve_fit(model_wrapper,
                           xdata=[irradiance, temperature],
                           ydata=eta,
                           p0=p0,
                           **kwargs
                           )
    return popt, pcov


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
            k_a=100, k_d=-6.0, tc_d=-0.02, k_rs=0.01, k_rsh=0.01)
    array([100.        ,  89.13695871])

    >>> adr([1000, 200], 25,
            k_a=1.0, k_d=-6.0, tc_d=-0.02, k_rs=0.01, k_rsh=0.01)
    array([1.        , 0.89136959])


    Author: Anton Driesse, PV Performance Labs
    '''
    g = irradiance
    t = temperature

    k_a = np.array(k_a)
    k_d = np.array(k_d)
    tc_d = np.array(tc_d)
    k_rs = np.array(k_rs)
    k_rsh = np.array(k_rsh)

    # normalize the irradiance
    G_REF = np.array(1000.)
    s = g / G_REF

    # obtain the difference from reference temperature
    T_REF = np.array(25.)
    dt   = t - T_REF

    # equation 29 in JPV
    s_o     = exp10(k_d + (dt * tc_d))
    s_o_ref = exp10(k_d)

    # equation 28 and 30 in JPV
    # the constant k_v does not appear here because it cancels out
    v  = np.log(s / s_o     + 1)
    v /= np.log(1 / s_o_ref + 1)

    # equation 25 in JPV
    eta  = k_a * ((1 + k_rs + k_rsh) * v - k_rs * s - k_rsh * v**2)

    return eta

