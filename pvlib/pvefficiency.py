"""
This module contains implementations of several PV module efficiency models.

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

from pvpltools.iec61853 import BilinearInterpolator


def fit_efficiency_model(irradiance, temperature, eta, model, p0=None,
                         **kwargs):
    """
    Determine the parameters of a module efficiency model by non-linear
    least-squares fit.

    This is a convenience function that calls the scipy curve_fit function
    with suitable parameters and defaults.

    Parameters
    ----------
    irradiance : non-negative numeric, W/m²
        The effective irradiance incident on the PV module.

    temperature : numeric, °C
        The module operating temperature.

    eta : numeric
        The efficiency of the module at the specified irradiance and
        temperature.

    model : function
        A PV module efficiency function such as `adr`.  It must take
        irradiance and temperature as the first two arguments and the
        model-specific parameters as the remaining arguments.

    p0 : array_like, optional
        Initial guess for the parameters, which may speed up the fit process.

    kwargs :
        Optional keyword arguments passed to `curve_fit`.

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared
        residuals of ``model(irradiance, temperature, *popt) - eta`` is minimized.

    pcov : 2-D array
        The estimated covariance of popt. See `curve_fit` for details.

    Raises
    ------
    (These errors and warnings are from `curve_fit`.)

    ValueError
        if either `ydata` or `xdata` contain NaNs, or if incompatible options
        are used.

    RuntimeError
        if the least-squares minimization fails.

    OptimizeWarning
        if covariance of the parameters can not be estimated.

    See also
    --------
    pvpltools.module_efficiency.adr
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
    irradiance : non-negative numeric, W/m²
        The effective irradiance incident on the PV module.

    temperature : numeric, °C
        The module operating temperature.

    k_a : float
        Absolute scaling factor, which is equal to the efficiency at
        reference conditions. This factor allows the model to be used
        with relative or absolute efficiencies, and to accommodate data sets
        which are not perfectly normalized but have a slight bias at
        the reference conditions.

    k_d : negative float
        “Dark irradiance” or diode coefficient which influences the voltage
        increase with irradiance.

    tc_d : float
        Temperature coefficient of the diode coefficient, which indirectly
        influences voltage. Because it is the only temperature coefficient
        in the model, its value will also reflect secondary temperature
        dependencies that are present in the PV module.

    k_rs and k_rsh : float
        Series and shunt resistance loss factors. Because of the normalization
        they can be read as power loss fractions at reference conditions.
        For example, if k_rs is 0.05, the internal loss assigned to the
        series resistance has a magnitude equal to 5% of the module output.

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

    References
    ----------
    .. [1] A. Driesse and J. S. Stein, "From IEC 61853 power measurements
       to PV system simulations", Sandia Report No. SAND2020-3877, 2020.

    .. [2] A. Driesse, M. Theristis and J. S. Stein, "A New Photovoltaic
       Module Efficiency Model for Energy Prediction and Rating",
       forthcoming.

    Author: Anton Driesse, PV Performance Labs
    '''
    g = np.asanyarray(irradiance)
    t = np.asanyarray(temperature)

    # normalize the irradiance
    G_REF = 1000
    s = g / G_REF

    # obtain the difference from reference temperature
    T_REF = 25
    dt   = t - T_REF
    t_abs = t + 273.15

    # equation 29 in JPV
    s_o     = 10**(k_d + (tc_d * dt))
    s_o_ref = 10**(k_d)

    # equation 28 and 30 in JPV
    # the constant k_v does not appear here because it cancels out
    v  = np.log(s / s_o     + 1)
    v /= np.log(1 / s_o_ref + 1)

    # equation 25 in JPV
    eta  = k_a * ((1 + k_rs + k_rsh) * v - k_rs * s - k_rsh * v**2)

    return eta


def heydenreich(irradiance, temperature, a, b, c, gamma_pmp):
    """
    Calculate PV module efficiency using the Heydenreich model.

    The efficiency varies with irradiance and operating temperature
    and is determined by three parameters for irradiance dependency and
    one for temperature dependency as described in  [1]_.

    Parameters
    ----------
    irradiance : non-negative numeric, W/m²
        The effective irradiance incident on the PV module.

    temperature : numeric, °C
        The module operating temperature.

    a, b, c : float
        Three model parameters usually determined by regression.

    gamma_pmp : float
        The temperature coefficient of power, which may be taken
        from the module datasheet or also determined by regression.

    Returns
    -------
    eta : numeric
        The efficiency of the module at the specified irradiance and
        temperature.

    See also
    --------
    fit_efficiency_model
    adr

    Notes
    -----
    A comprehensive comparison of efficiency models is found in [2]_ and [3]_.

    References
    ----------
    .. [1] W. Heydenreich, et al., "Describing the world with three parameters:
       a new approach to PV module power modelling," in 23rd European PV
       Solar Energy Conference and Exhibition (EU PVSEC), 2008, pp. 2786-2789.

    .. [2] A. Driesse and J. S. Stein, "From IEC 61853 power measurements
       to PV system simulations", Sandia Report No. SAND2020-3877, 2020.

    .. [3] A. Driesse, M. Theristis and J. S. Stein, "A New Photovoltaic
       Module Efficiency Model for Energy Prediction and Rating",
       forthcoming.

    Author: Anton Driesse, PV Performance Labs
    """
    from numpy import log, exp, square

    g = np.asanyarray(irradiance)
    t = np.asanyarray(temperature)

    dt = t - 25

    eta = (
           # power loss in R series
           a * g +
           # power gain from voltage * current
           b * log(g + 1) +
           # power loss in R shunt (constant Rsh)
           c * (square(log(g + exp(1))) / (g + 1) - 1)
           )

    eta *= 1 + gamma_pmp * dt

    return eta


def motherpv(irradiance, temperature, a, b, c, d, gamma_ref, aa, bb):
    """
    Calculate PV module efficiency using the MotherPV model.

    The efficiency varies with irradiance and operating temperature
    and is determined by 7 parameters as described in  [1]_.

    Parameters
    ----------
    irradiance : non-negative numeric, W/m²
        The effective irradiance incident on the PV module.

    temperature : numeric, °C
        The module operating temperature.

    a, b, c, d, aa, bb : float
        Six model parameters usually determined by regression.

    gamma_pmp : float
        The temperature coefficient of power, which may be taken
        from the module datasheet or also determined by regression.

    Returns
    -------
    eta : numeric
        The efficiency of the module at the specified irradiance and
        temperature.

    See also
    --------
    fit_efficiency_model
    adr

    Notes
    -----
    A comprehensive comparison of efficiency models is found in [2]_ and [3]_.

    References
    ----------
    .. [1] A. G. de Montgareuil, et al., "A new tool for the MotherPV method:
       modelling of the irradiance coefficient of photovoltaic modules,"
       in 24th European Photovoltaic Solar Energy Conference (EU PVSEC),
       2009, pp. 21-25.

    .. [2] A. Driesse and J. S. Stein, "From IEC 61853 power measurements
       to PV system simulations", Sandia Report No. SAND2020-3877, 2020.

    .. [3] A. Driesse, M. Theristis and J. S. Stein, "A New Photovoltaic
       Module Efficiency Model for Energy Prediction and Rating",
       forthcoming.

    Author: Anton Driesse, PV Performance Labs
    """
    from numpy import log

    g = np.asanyarray(irradiance)
    t = np.asanyarray(temperature)

    s = g / 1000
    dt = t - 25

    eta = ( 1 + a * (s - 1)    + b * log(s)
              + c * (s - 1)**2 + d * log(s)**2
          )
    gamma = gamma_ref * ( 1 + aa * (s - 1) + bb * log(s))

    eta *= 1 + gamma * dt

    return eta


def pvgis(irradiance, temperature, k1, k2, k3, k4, k5, k6):
    """
    Calculate PV module efficiency using the PVGIS model.

    The efficiency varies with irradiance and operating temperature
    and is determined by 6 parameters as described in [1]_.

    Parameters
    ----------
    irradiance : non-negative numeric, W/m²
        The effective irradiance incident on the PV module.

    temperature : numeric, °C
        The module operating temperature.

    k1, k2, k3, k4, k5, k6 : float
        Six model parameters usually determined by regression.

    Returns
    -------
    eta : numeric
        The efficiency of the module at the specified irradiance and
        temperature.

    See also
    --------
    fit_efficiency_model
    adr

    Notes
    -----
    A comprehensive comparison of efficiency models is found in [2]_ and [3]_.

    References
    ----------
    .. [1] T. Huld, et al., "A power-rating model for crystalline silicon
       PV modules," Solar Energy Materials and Solar Cells, vol. 95,
       pp. 3359-3369, 2011.

    .. [2] A. Driesse and J. S. Stein, "From IEC 61853 power measurements
       to PV system simulations", Sandia Report No. SAND2020-3877, 2020.

    .. [3] A. Driesse, M. Theristis and J. S. Stein, "A New Photovoltaic
       Module Efficiency Model for Energy Prediction and Rating",
       forthcoming.

    Author: Anton Driesse, PV Performance Labs
    """
    from numpy import log

    g = np.asanyarray(irradiance)
    t = np.asanyarray(temperature)

    g = g / 1000
    dt = t - 25

    eta = ( 1
            + k1 * log(g)
            + k2 * log(g)**2
            + dt * (k3
                    + k4 * log(g)
                    + k5 * log(g)**2
                    )
            + k6 * dt**2
          )

    return eta


def mpm6(irradiance, temperature, c1, c2, c3, c4, c6=0.0):
    """
    Calculate PV module efficiency using the MPM6 model (without windspeed).

    The efficiency varies with irradiance and operating temperature
    and is determined by 5 parameters as described in [1]_.  A sixth
    parameter captures the effect of windspeed but is not used in this
    implementation.

    Parameters
    ----------
    irradiance : non-negative numeric, W/m²
        The effective irradiance incident on the PV module.

    temperature : numeric, °C
        The module operating temperature.

    c1, c2, c3, c4, c6 : float
        Five model parameters usually determined by regression.

    Returns
    -------
    eta : numeric
        The efficiency of the module at the specified irradiance and
        temperature.

    See also
    --------
    mpm5
    fit_efficiency_model
    adr

    Notes
    -----
    The author of MPM6 recommends the fitting constraint c6 <= 0.

    A comprehensive comparison of efficiency models is found in [2]_ and [3]_.

    References
    ----------
    .. [1] S. Ransome and J. Sutterlueti, "How to Choose the Best Empirical
       Model for Optimum Energy Yield Predictions," in 44th IEEE Photovoltaic
       Specialist Conference (PVSC), 2017, pp. 652-657.

    .. [2] A. Driesse and J. S. Stein, "From IEC 61853 power measurements
       to PV system simulations", Sandia Report No. SAND2020-3877, 2020.

    .. [3] A. Driesse, M. Theristis and J. S. Stein, "A New Photovoltaic
       Module Efficiency Model for Energy Prediction and Rating",
       forthcoming.

    Author: Anton Driesse, PV Performance Labs
    """

    g = np.asanyarray(irradiance)
    t = np.asanyarray(temperature)

    g = g / 1000
    dt = t - 25

    eta = (
           # "actual/nominal"
           c1
           # loss due to temperature
           + c2 * dt
           # loss at low light / due to Voc
           + c3 * np.log10(g)
           # loss at high light / due to Rs
           + c4 * g
           # loss due to Rsh
           + c6 / g
           )
    return eta


def mpm5(irradiance, temperature, c1, c2, c3, c4):
    """
    Call `mpm6` with one less parameter.  See `mpm6` for more information.
    """
    return mpm6(irradiance, temperature, c1, c2, c3, c4, c6=0.0)


def fit_bilinear(irradiance, temperature, eta):
    """
    Prepare a bilinear interpolant for module efficiency.

    This function allows the class `pvpltools.iec61853.BilinearInterpolator`
    to be used in a way that is compatible with other efficiency models
    in this module.

    Parameters
    ----------
    irradiance : non-negative numeric, W/m²
        The effective irradiance incident on the PV module.

    temperature : numeric, °C
        The module operating temperature.

    eta : numeric
        The efficiency of the module at the specified irradiance and
        temperature.

    Returns
    -------
    interpolator : object
        A callable `BilinearInterpolator` object

    See also
    --------
    pvpltools.module_efficiency.bilinear
    pvpltools.iec61853.BilinearInterpolator

    Notes
    -----
    Unlike the other efficiency models, bilinear interpolation only works
    with a regular grid of measurements.  Missing values at low irradiance
    high temperature and vice versa are filled using the method described
    in [1]_ and [2]_.

    References
    ----------
    .. [1] A. Driesse and J. S. Stein, "From IEC 61853 power measurements
       to PV system simulations", Sandia Report No. SAND2020-3877, 2020.

    .. [2] A. Driesse, M. Theristis and J. S. Stein, "A New Photovoltaic
       Module Efficiency Model for Energy Prediction and Rating",
       forthcoming.

    Author: Anton Driesse, PV Performance Labs
    """
    # (re)construct the matrix as a grid for the BilinearInterpolator
    data = pd.DataFrame([irradiance, temperature, eta]).T
    grid = data.pivot(*data.columns)

    # now create the interpolator object
    interpolator = BilinearInterpolator(grid)
    return interpolator


def bilinear(irradiance, temperature, interpolator):
    """
    Calculate PV module efficiency using bilinear interpolation/extrapolation.

    This function allows the class `pvpltools.iec61853.BilinearInterpolator`
    to be used in a way that is compatible with other efficiency models
    in this module.

    Parameters
    ----------
    irradiance : non-negative numeric, W/m²
        The effective irradiance incident on the PV module.

    temperature : numeric, °C
        The module operating temperature.

    interpolator : object
        A callable `BilinearInterpolator` object

    Returns
    -------
    eta : numeric
        The efficiency of the module at the specified irradiance and
        temperature.

    See also
    --------
    module_efficiency.fit_bilinear
    pvpltools.iec61853.BilinearInterpolator

    References
    ----------
    .. [1] A. Driesse and J. S. Stein, "From IEC 61853 power measurements
       to PV system simulations", Sandia Report No. SAND2020-3877, 2020.

    .. [2] A. Driesse, M. Theristis and J. S. Stein, "A New Photovoltaic
       Module Efficiency Model for Energy Prediction and Rating",
       forthcoming.

    Author: Anton Driesse, PV Performance Labs
    """
    return interpolator(irradiance, temperature)

