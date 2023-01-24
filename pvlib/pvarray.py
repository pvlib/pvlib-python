"""
This module contains implementations of PV module and array electrical models.

These models are used to predict the electrical behavior of pv modules
or collections of pv modules (arrays).  The primary inputs are
effective irradiance and operating temperature and the outputs may range from
power or efficiency at the maximum power point to complete IV curves.
Supporting functions and parameter fitting functions may also be found here.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import exp10


def pvefficiency_adr(effective_irradiance, temp_cell,
                     k_a, k_d, tc_d, k_rs, k_rsh):
    '''
    Calculate PV module efficiency using the ADR model.

    The efficiency varies with irradiance and operating temperature
    and is determined by 5 model parameters as described in [1]_.

    Parameters
    ----------
    effective_irradiance : numeric, non-negative
        The effective irradiance incident on the PV module. [W/m^2]

    temp_cell : numeric
        The PV module operating temperature. [°C]

    k_a : numeric
        Absolute scaling factor, which is equal to the efficiency at
        reference conditions. This factor allows the model to be used
        with relative or absolute efficiencies, and to accommodate data sets
        which are not perfectly normalized but have a slight bias at
        the reference conditions. [unitless]

    k_d : numeric, negative
        “Dark irradiance” or diode coefficient which influences the voltage
        increase with irradiance. [unitless]

    tc_d : numeric
        Temperature coefficient of the diode coefficient, which indirectly
        influences voltage. Because it is the only temperature coefficient
        in the model, its value will also reflect secondary temperature
        dependencies that are present in the PV module. [unitless]

    k_rs : numeric
        Series resistance loss coefficient. Because of the normalization
        it can be read as a power loss fraction at reference conditions.
        For example, if ``k_rs`` is 0.05, the internal loss assigned to the
        series resistance has a magnitude equal to 5% of the module output.
        [unitless]

    k_rsh : numeric
        Shunt resistance loss coefficient. Can be interpreted as a power
        loss fraction at reference conditions like ``k_rs``.
        [unitless]

    Returns
    -------
    eta : numeric
        The efficiency of the module at the specified irradiance and
        temperature.

    Notes
    -----
    Efficiency values ``eta`` may be absolute or relative, and may be expressed
    as percent or per unit.  This is determined by the efficiency data
    used to derive values for the 5 model parameters.  The first model
    parameter ``k_a`` is equal to the efficiency at STC and therefore
    indicates the efficiency scale being used. ``k_a`` can also be changed
    freely to adjust the scale, or to change the module to a slightly
    higher or lower efficiency class.

    All arguments may be scalars or array-like. If multiple arguments
    are array-like they must be the same shape or broadcastable to the
    same shape.

    See also
    --------
    pvlib.pvarray.fit_pvefficiency_adr

    References
    ----------
    .. [1] A. Driesse and J. S. Stein, "From IEC 61853 power measurements
       to PV system simulations", Sandia Report No. SAND2020-3877, 2020.
       :doi:`10.2172/1615179`

    .. [2] A. Driesse, M. Theristis and J. S. Stein, "A New Photovoltaic Module
       Efficiency Model for Energy Prediction and Rating," in IEEE Journal
       of Photovoltaics, vol. 11, no. 2, pp. 527-534, March 2021.
       :doi:`10.1109/JPHOTOV.2020.3045677`

    Examples
    --------
    >>> pvefficiency_adr([1000, 200], 25,
            k_a=100, k_d=-6.0, tc_d=0.02, k_rs=0.05, k_rsh=0.10)
    array([100.        ,  92.79729308])

    >>> pvefficiency_adr([1000, 200], 25,
            k_a=1.0, k_d=-6.0, tc_d=0.02, k_rs=0.05, k_rsh=0.10)
    array([1.        , 0.92797293])

    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs, Dec. 2022
    # Adapted from https://github.com/adriesse/pvpltools-python

    k_a = np.array(k_a)
    k_d = np.array(k_d)
    tc_d = np.array(tc_d)
    k_rs = np.array(k_rs)
    k_rsh = np.array(k_rsh)

    # normalize the irradiance
    G_REF = np.array(1000.)
    s = effective_irradiance / G_REF

    # obtain the difference from reference temperature
    T_REF = np.array(25.)
    dt = temp_cell - T_REF

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


def fit_pvefficiency_adr(effective_irradiance, temp_cell, eta,
                         dict_output=True, **kwargs):
    """
    Determine the parameters of the ADR module efficiency model by non-linear
    least-squares fit to lab or field measurements.

    Parameters
    ----------
    effective_irradiance : numeric, non-negative
        Effective irradiance incident on the PV module. [W/m^2]

    temp_cell : numeric
        PV module operating temperature. [°C]

    eta : numeric
        Efficiency of the PV module at the specified irradiance and
        temperature(s). [unitless] or [%]

    dict_output : boolean, optional
        When True (default), return the result as a dictionary; when False,
        return the result as a numpy array.

    kwargs :
        Optional keyword arguments passed to `scipy.optimize.curve_fit`.
        These kwargs can over-ride some options set within this function,
        which could be interesting for very advanced users.

    Returns
    -------
    popt : array or dict
        Optimal values for the parameters.

    Notes
    -----
    The best fits are obtained when the lab or field data include a wide range
    of both irradiance and temperature values.  A minimal data set
    would consist of 6 operating points covering low, medium and high
    irradiance levels at two operating temperatures.

    See also
    --------
    pvlib.pvarray.pvefficiency_adr
    scipy.optimize.curve_fit

    """
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs, Dec. 2022
    # Adapted from https://github.com/adriesse/pvpltools-python

    irradiance = np.asarray(effective_irradiance, dtype=float).reshape(-1)
    temperature = np.asarray(temp_cell, dtype=float).reshape(-1)
    eta = np.asarray(eta, dtype=float).reshape(-1)

    eta_max = np.max(eta)

    P_NAMES = ['k_a', 'k_d', 'tc_d', 'k_rs', 'k_rsh']
    P_MAX   = [+np.inf,   0, +0.1, 1, 1]                           # noQA: E221
    P_MIN   = [0,       -12, -0.1, 0, 0]                           # noQA: E221
    P0      = [eta_max,  -6,  0.0, 0, 0]                           # noQA: E221
    P_SCALE = [eta_max,  10,  0.1, 1, 1]

    SIGMA = 1 / np.sqrt(irradiance / 1000)

    fit_options = dict(p0=P0,
                       bounds=[P_MIN, P_MAX],
                       method='trf',
                       x_scale=P_SCALE,
                       loss='soft_l1',
                       f_scale=eta_max * 0.05,
                       sigma=SIGMA,
                       )

    fit_options.update(kwargs)

    def adr_wrapper(xdata, *params):
        return pvefficiency_adr(*xdata, *params)

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
