# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:40:42 2019

@author: cwhanse
"""

import numpy as np
import pandas as pd


TEMP_MODEL_PARAMS = {
    'sapm': {'open_rack_cell_glassback': (-3.47, -.0594, 3),
             'roof_mount_cell_glassback': (-2.98, -.0471, 1),
             'open_rack_cell_polymerback': (-3.56, -.0750, 3),
             'insulated_back_polymerback': (-2.81, -.0455, 0),
             'open_rack_polymer_thinfilm_steel': (-3.58, -.113, 3),
             '22x_concentrator_tracker': (-3.23, -.130, 13)},
    'pvsyst': {'freestanding': (29.0, 0), 'insulated': (15.0, 0)}
}


def sapm(poa_global, temp_air, wind_speed, a, b, deltaT, irrad_ref=1000):
    r'''
    Estimate cell and module temperatures per the Sandia PV Array
    Performance Model [1].

    Parameters
    ----------
    poa_global : float or Series
        Total incident irradiance in W/m^2.

    temp_air : float or Series
        Ambient dry bulb temperature in degrees C.

    wind_speed : float or Series
        Wind speed in m/s at a height of 10 meters.

    a : float
        Parameter `a` in :eq:`eq1`.

    b : float
        Parameter `b` in :eq:`eq1`.

    deltaT : float, [C]
        Parameter :math:`\Delta T` in :eq:`eq2`.

    irrad_ref : float, default 1000 [W/m2]
        Reference irradiance, parameter :math:`E0` in :eq:`eq2`.

    Returns
    --------
    DataFrame with columns 'temp_cell' and 'temp_module'.
    Values in degrees C.

    Notes
    -----
    The model for cell temperature :math:`T_{C}` and module temperature
    :math:`T_{m}` is given by a pair of equations (Eq. 11 and 12 in [1]).

    .. :math::
        :label: eq1

        T_{m} = E \times \exp (a + b \times WS) + T_{a}

        :label: eq2

        T_{C} = T_{m} + \frac{E}{E0} \Delta T

    Inputs to the model are plane-of-array irradiance :math:`E` (W/m2) and
    ambient air temperature :math:`T_{a}` (C). Model outputs are surface
    temperature at the back of the module :math:`T_{m}` and cell temperature
    :math:`T_{C}`. Model parameters depend both on the module construction and
    its mounting. Parameter sets are provided in [1] for representative modules
    and mounting.

    | Module | Mounting | a | b | :math:`\Delta T [\degree C]` |
    |:-------|:---------|---:|---:|-----------------------------:|
    | glass/cell/glass | open rack | -3.47 | -0.0594 | 3 |
    | glass/cell/glass | close roof mount | -2.98 | -0.0471 | 1 |
    | glass/cell/polymer | open rack | -3.56 | -0.075 | 3 |
    | glass/cell/polymer | insulated back | -2.81 | -0.0455 | 0 |

    References
    ----------
    [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
    Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
    NM.

    '''
    temp_module = pd.Series(poa_global * np.exp(a + b * wind_speed) + temp_air)
    temp_cell = temp_module + (poa_global / irrad_ref) * (deltaT)
    return pd.DataFrame({'temp_cell': temp_cell, 'temp_module': temp_module})


def pvsyst(poa_global, temp_air, wind_speed=1.0, eta_m=0.1,
           alpha_absorption=0.9, model='freestanding'):
    """
    Calculate cell temperature using an emperical heat loss factor model
    as implemented in PVsyst.

    The heat loss factors provided through the 'model' argument
    represent the combined effect of convection, radiation and conduction,
    and their values are experimentally determined.

    Parameters
    ----------
    poa_global : float or Series
        Total incident irradiance in W/m^2.

    temp_air : float or Series
        Ambient dry bulb temperature in degrees C.

    wind_speed : float or Series, default 1.0
        Wind speed in m/s measured at the same height for which the wind loss
        factor was determined.  The default value is 1.0, which is the wind
        speed at module height used to determine NOCT.

    eta_m : numeric, default 0.1
        Module external efficiency as a fraction, i.e., DC power / poa_global.

    alpha_absorption : numeric, default 0.9
        Absorption coefficient

    model : string, tuple, or list (no dict), default 'freestanding'
        Heat loss factors to be used.

        If string, can be:

            * 'freestanding' (default)
                Modules with rear surfaces exposed to open air (e.g. rack
                mounted).
            * 'insulated'
                Modules with rear surfaces in close proximity to another
                surface (e.g. roof mounted).

        If tuple/list, supply parameters in the following order:

            * constant_loss_factor : float
                Combined heat loss factor coefficient. Freestanding
                default is 29, fully insulated arrays is 15.

            * wind_loss_factor : float
                Combined heat loss factor influenced by wind. Default is 0.

    Returns
    -------
    temp_cell : numeric or Series
        Cell temperature in degrees Celsius

    References
    ----------
    [1]"PVsyst 6 Help", Files.pvsyst.com, 2018. [Online]. Available:
    http://files.pvsyst.com/help/index.html. [Accessed: 10- Dec- 2018].

    [2] Faiman, D. (2008). "Assessing the outdoor operating temperature of
    photovoltaic modules." Progress in Photovoltaics 16(4): 307-315.
    """

    pvsyst_presets = TEMP_MODEL_PARAMS['pvsyst']

    if isinstance(model, str):
        model = model.lower()
        constant_loss_factor, wind_loss_factor = pvsyst_presets[model]
    elif isinstance(model, (tuple, list)):
        constant_loss_factor, wind_loss_factor = model
    else:
        raise TypeError(
            "Please provide model as a str, or tuple/list."
        )

    total_loss_factor = wind_loss_factor * wind_speed + constant_loss_factor
    heat_input = poa_global * alpha_absorption * (1 - eta_m)
    temp_difference = heat_input / total_loss_factor
    temp_cell = pd.Series(temp_air + temp_difference)

    return pd.DataFrame({'temp_cell': temp_cell})
