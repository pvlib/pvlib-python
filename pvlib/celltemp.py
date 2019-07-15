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


def sapm(poa_global, temp_air, wind_speed, model='open_rack_cell_glassback'):
    '''
    Estimate cell and module temperatures per the Sandia PV Array
    Performance Model (SAPM, SAND2004-3535), from the incident
    irradiance, wind speed, ambient temperature, and SAPM module
    parameters.

    Parameters
    ----------
    poa_global : float or Series
        Total incident irradiance in W/m^2.

    temp_air : float or Series
        Ambient dry bulb temperature in degrees C.

    wind_speed : float or Series
        Wind speed in m/s at a height of 10 meters.

    model : string, list, or dict, default 'open_rack_cell_glassback'
        Model to be used.

        If string, can be:

            * 'open_rack_cell_glassback' (default)
            * 'roof_mount_cell_glassback'
            * 'open_rack_cell_polymerback'
            * 'insulated_back_polymerback'
            * 'open_rack_polymer_thinfilm_steel'
            * '22x_concentrator_tracker'

        If dict, supply the following parameters
        (if list, in the following order):

            * a : float
                SAPM module parameter for establishing the upper
                limit for module temperature at low wind speeds and
                high solar irradiance.

            * b : float
                SAPM module parameter for establishing the rate at
                which the module temperature drops as wind speed increases
                (see SAPM eqn. 11).

            * deltaT : float
                SAPM module parameter giving the temperature difference
                between the cell and module back surface at the
                reference irradiance, E0.

    Returns
    --------
    DataFrame with columns 'temp_cell' and 'temp_module'.
    Values in degrees C.

    References
    ----------
    [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
    Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
    NM.

    See Also
    --------
    sapm
    '''

    temp_models = TEMP_MODEL_PARAMS['sapm']

    if isinstance(model, str):
        model = temp_models[model.lower()]

    elif isinstance(model, (dict, pd.Series)):
        model = [model['a'], model['b'], model['deltaT']]

    a = model[0]
    b = model[1]
    deltaT = model[2]

    E0 = 1000.  # Reference irradiance

    temp_module = pd.Series(poa_global * np.exp(a + b * wind_speed) + temp_air)

    temp_cell = temp_module + (poa_global / E0) * (deltaT)

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
