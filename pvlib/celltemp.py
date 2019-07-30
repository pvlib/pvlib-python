# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:40:42 2019

@author: cwhanse
"""

import numpy as np


TEMPERATURE_MODEL_PARAMETERS = {
    'sapm': {
        'open_rack_glass_glass': {'a': -3.47, 'b': -.0594, 'deltaT': 3},
        'close_mount_glass_glass': {'a': -2.98, 'b': -.0471, 'deltaT': 1},
        'open_rack_glass_polymer': {'a': -3.56, 'b': -.0750, 'deltaT': 3},
        'insulated_back_glass_polymer': {'a': -2.81, 'b': -.0455, 'deltaT': 0},
            },
    'pvsyst': {'freestanding': {'u_c': 29.0, 'u_v': 0},
               'insulated': {'u_c': 15.0, 'u_v': 0}}
}


def sapm_cell(poa_global, air_temperature, wind_speed, a, b, deltaT,
              irrad_ref=1000):
    r'''
    Calculate cell temperature per the Sandia PV Array Performance Model [1].

    Parameters
    ----------
    poa_global : float or Series
        Total incident irradiance [W/m^2].

    air_temperature : float or Series
        Ambient dry bulb temperature [C].

    wind_speed : float or Series
        Wind speed at a height of 10 meters [m/s].

    a : float
        Parameter :math:`a` in :eq:`sapm1`.

    b : float
        Parameter :math:`b` in :eq:`sapm1`.

    deltaT : float
        Parameter :math:`\Delta T` in :eq:`sapm2` [C].

    irrad_ref : float, default 1000
        Reference irradiance, parameter :math:`E_{0}` in :eq:`sapm2` [W/m^2].

    Returns
    -------
    float or Series, values in degrees C.

    Notes
    -----
    The model for cell temperature :math:`T_{C}` is given by a pair of
    equations (Eq. 11 and 12 in [1]).

    .. math::
       :label: sapm1

        T_{m} = E \times \exp (a + b \times WS) + T_{a}

    .. math::
       :label: sapm2

        T_{C} = T_{m} + \frac{E}{E_{0}} \Delta T

    The module back surface temperature :math:`T_{m}` is implemented in
    ``cell_temperature.sapm_module``.

    Inputs to the model are plane-of-array irradiance :math:`E` (W/m2) and
    ambient air temperature :math:`T_{a}` (C). Model parameters depend both on
    the module construction and its mounting. Parameter sets are provided in
    [1] for representative modules and mounting, and are coded for convenience
    in ``cell_temperature.TEMPERATURE_MODEL_PARAMETERS``.

    +---------------+----------------+-------+---------+---------------------+
    | Module        | Mounting       | a     | b       | :math:`\Delta T [C]`|
    +===============+================+=======+=========+=====================+
    | glass/glass   | open rack      | -3.47 | -0.0594 | 3                   |
    +---------------+----------------+-------+---------+---------------------+
    | glass/glass   | close roof     | -2.98 | -0.0471 | 1                   |
    +---------------+----------------+-------+---------+---------------------+
    | glass/polymer | open rack      | -3.56 | -0.075  | 3                   |
    +---------------+----------------+-------+---------+---------------------+
    | glass/polymer | insulated back | -2.81 | -0.0455 | 0                   |
    +---------------+----------------+-------+---------+---------------------+

    References
    ----------
    [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
    Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
    NM.

    '''
    module_temperature = sapm_module(poa_global, air_temperature, wind_speed,
                                     a, b)
    return module_temperature + (poa_global / irrad_ref) * (deltaT)


def sapm_module(poa_global, air_temperature, wind_speed, a, b):
    r'''
    Calculate module back surface temperature per the Sandia PV Array
    Performance Model [1].

    Parameters
    ----------
    poa_global : float or Series
        Total incident irradiance [W/m^2].

    air_temperature : float or Series
        Ambient dry bulb temperature [C].

    wind_speed : float or Series
        Wind speed at a height of 10 meters [m/s].

    a : float
        Parameter :math:`a` in :eq:`sapm1`.

    b : float
        Parameter :math:`b` in :eq:`sapm1`.

    Returns
    -------
    float or Series, values in degrees C.

    Notes
    -----
    The model for module temperature :math:`T_{m}` is given by Eq. 11 in [1].

    .. math::
       :label: sapm1
        T_{m} = E \times \exp (a + b \times WS) + T_{a}

    Inputs to the model are plane-of-array irradiance :math:`E` (W/m2) and
    ambient air temperature :math:`T_{a}` (C). Model outputs are surface
    temperature at the back of the module :math:`T_{m}` and cell temperature
    :math:`T_{C}`. Model parameters depend both on the module construction and
    its mounting. Parameter sets are provided in [1] for representative modules
    and mounting.

    +---------------+----------------+-------+---------+---------------------+
    | Module        | Mounting       | a     | b       | :math:`\Delta T [C]`|
    +===============+================+=======+=========+=====================+
    | glass/glass   | open rack      | -3.47 | -0.0594 | 3                   |
    +---------------+----------------+-------+---------+---------------------+
    | glass/glass   | close roof     | -2.98 | -0.0471 | 1                   |
    +---------------+----------------+-------+---------+---------------------+
    | glass/polymer | open rack      | -3.56 | -0.075  | 3                   |
    +---------------+----------------+-------+---------+---------------------+
    | glass/polymer | insulated back | -2.81 | -0.0455 | 0                   |
    +---------------+----------------+-------+---------+---------------------+

    References
    ----------
    [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
    Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
    NM.

    '''
    return poa_global * np.exp(a + b * wind_speed) + air_temperature


def pvsyst_cell(poa_global, air_temperature, wind_speed=1.0,
                constant_loss_factor=29.0, wind_loss_factor=0.0, eta_m=0.1,
                alpha_absorption=0.9):
    r"""
    Calculate cell temperature using an empirical heat loss factor model
    as implemented in PVsyst.

    Parameters
    ----------
    poa_global : float or Series
        Total incident irradiance [W/m^2].

    air_temperature : float or Series
        Ambient dry bulb temperature [C].

    wind_speed : float or Series, default 1.0
        Wind speed in m/s measured at the same height for which the wind loss
        factor was determined.  The default value 1.0 m/2 is the wind
        speed at module height used to determine NOCT. [m/s]

    u_c : float, default 29.0
        Combined heat loss factor coefficient. The default value is
        representative of freestanding modules with the rear surfaces exposed
        to open air (e.g., rack mounted). Parameter :math:`U_{c}` in
        :eq:`pvsyst` [W/(m^2 C)].

    u_v : float, default 0.0
        Combined heat loss factor influenced by wind. Parameter :math:`U_{v}`
        in :eq:`pvsyst` [(W/m^2 C)(m/s)].

    eta_m : numeric, default 0.1
        Module external efficiency as a fraction, i.e., DC power / poa_global.
        Parameter :math:`\eta_{m}` in :eq:`pvsyst`.

    alpha_absorption : numeric, default 0.9
        Absorption coefficient. Parameter :math:`\alpha` in :eq:`pvsyst`.

    Returns
    -------
    float or Series, values in degrees Celsius

    Notes
    -----
    The Pvsyst model for cell temperature :math:`T_{C}` is given by

    .. math::
       :label: pvsyst

        T_{C} = T_{a} + \frac{\alpha E (1 - \eta_{m})}{U_{c} + U_{v} \times WS}

    Inputs to the model are plane-of-array irradiance :math:`E` (W/m2), ambient
    air temperature :math:`T_{a}` (C) and wind speed :math:`WS` (m/s). Model
    output is cell temperature :math:`T_{C}`. Model parameters depend both on
    the module construction and its mounting. Parameters are provided in
    [1] for open (freestanding) and close (insulated) mounting configurations.
    The heat loss factors provided represent the combined effect of convection,
    radiation and conduction, and their values are experimentally determined.

    +--------------+---------------+---------------+
    | Mounting     | :math:`U_{c}` | :math:`U_{v}` |
    +==============+===============+===============+
    | freestanding | 29.0          | 0.0           |
    +--------------+---------------+---------------+
    | insulated    | 15.0          | 0.0           |
    +--------------+---------------+---------------+

    References
    ----------
    [1]"PVsyst 6 Help", Files.pvsyst.com, 2018. [Online]. Available:
    http://files.pvsyst.com/help/index.html. [Accessed: 10- Dec- 2018].

    [2] Faiman, D. (2008). "Assessing the outdoor operating temperature of
    photovoltaic modules." Progress in Photovoltaics 16(4): 307-315.
    """

    total_loss_factor = wind_loss_factor * wind_speed + constant_loss_factor
    heat_input = poa_global * alpha_absorption * (1 - eta_m)
    temp_difference = heat_input / total_loss_factor
    return air_temperature + temp_difference
