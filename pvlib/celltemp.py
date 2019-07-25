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
        Parameter :math:`a` in :eq:`sapm1`.

    b : float
        Parameter :math:`b` in :eq:`sapm1`.

    deltaT : float, [C]
        Parameter :math:`\Delta T` in :eq:`sapm2`.

    irrad_ref : float, default 1000 [W/m2]
        Reference irradiance, parameter :math:`E_{0}` in :eq:`sapm2`.

    Returns
    --------
    DataFrame with columns 'temp_cell' and 'temp_module'.
    Values in degrees C.

    Notes
    -----
    The model for cell temperature :math:`T_{C}` and module temperature
    :math:`T_{m}` is given by a pair of equations (Eq. 11 and 12 in [1]).

    .. math::
       :label: sapm1

        T_{m} = E \times \exp (a + b \times WS) + T_{a}

    .. math::
       :label: sapm2

        T_{C} = T_{m} + \frac{E}{E_{0}} \Delta T

    Inputs to the model are plane-of-array irradiance :math:`E` (W/m2) and
    ambient air temperature :math:`T_{a}` (C). Model outputs are surface
    temperature at the back of the module :math:`T_{m}` and cell temperature
    :math:`T_{C}`. Model parameters depend both on the module construction and
    its mounting. Parameter sets are provided in [1] for representative modules
    and mounting.

    +--------------------+------------------+-------+---------+---------------------+  # noqa: E501
    | Module             | Mounting         | a     | b       | :math:`\Delta T [C]`|  # noqa: E501
    +====================+==================+=======+=========+=====================+  # noqa: E501
    | glass/cell/glass   | open rack        | -3.47 | -0.0594 | 3                   |  # noqa: E501
    +--------------------+------------------+-------+---------+---------------------+  # noqa: E501
    | glass/cell/glass   | close roof mount | -2.98 | -0.0471 | 1                   |  # noqa: E501
    +--------------------+------------------+-------+---------+---------------------+  # noqa: E501
    | glass/cell/polymer | open rack        | -3.56 | -0.075  | 3                   |  # noqa: E501
    +--------------------+------------------+-------+---------+---------------------+  # noqa: E501
    | glass/cell/polymer | insulated back   | -2.81 | -0.0455 | 0                   |  # noqa: E501
    +--------------------+------------------+-------+---------+---------------------+  # noqa: E501

    References
    ----------
    [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
    Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
    NM.

    '''
    temp_module = pd.Series(poa_global * np.exp(a + b * wind_speed) + temp_air)
    temp_cell = temp_module + (poa_global / irrad_ref) * (deltaT)
    return pd.DataFrame({'temp_cell': temp_cell, 'temp_module': temp_module})


def pvsyst(poa_global, temp_air, wind_speed=1.0, constant_loss_factor=29.0,
           wind_loss_factor=0.0, eta_m=0.1, alpha_absorption=0.9):
    r"""
    Calculate cell temperature using an empirical heat loss factor model
    as implemented in PVsyst.

    The heat loss factors provided through the 'model' argument
    represent the combined effect of convection, radiation and conduction,
    and their values are experimentally determined.

    Parameters
    ----------
    poa_global : float or Series
        Total incident irradiance [:math:`\frac{W}{m^2} ].

    temp_air : float or Series
        Ambient dry bulb temperature [C].

    wind_speed : float or Series, default 1.0 [m/s]
        Wind speed in m/s measured at the same height for which the wind loss
        factor was determined.  The default value is 1.0, which is the wind
        speed at module height used to determine NOCT.

    constant_loss_factor : float, default 29.0 [:math:`\frac{W}{m^2 C}]
        Combined heat loss factor coefficient. The default value is
        representative of freestanding modules with the rear surfaces exposed
        to open air (e.g., rack mounted). Parameter :math:`U_{c}` in
        :eq:`pvsyst`.

    wind_loss_factor : float, default 0.0 [:math:`\frac{W}{m^2 C} \frac{m}{s}`]
        Combined heat loss factor influenced by wind. Parameter :math:`U_{c}`
        in :eq:`pvsyst`.

    eta_m : numeric, default 0.1
        Module external efficiency as a fraction, i.e., DC power / poa_global.
        Parameter :math:`\eta_{m}` in :eq:`pvsyst`.

    alpha_absorption : numeric, default 0.9
        Absorption coefficient. Parameter :math:`\alpha` in :eq:`pvsyst`.

    Returns
    -------
    DataFrame with columns 'temp_cell', values in degrees Celsius

    Notes
    -----
    The Pvsyst model for cell temperature :math:`T_{C}` is given by

    .. math::
       :label: pvsyst

        T_{C} = T_{a} + \frac{\alpha E (1 - \eta_{m})}{U_{c} + U_{v} WS}

    Inputs to the model are plane-of-array irradiance :math:`E` (W/m2) and
    ambient air temperature :math:`T_{a}` (C). Model output is cell temperature
    :math:`T_{C}`. Model parameters depend both on the module construction and
    its mounting. Parameter sets are provided in [1] for open (freestanding)
    close (insulated) mounting configurations.

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
    temp_cell = pd.Series(temp_air + temp_difference)

    return pd.DataFrame({'temp_cell': temp_cell})
