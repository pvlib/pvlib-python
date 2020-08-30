"""
The ``temperature`` module contains functions for modeling temperature of
PV modules and cells.
"""

import numpy as np
import pandas as pd


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


def _temperature_model_params(model, parameter_set):
    try:
        params = TEMPERATURE_MODEL_PARAMETERS[model]
        return params[parameter_set]
    except KeyError:
        msg = ('{} is not a named set of parameters for the {} cell'
               ' temperature model.'
               ' See pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS'
               ' for names'.format(parameter_set, model))
        raise KeyError(msg)


def sapm_cell(poa_global, temp_air, wind_speed, a, b, deltaT,
              irrad_ref=1000):
    r'''
    Calculate cell temperature per the Sandia Array Performance Model.

    See [1]_ for details on the Sandia Array Performance Model.

    Parameters
    ----------
    poa_global : numeric
        Total incident irradiance [W/m^2].

    temp_air : numeric
        Ambient dry bulb temperature [C].

    wind_speed : numeric
        Wind speed at a height of 10 meters [m/s].

    a : float
        Parameter :math:`a` in :eq:`sapm1`.

    b : float
        Parameter :math:`b` in :eq:`sapm1`.

    deltaT : float
        Parameter :math:`\Delta T` in :eq:`sapm2` [C].

    irrad_ref : float, default 1000
        Reference irradiance, parameter :math:`E_{0}` in
        :eq:`sapm2` [W/m^2].

    Returns
    -------
    numeric, values in degrees C.

    Notes
    -----
    The model for cell temperature :math:`T_{C}` is given by a pair of
    equations (Eq. 11 and 12 in [1]_).

    .. math::
       :label: sapm1

       T_{m} = E \times \exp (a + b \times WS) + T_{a}

    .. math::
       :label: sapm2

       T_{C} = T_{m} + \frac{E}{E_{0}} \Delta T

    The module back surface temperature :math:`T_{m}` is implemented in
    :py:func:`~pvlib.temperature.sapm_module`.

    Inputs to the model are plane-of-array irradiance :math:`E` (W/m2) and
    ambient air temperature :math:`T_{a}` (C). Model parameters depend both on
    the module construction and its mounting. Parameter sets are provided in
    [1]_ for representative modules and mounting, and are coded for convenience
    in ``pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS``.

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
    .. [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
       Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
       NM.

    See also
    --------
    sapm_cell_from_module
    sapm_module

    Examples
    --------
    >>> from pvlib.temperature import sapm_cell, TEMPERATURE_MODEL_PARAMETERS
    >>> params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    >>> sapm_cell(1000, 10, 0, **params)
    44.11703066106086
    '''
    module_temperature = sapm_module(poa_global, temp_air, wind_speed,
                                     a, b)
    return sapm_cell_from_module(module_temperature, poa_global, deltaT,
                                 irrad_ref)


def sapm_module(poa_global, temp_air, wind_speed, a, b):
    r'''
    Calculate module back surface temperature per the Sandia Array
    Performance Model.

    See [1]_ for details on the Sandia Array Performance Model.

    Parameters
    ----------
    poa_global : numeric
        Total incident irradiance [W/m^2].

    temp_air : numeric
        Ambient dry bulb temperature [C].

    wind_speed : numeric
        Wind speed at a height of 10 meters [m/s].

    a : float
        Parameter :math:`a` in :eq:`sapm1mod`.

    b : float
        Parameter :math:`b` in :eq:`sapm1mod`.

    Returns
    -------
    numeric, values in degrees C.

    Notes
    -----
    The model for module temperature :math:`T_{m}` is given by Eq. 11 in [1]_.

    .. math::
       :label: sapm1mod

       T_{m} = E \times \exp (a + b \times WS) + T_{a}

    Inputs to the model are plane-of-array irradiance :math:`E` (W/m2) and
    ambient air temperature :math:`T_{a}` (C). Model outputs are surface
    temperature at the back of the module :math:`T_{m}` and cell temperature
    :math:`T_{C}`. Model parameters depend both on the module construction and
    its mounting. Parameter sets are provided in [1]_ for representative
    modules and mounting, and are coded for convenience in
    ``temperature.TEMPERATURE_MODEL_PARAMETERS``.

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
    .. [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
       Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
       NM.

    See also
    --------
    sapm_cell
    sapm_cell_from_module
    '''
    return poa_global * np.exp(a + b * wind_speed) + temp_air


def sapm_cell_from_module(module_temperature, poa_global, deltaT,
                          irrad_ref=1000):
    r'''
    Calculate cell temperature from module temperature using the Sandia Array
    Performance Model.

    See [1]_ for details on the Sandia Array Performance Model.

    Parameters
    ----------
    module_temperature : numeric
        Temperature of back of module surface [C].

    poa_global : numeric
        Total incident irradiance [W/m^2].

    deltaT : float
        Parameter :math:`\Delta T` in :eq:`sapm2_cell_from_mod` [C].

    irrad_ref : float, default 1000
        Reference irradiance, parameter :math:`E_{0}` in
        :eq:`sapm2` [W/m^2].

    Returns
    -------
    numeric, values in degrees C.

    Notes
    -----
    The model for cell temperature :math:`T_{C}` is given by Eq. 12 in [1]_.

    .. math::
       :label: sapm2_cell_from_mod

       T_{C} = T_{m} + \frac{E}{E_{0}} \Delta T

    The module back surface temperature :math:`T_{m}` is implemented in
    :py:func:`~pvlib.temperature.sapm_module`.

    Model parameters depend both on the module construction and its mounting.
    Parameter sets are provided in [1]_ for representative modules and
    mounting, and are coded for convenience in
    ``pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS``.

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
    .. [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
       Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
       NM.

    See also
    --------
    sapm_cell
    sapm_module
    '''
    return module_temperature + (poa_global / irrad_ref) * deltaT


def pvsyst_cell(poa_global, temp_air, wind_speed=1.0, u_c=29.0, u_v=0.0,
                eta_m=0.1, alpha_absorption=0.9):
    r"""
    Calculate cell temperature using an empirical heat loss factor model
    as implemented in PVsyst.

    Parameters
    ----------
    poa_global : numeric
        Total incident irradiance [W/m^2].

    temp_air : numeric
        Ambient dry bulb temperature [C].

    wind_speed : numeric, default 1.0
        Wind speed in m/s measured at the same height for which the wind loss
        factor was determined.  The default value 1.0 m/2 is the wind
        speed at module height used to determine NOCT. [m/s]

    u_c : float, default 29.0
        Combined heat loss factor coefficient. The default value is
        representative of freestanding modules with the rear surfaces exposed
        to open air (e.g., rack mounted). Parameter :math:`U_{c}` in
        :eq:`pvsyst`.
        :math:`\left[\frac{\text{W}/{\text{m}^2}}{\text{C}}\right]`

    u_v : float, default 0.0
        Combined heat loss factor influenced by wind. Parameter :math:`U_{v}`
        in :eq:`pvsyst`.
        :math:`\left[ \frac{\text{W}/\text{m}^2}{\text{C}\ \left( \text{m/s} \right)} \right]`

    eta_m : numeric, default 0.1
        Module external efficiency as a fraction, i.e., DC power / poa_global.
        Parameter :math:`\eta_{m}` in :eq:`pvsyst`.

    alpha_absorption : numeric, default 0.9
        Absorption coefficient. Parameter :math:`\alpha` in :eq:`pvsyst`.

    Returns
    -------
    numeric, values in degrees Celsius

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
    [1]_ for open (freestanding) and close (insulated) mounting configurations,
    , and are coded for convenience in
    ``temperature.TEMPERATURE_MODEL_PARAMETERS``. The heat loss factors
    provided represent the combined effect of convection, radiation and
    conduction, and their values are experimentally determined.

    +--------------+---------------+---------------+
    | Mounting     | :math:`U_{c}` | :math:`U_{v}` |
    +==============+===============+===============+
    | freestanding | 29.0          | 0.0           |
    +--------------+---------------+---------------+
    | insulated    | 15.0          | 0.0           |
    +--------------+---------------+---------------+

    References
    ----------
    .. [1] "PVsyst 6 Help", Files.pvsyst.com, 2018. [Online]. Available:
       http://files.pvsyst.com/help/index.html. [Accessed: 10- Dec- 2018].

    .. [2] Faiman, D. (2008). "Assessing the outdoor operating temperature of
       photovoltaic modules." Progress in Photovoltaics 16(4): 307-315.

    Examples
    --------
    >>> from pvlib.temperature import pvsyst_cell, TEMPERATURE_MODEL_PARAMETERS
    >>> params = TEMPERATURE_MODEL_PARAMETERS['pvsyst']['freestanding']
    >>> pvsyst_cell(1000, 10, **params)
    37.93103448275862
    """

    total_loss_factor = u_c + u_v * wind_speed
    heat_input = poa_global * alpha_absorption * (1 - eta_m)
    temp_difference = heat_input / total_loss_factor
    return temp_air + temp_difference


def faiman(poa_global, temp_air, wind_speed=1.0, u0=25.0, u1=6.84):
    r'''
    Calculate cell or module temperature using the Faiman model.  The Faiman
    model uses an empirical heat loss factor model [1]_ and is adopted in the
    IEC 61853 standards [2]_ and [3]_.

    Usage of this model in the IEC 61853 standard does not distinguish
    between cell and module temperature.

    Parameters
    ----------
    poa_global : numeric
        Total incident irradiance [W/m^2].

    temp_air : numeric
        Ambient dry bulb temperature [C].

    wind_speed : numeric, default 1.0
        Wind speed in m/s measured at the same height for which the wind loss
        factor was determined.  The default value 1.0 m/s is the wind
        speed at module height used to determine NOCT. [m/s]

    u0 : numeric, default 25.0
        Combined heat loss factor coefficient. The default value is one
        determined by Faiman for 7 silicon modules.
        :math:`\left[\frac{\text{W}/{\text{m}^2}}{\text{C}}\right]`

    u1 : numeric, default 6.84
        Combined heat loss factor influenced by wind. The default value is one
        determined by Faiman for 7 silicon modules.
        :math:`\left[ \frac{\text{W}/\text{m}^2}{\text{C}\ \left( \text{m/s} \right)} \right]`

    Returns
    -------
    numeric, values in degrees Celsius

    Notes
    -----
    All arguments may be scalars or vectors. If multiple arguments
    are vectors they must be the same length.

    References
    ----------
    .. [1] Faiman, D. (2008). "Assessing the outdoor operating temperature of
       photovoltaic modules." Progress in Photovoltaics 16(4): 307-315.

    .. [2] "IEC 61853-2 Photovoltaic (PV) module performance testing and energy
       rating - Part 2: Spectral responsivity, incidence angle and module
       operating temperature measurements". IEC, Geneva, 2018.

    .. [3] "IEC 61853-3 Photovoltaic (PV) module performance testing and energy
       rating - Part 3: Energy rating of PV modules". IEC, Geneva, 2018.

    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Dec., 2019

    # The following lines may seem odd since u0 & u1 are probably scalar,
    # but it serves an indirect and easy way of allowing lists and
    # tuples for the other function arguments.
    u0 = np.asanyarray(u0)
    u1 = np.asanyarray(u1)

    total_loss_factor = u0 + u1 * wind_speed
    heat_input = poa_global
    temp_difference = heat_input / total_loss_factor
    return temp_air + temp_difference


def _fuentes_hconv(tave, windmod, tinoct, temp_delta, xlen, check_reynold):
    # Calculate the convective coefficient as in Fuentes 1987 -- a mixture of
    # free, laminar, and turbulent convection.
    densair = 0.003484 * 101325.0 / tave  # density
    visair = 0.24237e-6 * tave**0.76 / densair  # kinematic viscosity
    condair = 2.1695e-4 * tave**0.84  # thermal conductivity
    reynold = windmod * xlen / visair
    # the boundary between laminar and turbulent is modeled as an abrupt
    # change at Re = 1.2e5:
    if check_reynold and reynold > 1.2e5:
        # turbulent convection
        hforce = 0.0282 / reynold**0.2 * densair * windmod * 1007 / 0.71**0.4
    else:
        # laminar convection
        hforce = 0.8600 / reynold**0.5 * densair * windmod * 1007 / 0.71**0.67
    # free convection via Grashof number
    # NB: the 0.5 factor is from assuming tilt=30; should tilt be a parameter?
    grashof = 9.8 / tave * temp_delta * xlen**3 / visair**2 * 0.5
    # product of Nusselt number and (k/l)
    hfree = 0.21 * (grashof * 0.71)**0.32 * condair / xlen
    # combine free and forced components
    hconv = (hfree**3 + hforce**3)**(1/3)
    return hconv


def fuentes(poa_global, temp_air, wind_speed, inoct, module_height=5,
            wind_height=9.144, emissivity=0.84, absorption=0.83,
            hydraulic_diameter=0.5):
    """
    Calculate cell or module temperature using the Fuentes model.

    The Fuentes model is a first-principles heat transfer energy balance
    model [1]_ that is used in PVWatts for cell temperature modeling [2]_.

    Parameters
    ----------
    poa_global : pandas Series
        Total incident irradiance [W/m^2]

    temp_air : pandas Series
        Ambient dry bulb temperature [C]

    wind_speed : pandas Series
        Wind speed [m/s]

    inoct : float
        The "installed" nominal operating cell temperature as defined in [1]_.
        PVWatts assumes this value to be 45 C for rack-mounted arrays and
        49 C for roof mount systems with restricted air flow around the
        module.  [C]

    module_height : float, default 5.0
        The height above ground of the module above the ground. The PVWatts
        default is 5.0 [m]

    wind_height : float, default 9.144
        The height above ground at which ``wind_speed`` is measured. The
        PVWatts defauls is 9.144 [m]

    emissivity : float, default 0.84
        The effectiveness of the module at radiating thermal energy. [unitless]

    absorption : float, default 0.83
        The fraction of incident irradiance that is converted to thermal
        energy in the module. [unitless]

    hydraulic_diameter : float, default 0.5
        The hydraulic diameter of the module. The default value of 0.5 is
        provided in [1]_ for a module with dimensions 0.3m x 1.2m.  [m]

    Returns
    -------
    temperature_cell : pandas Series
        The modeled cell temperature [C]

    Notes
    -----
    This function returns slightly different values from PVWatts at night
    and just after dawn. This is because the SAM SSC assumes that module
    temperature equals ambient temperature when irradiance is zero so it can
    skip the heat balance calculation at night.

    References
    ----------
    .. [1] Fuentes, M. K. A Simplifed Thermal Model for Flat-Plate
           Photovoltaic Arrays. SAND85-0330. 1987.
           http://prod.sandia.gov/techlib/access-control.cgi/1985/850330.pdf
    .. [2] Dobos, A. P. PVWatts Version 5 Manual. NREL/TP-6A20-62641. 2014.
           doi:10.2172/1158421.
    """
    # ported from the FORTRAN77 code provided in Appendix A of Fuentes 1987;
    # nearly all variable names are kept the same for ease of comparison.

    boltz = 5.669e-8
    emiss = emissivity
    absorp = absorption
    xlen = hydraulic_diameter
    cap0 = 11000
    tinoct = inoct + 273.15

    # convective coefficient of top surface of module at NOCT
    windmod = 1.0
    tave = (tinoct + 293.15) / 2
    hconv = _fuentes_hconv(tave, windmod, tinoct, tinoct - 293.15, xlen, False)

    # determine the ground temperature ratio and the ratio of the total
    # convection to the top side convection
    hground = emiss * boltz * (tinoct**2 + 293.15**2) * (tinoct + 293.15)
    backrat = (
        absorp * 800.0
        - emiss * boltz * (tinoct**4 - 282.21**4)
        - hconv * (tinoct - 293.15)
    ) / ((hground + hconv) * (tinoct - 293.15))
    tground = (tinoct**4 - backrat * (tinoct**4 - 293.15**4))**0.25
    tground = np.clip(tground, 293.15, tinoct)

    tgrat = (tground - 293.15) / (tinoct - 293.15)
    convrat = (absorp * 800 - emiss * boltz * (
        2 * tinoct**4 - 282.21**4 - tground**4)) / (hconv * (tinoct - 293.15))

    # adjust the capacitance (thermal mass) of the module based on the INOCT.
    # It is a function of INOCT because high INOCT implies thermal coupling
    # with the racking (e.g. roofmount), so the thermal mass is increased.
    # `cap` has units J/(m^2 C) -- see Table 3, Equations 26 & 27
    cap = cap0
    if tinoct > 321.15:
        cap = cap * (1 + (tinoct - 321.15) / 12)

    # iterate through timeseries inputs
    sun0 = 0
    tmod0 = 293.15

    # n.b. the way Fuentes calculates the first timedelta makes it seem like
    # the value doesn't matter -- rather than recreate it here, just assume
    # it's the same as the second timedelta:
    timedelta_hours = np.diff(poa_global.index).astype(float) / 1e9 / 60 / 60
    timedelta_hours = np.append([timedelta_hours[0]], timedelta_hours)

    df = pd.DataFrame({
        'tamb': temp_air + 273.15,
        'sun': poa_global * absorp,
        'wind_speed': wind_speed,
        'dtime': timedelta_hours,
    })

    # Two of the calculations are easily vectorized, so precalculate them:
    # sky temperature -- Equation 24
    df['tsky'] = 0.68 * (0.0552 * df['tamb']**1.5) + 0.32 * df['tamb']
    # wind speed at module height -- Equation 22
    # not sure why the 1e-4 factor is included -- maybe the equations don't
    # behave well if wind == 0?
    df['windmod'] = df['wind_speed'] * (module_height/wind_height)**0.2 + 1e-4

    tmod0 = 293.15

    for idx, row in df.iterrows():

        tamb = row['tamb']
        sun = row['sun']
        windmod = row['windmod']
        tsky = row['tsky']
        dtime = row['dtime']

        # solve the heat transfer equation, iterating because the heat loss
        # terms depend on tmod. NB Fuentes doesn't show that 10 iterations is
        # sufficient for convergence.
        tmod = tmod0
        for j in range(10):
            # overall convective coefficient
            tave = (tmod + tamb) / 2
            hconv = convrat * _fuentes_hconv(tave, windmod, tinoct,
                                             abs(tmod-tamb), xlen, True)
            # sky radiation coefficient (Equation 3)
            hsky = emiss * boltz * (tmod**2 + tsky**2) * (tmod + tsky)
            # ground radiation coeffieicient (Equation 4)
            tground = tamb + tgrat * (tmod - tamb)
            hground = emiss * boltz * (tmod**2 + tground**2) * (tmod + tground)
            # thermal lag -- Equation 8
            eigen = - (hconv + hsky + hground) / cap * dtime * 3600
            # not sure why this check is done, maybe as a speed optimization?
            if eigen > -10:
                ex = np.exp(eigen)
            else:
                ex = 0
            # Equation 7 -- note that `sun` and `sun0` already account for
            # absorption (alpha)
            tmod = tmod0 * ex + (
                (1 - ex) * (
                    hconv * tamb
                    + hsky * tsky
                    + hground * tground
                    + sun0
                    + (sun - sun0) / eigen
                ) + sun - sun0
            ) / (hconv + hsky + hground)

        df.loc[idx, 'tmod'] = tmod
        tmod0 = tmod
        sun0 = sun

    return df['tmod'] - 273.15
