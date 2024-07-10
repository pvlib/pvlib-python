"""The ``windspeed`` module contains functions for calculating wind speed."""

import numpy as np
import pandas as pd


# Values of the Hellmann exponent
HELLMANN_SURFACE_EXPONENTS = {
    'unstable_air_above_open_water_surface': 0.06,
    'neutral_air_above_open_water_surface': 0.10,
    'stable_air_above_open_water_surface': 0.27,
    'unstable_air_above_flat_open_coast': 0.11,
    'neutral_air_above_flat_open_coast': 0.16,
    'stable_air_above_flat_open_coast': 0.40,
    'unstable_air_above_human_inhabited_areas': 0.27,
    'neutral_air_above_human_inhabited_areas': 0.34,
    'stable_air_above_human_inhabited_areas': 0.60,
}


def wind_speed_at_height_hellmann(wind_speed_measured, height_measured,
                                  height_desired, exponent=None,
                                  surface_type=None):
    r"""
    Estimate wind speed for different heights.

    The model is based on the power law equation by Hellmann [1]_, [2]_.

    Parameters
    ----------
    wind_speed_measured : numeric
        Measured wind speed. [m/s]

    height_measured : float
        The height at which the wind speed is measured. [m]

    height_desired : float
        The height at which the wind speed will be estimated. [m]

    exponent : float, optional
        Exponent based on the surface type. [-]

    surface_type : string, optional
        If supplied, overrides ``exponent``. Can be one of the following
        (see [1]_):

        * ``'unstable_air_above_open_water_surface'``
        * ``'neutral_air_above_open_water_surface'``
        * ``'stable_air_above_open_water_surface'``
        * ``'unstable_air_above_flat_open_coast'``
        * ``'neutral_air_above_flat_open_coast'``
        * ``'stable_air_above_flat_open_coast'``
        * ``'unstable_air_above_human_inhabited_areas'``
        * ``'neutral_air_above_human_inhabited_areas'``
        * ``'stable_air_above_human_inhabited_areas'``

    Returns
    -------
    wind_speed : numeric
        Adjusted wind speed for the desired height. [m/s]

    Raises
    ------
    ValueError
        If neither of ``exponent`` nor a ``surface_type`` is given.
        If both ``exponent`` nor a ``surface_type`` is given. These parameters
        are mutually exclusive.

    KeyError
        If the specified ``surface_type`` is invalid.

    Notes
    -----
    The equation for calculating the wind speed at a height of :math:`h` is
    given by the following power law equation [1]_, [2]_:

    .. math::
       :label: wind speed

        \windspeed_h = windspeed_{href} \cdot (\frac{h}{h_{ref}})^a

    where :math:`h` [m] is the height at which we would like to calculate the
    wind speed, :math:`h_{ref}` [m] is the reference height at which the wind
    speed is known, and :math:`windspeed_h` [m/s] and :math:`windspeed_{href}`
    [m/s] are the corresponding wind speeds at these heights. :math:`a` is a
    value that depends on the surface type. Some values found in the literature
    [1]_ for :math:`a` are the following:

    .. table:: Values for the Hellmann-exponent

    +-----------+--------------------+------------------+------------------+
    | Stability | Open water surface | Flat, open coast | Cities, villages |
    +===========+====================+==================+==================+
    | Unstable  | 0.06               | 0.10             | 0.27             |
    +-----------+--------------------+------------------+------------------+
    | Neutral   | 0.11               | 0.16             | 0.40             |
    +-----------+--------------------+------------------+------------------+
    | Stable    | 0.27               | 0.34             | 0.60             |
    +-----------+--------------------+------------------+------------------+

    In a report by SANDIA [3]_, this equation was experimentally tested for a
    height of 30 ft (9.144 m) and the following coefficients were recommended:
    :math:`h_{ref} = 9.144` [m], :math:`a = 0.219` [-], and
    :math:`windspeed_{href}` is the wind speed at a height of 9.144 [m].

    It should be noted that the equation retuns a value of NaN if the
    calculated wind speed is negative or a complex number.

    Warning
    -------
    Module temperature functions often require wind speeds at a heigt of 10 m
    and not the wind speed at the module height.

    For example, the following temperature functions require the input wind
    speed to be 10 m: :py:func:`~pvlib.temperature.sapm_cell`,
    :py:func:`~pvlib.temperature.sapm_module`, and
    :py:func:`~pvlib.temperature.generic_linear`, whereas the
    :py:func:`~pvlib.temperature.fuentes` model requires wind speed at 9.144 m.

    Additionally, the heat loss coefficients of some models have been developed
    for wind speed measurements at 10 m (e.g.,
    :py:func:`~pvlib.temperature.pvsyst_cell`,
    :py:func:`~pvlib.temperature.faiman`, and
    :py:func:`~pvlib.temperature.faiman_rad`).

    References
    ----------
    .. [1] Kaltschmitt M., Streicher W., Wiese A. (2007). "Renewable Energy:
       Technology, Economics and Environment." Springer,
       :doi:`10.1007/3-540-70949-5`.

    .. [2] Hellmann G. (1915). "Über die Bewegung der Luft in den untersten
       Schichten der Atmosphäre." Meteorologische Zeitschrift, 32

    .. [3] Menicucci D.F., Hall I.J. (1985). "Estimating wind speed as a
       function of height above ground: An analysis of data obtained at the
       southwest residential experiment station, Las Cruses, New Mexico."
       SAND84-2530, Sandia National Laboratories.
       `link <https://www.osti.gov/biblio/5628264>`_
    """
    if surface_type is not None and exponent is None:
        # use the Hellmann exponent from dictionary
        exponent = HELLMANN_SURFACE_EXPONENTS[surface_type]
    elif surface_type is None and exponent is not None:
        # use the provided exponent
        pass
    else:
        raise ValueError(
            "Either a `surface_type` has to be chosen or an exponent")

    wind_speed = wind_speed_measured * (
        (height_desired / height_measured) ** exponent)

    # if the provided height is negative the calculated wind speed is complex
    # so a NaN value is returned
    if isinstance(wind_speed, complex):
        wind_speed = np.nan

    # if wind speed is nagative return NaN
    wind_speed = np.where(wind_speed < 0, np.nan, wind_speed)

    if isinstance(wind_speed_measured, pd.Series):
        wind_speed = pd.Series(wind_speed, index=wind_speed_measured.index)

    return wind_speed
