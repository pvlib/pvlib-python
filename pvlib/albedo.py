"""
The ``albedo`` module contains functions for modeling albedo.
"""

from pvlib.tools import sind
import numpy as np
import pandas as pd


# Sources of for the albedo values are provided in
# pvlib.irradiance.get_ground_diffuse.
SURFACE_ALBEDOS = {
    'urban': 0.18,
    'grass': 0.20,
    'fresh grass': 0.26,
    'soil': 0.17,
    'sand': 0.40,
    'snow': 0.65,
    'fresh snow': 0.75,
    'asphalt': 0.12,
    'concrete': 0.30,
    'aluminum': 0.85,
    'copper': 0.74,
    'fresh steel': 0.35,
    'dirty steel': 0.08,
    'sea': 0.06,
}

WATER_COLOR_COEFFS = {
    'clear_water_no_waves': 0.13,
    'clear_water_ripples_up_to_2.5cm': 0.16,
    'clear_water_ripples_larger_than_2.5cm_occasional_whitecaps': 0.23,
    'clear_water_frequent_whitecaps': 0.3,
    'green_water_ripples_up_to_2.5cm': 0.22,
    'muddy_water_no_waves': 0.19
}

WATER_ROUGHNESS_COEFFS = {
    'clear_water_no_waves': 0.29,
    'clear_water_ripples_up_to_2.5cm': 0.7,
    'clear_water_ripples_larger_than_2.5cm_occasional_whitecaps': 1.25,
    'clear_water_frequent_whitecaps': 2,
    'green_water_ripples_up_to_2.5cm': 0.7,
    'muddy_water_no_waves': 0.29
}


def inland_water_dvoracek(solar_elevation, surface_condition=None,
                          color_coeff=None, wave_roughness_coeff=None):
    r"""
    Estimation of albedo for inland water bodies.

    The available surface conditions are for inland water bodies, e.g., lakes
    and ponds. For ocean/open sea, see
    :py:const:`pvlib.albedo.SURFACE_ALBEDOS`.

    Parameters
    ----------
    solar_elevation : numeric
        Sun elevation angle. [degrees]

    surface_condition : string, optional
        If supplied, overrides ``color_coeff`` and ``wave_roughness_coeff``.
        ``surface_condition`` can be one of the following:

        * ``'clear_water_no_waves'``
        * ``'clear_water_ripples_up_to_2.5cm'``
        * ``'clear_water_ripples_larger_than_2.5cm_occasional_whitecaps'``
        * ``'clear_water_frequent_whitecaps'``
        * ``'green_water_ripples_up_to_2.5cm'``
        * ``'muddy_water_no_waves'``

    color_coeff : float, optional
        Water color coefficient. [-]

    wave_roughness_coeff : float, optional
        Water wave roughness coefficient. [-]

    Returns
    -------
    albedo : numeric
        Albedo for inland water bodies. [-]

    Raises
    ------
    ValueError
        If neither of ``surface_condition`` nor a combination of
        ``color_coeff`` and ``wave_roughness_coeff`` are given.
        If ``surface_condition`` and any of ``color_coeff`` or
        ``wave_roughness_coeff`` are given. These parameters are
        mutually exclusive.

    KeyError
        If ``surface_condition`` is invalid.

    Notes
    -----
    The equation for calculating the albedo :math:`\rho` is given by

    .. math::
       :label: albedo

        \rho = c^{(r \cdot \sin(\alpha) + 1)}

    Inputs to the model are the water color coefficient :math:`c` [-], the
    water wave roughness coefficient :math:`r` [-] and the solar elevation
    :math:`\alpha` [degrees]. Parameters are provided in [1]_ , and are coded
    for convenience in :data:`~pvlib.albedo.WATER_COLOR_COEFFS` and
    :data:`~pvlib.albedo.WATER_ROUGHNESS_COEFFS`. The values of these
    coefficients are experimentally determined.

    +------------------------+-------------------+-------------------------+
    | Surface and condition  | Color coefficient | Wave roughness          |
    |                        | (:math:`c`)       | coefficient (:math:`r`) |
    +========================+===================+=========================+
    | Clear water, no waves  | 0.13              | 0.29                    |
    +------------------------+-------------------+-------------------------+
    | Clear water, ripples   | 0.16              | 0.70                    |
    | up to 2.5 cm           |                   |                         |
    +------------------------+-------------------+-------------------------+
    | Clear water, ripples   | 0.23              | 1.25                    |
    | larger than 2.5 cm     |                   |                         |
    | (occasional whitecaps) |                   |                         |
    +------------------------+-------------------+-------------------------+
    | Clear water,           | 0.30              | 2.00                    |
    | frequent whitecaps     |                   |                         |
    +------------------------+-------------------+-------------------------+
    | Green water, ripples   | 0.22              | 0.70                    |
    | up to 2.5cm            |                   |                         |
    +------------------------+-------------------+-------------------------+
    | Muddy water, no waves  | 0.19              | 0.29                    |
    +------------------------+-------------------+-------------------------+

    References
    ----------
    .. [1] Dvoracek M.J., Hannabas B. (1990). "Prediction of albedo for use in
       evapotranspiration and irrigation scheduling." IN: Visions of the Future
       American Society of Agricultural Engineers 04-90: 692-699.
    """

    if surface_condition is not None and (
        color_coeff is None and wave_roughness_coeff is None
    ):
        # use surface_condition
        color_coeff = WATER_COLOR_COEFFS[surface_condition]
        wave_roughness_coeff = WATER_ROUGHNESS_COEFFS[surface_condition]

    elif surface_condition is None and not (
        color_coeff is None or wave_roughness_coeff is None
    ):
        # use provided color_coeff and wave_roughness_coeff
        pass
    else:
        raise ValueError(
            "Either a `surface_condition` has to be chosen or"
            " a combination of `color_coeff` and"
            " `wave_roughness_coeff`.")

    solar_elevation_positive = np.where(solar_elevation < 0, 0,
                                        solar_elevation)

    albedo = color_coeff ** (wave_roughness_coeff *
                             sind(solar_elevation_positive) + 1)

    if isinstance(solar_elevation, pd.Series):
        albedo = pd.Series(albedo, index=solar_elevation.index)

    return albedo
