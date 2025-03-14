"""
The ``atmosphere`` module contains methods to calculate relative and
absolute airmass, determine pressure from altitude or vice versa, and wind
speed at different heights.
"""

import numpy as np
import pandas as pd

APPARENT_ZENITH_MODELS = ('simple', 'kasten1966', 'kastenyoung1989',
                          'gueymard1993', 'pickering2002')
TRUE_ZENITH_MODELS = ('youngirvine1967', 'young1994')
AIRMASS_MODELS = APPARENT_ZENITH_MODELS + TRUE_ZENITH_MODELS


def pres2alt(pressure):
    '''
    Determine altitude from site pressure.

    Parameters
    ----------
    pressure : numeric
        Atmospheric pressure. [Pa]

    Returns
    -------
    altitude : numeric
        Altitude above sea level. [m]

    Notes
    ------
    The following assumptions are made

    ============================   ================
    Parameter                      Value
    ============================   ================
    Base pressure                  101325 Pa
    Temperature at zero altitude   288.15 K
    Gravitational acceleration     9.80665 m/s^2
    Lapse rate                     -6.5E-3 K/m
    Gas constant for air           287.053 J/(kg K)
    Relative Humidity              0%
    ============================   ================

    References
    -----------
    .. [1] "A Quick Derivation relating altitude to air pressure" from
       Portland State Aerospace Society, Version 1.03, 12/22/2004.
    '''

    alt = 44331.5 - 4946.62 * pressure ** (0.190263)

    return alt


def alt2pres(altitude):
    '''
    Determine site pressure from altitude.

    Parameters
    ----------
    altitude : numeric
        Altitude above sea level. [m]

    Returns
    -------
    pressure : numeric
        Atmospheric pressure. [Pa]

    Notes
    ------
    The following assumptions are made

    ============================   ================
    Parameter                      Value
    ============================   ================
    Base pressure                  101325 Pa
    Temperature at zero altitude   288.15 K
    Gravitational acceleration     9.80665 m/s^2
    Lapse rate                     -6.5E-3 K/m
    Gas constant for air           287.053 J/(kg K)
    Relative Humidity              0%
    ============================   ================

    References
    -----------
    .. [1] "A Quick Derivation relating altitude to air pressure" from
       Portland State Aerospace Society, Version 1.03, 12/22/2004.
    '''

    press = 100 * ((44331.514 - altitude) / 11880.516) ** (1 / 0.1902632)

    return press


def get_absolute_airmass(airmass_relative, pressure=101325.):
    r'''
    Determine absolute (pressure-adjusted) airmass from relative
    airmass and pressure.

    The calculation for absolute airmass (:math:`AM_a`) is

    .. math::
        AM_a = AM_r \frac{P}{101325}

    where :math:`AM_r` is relative air mass at sea level and :math:`P` is
    atmospheric pressure.

    Parameters
    ----------
    airmass_relative : numeric
        The airmass at sea level. [unitless]

    pressure : numeric, default 101325
        Atmospheric pressure. [Pa]

    Returns
    -------
    airmass_absolute : numeric
        Absolute (pressure-adjusted) airmass

    References
    ----------
    .. [1] C. Gueymard, "Critical analysis and performance assessment of
       clear sky solar irradiance models using theoretical and measured
       data," Solar Energy, vol. 51, pp. 121-138, 1993.
    '''

    airmass_absolute = airmass_relative * pressure / 101325.

    return airmass_absolute


def get_relative_airmass(zenith, model='kastenyoung1989'):
    '''
    Calculate relative (not pressure-adjusted) airmass at sea level.

    Parameter ``model`` allows selection of different airmass models.

    Parameters
    ----------
    zenith : numeric
        Zenith angle of the sun. [degrees]

    model : string, default 'kastenyoung1989'
        Available models include the following:

        * 'simple' - secant(apparent zenith angle) -
          Note that this gives -Inf at zenith=90
        * 'kasten1966' - See reference [1] -
          requires apparent sun zenith
        * 'youngirvine1967' - See reference [2] -
          requires true sun zenith
        * 'kastenyoung1989' (default) - See reference [3] -
          requires apparent sun zenith
        * 'gueymard1993' - See reference [4] -
          requires apparent sun zenith
        * 'young1994' - See reference [5] -
          requires true sun zenith
        * 'pickering2002' - See reference [6] -
          requires apparent sun zenith
        * 'gueymard2003' - See references [7] and [8] -
          requires apparent sun zenith

    Returns
    -------
    airmass_relative : numeric
        Relative airmass at sea level. Returns NaN values for any
        zenith angle greater than 90 degrees. [unitless]

    Notes
    -----
    Some models use apparent (refraction-adjusted) zenith angle while
    other models use true (not refraction-adjusted) zenith angle. Apparent
    zenith angles should be calculated at sea level.

    References
    ----------
    .. [1] Fritz Kasten. "A New Table and Approximation Formula for the
       Relative Optical Air Mass". Technical Report 136, Hanover, N.H.:
       U.S. Army Material Command, CRREL.

    .. [2] A. T. Young and W. M. Irvine, "Multicolor Photoelectric
       Photometry of the Brighter Planets," The Astronomical Journal, vol.
       72, pp. 945-950, 1967.

    .. [3] Fritz Kasten and Andrew Young. "Revised optical air mass tables
       and approximation formula". Applied Optics 28:4735-4738

    .. [4] C. Gueymard, "Critical analysis and performance assessment of
       clear sky solar irradiance models using theoretical and measured
       data," Solar Energy, vol. 51, pp. 121-138, 1993.

    .. [5] A. T. Young, "AIR-MASS AND REFRACTION," Applied Optics, vol. 33,
       pp. 1108-1110, Feb 1994.

    .. [6] Keith A. Pickering. "The Ancient Star Catalog". DIO 12:1, 20,

    .. [7] C. Gueymard, "Direct solar transmittance and irradiance
       predictions with broadband models. Part I: detailed theoretical
       performance assessment". Solar Energy, vol 74, pp. 355-379, 2003.
       :doi:`10.1016/S0038-092X(03)00195-6`

    .. [8] C. Gueymard (2019). Clear-Sky Radiation Models and Aerosol Effects.
       In: Polo, J., Martín-Pomares, L., Sanfilippo, A. (eds) Solar Resources
       Mapping. Green Energy and Technology. Springer, Cham.
       :doi:`10.1007/978-3-319-97484-2_5`

    .. [9] Matthew J. Reno, Clifford W. Hansen and Joshua S. Stein, "Global
       Horizontal Irradiance Clear Sky Models: Implementation and Analysis"
       Sandia Report, (2012).
    '''

    # set zenith values greater than 90 to nans
    z = np.where(zenith > 90, np.nan, zenith)
    zenith_rad = np.radians(z)

    model = model.lower()

    if 'kastenyoung1989' == model:
        am = (1.0 / (np.cos(zenith_rad) +
              0.50572*((6.07995 + (90 - z)) ** - 1.6364)))
    elif 'kasten1966' == model:
        am = 1.0 / (np.cos(zenith_rad) + 0.15*((93.885 - z) ** - 1.253))
    elif 'simple' == model:
        am = 1.0 / np.cos(zenith_rad)
    elif 'pickering2002' == model:
        am = (1.0 / (np.sin(np.radians(90 - z +
              244.0 / (165 + 47.0 * (90 - z) ** 1.1)))))
    elif 'youngirvine1967' == model:
        sec_zen = 1.0 / np.cos(zenith_rad)
        am = sec_zen * (1 - 0.0012 * (sec_zen * sec_zen - 1))
    elif 'young1994' == model:
        am = ((1.002432*((np.cos(zenith_rad)) ** 2) +
              0.148386*(np.cos(zenith_rad)) + 0.0096467) /
              (np.cos(zenith_rad) ** 3 +
              0.149864*(np.cos(zenith_rad) ** 2) +
              0.0102963*(np.cos(zenith_rad)) + 0.000303978))
    elif 'gueymard1993' == model:
        am = (1.0 / (np.cos(zenith_rad) +
              0.00176759*(z)*((94.37515 - z) ** - 1.21563)))
    elif 'gueymard2003' == model:
        am = (1.0 / (np.cos(zenith_rad) +
              0.48353*(z**0.095846)/(96.741 - z)**1.754))
    else:
        raise ValueError('%s is not a valid model for relativeairmass', model)

    if isinstance(zenith, pd.Series):
        am = pd.Series(am, index=zenith.index)

    return am


def gueymard94_pw(temp_air, relative_humidity):
    r"""
    Calculates precipitable water (cm) from ambient air temperature (C)
    and relatively humidity (%) using an empirical model. The
    accuracy of this method is approximately 20% for moderate PW (1-3
    cm) and less accurate otherwise.

    The model was developed by expanding Eq. 1 in [2]_:

    .. math::

           Pw = 0.1 H_v \rho_v

    using Eq. 2 in [2]_

    .. math::

           \rho_v = 216.7 R_H e_s /T

    :math:`Pw` is the precipitable water (cm), :math:`H_v` is the apparent
    water vapor scale height (km) and :math:`\rho_v` is the surface water
    vapor density (g/m^3). . The expression for :math:`H_v` is Eq. 4 in [2]_:

    .. math::

           H_v = 0.4976 + 1.5265 \frac{T}{273.15}
               + \exp \left(13.6897 \frac{T}{273.15}
               - 14.9188 \left( \frac{T}{273.15} \right)^3 \right)

    In the expression for :math:`\rho_v`, :math:`e_s` is the saturation water
    vapor pressure (millibar). The expression for :math:`e_s` is Eq. 1 in [3]_

    .. math::

          e_s = \exp \left(22.330 - 49.140 \frac{100}{T} -
              10.922 \left(\frac{100}{T}\right)^2 -
              0.39015 \frac{T}{100} \right)

    Parameters
    ----------
    temp_air : numeric
        ambient air temperature :math:`T` at the surface. [C]
    relative_humidity : numeric
        relative humidity :math:`R_H` at the surface. [%]

    Returns
    -------
    pw : numeric
        precipitable water. [cm]

    References
    ----------
    .. [1] W. M. Keogh and A. W. Blakers, Accurate Measurement, Using Natural
       Sunlight, of Silicon Solar Cells, Prog. in Photovoltaics: Res.
       and Appl. 2004, vol 12, pp. 1-19 (:doi:`10.1002/pip.517`)

    .. [2] C. Gueymard, Analysis of Monthly Average Atmospheric Precipitable
       Water and Turbidity in Canada and Northern United States,
       Solar Energy vol 53(1), pp. 57-71, 1994.

    .. [3] C. Gueymard, Assessment of the Accuracy and Computing Speed of
       simplified saturation vapor equations using a new reference
       dataset, J. of Applied Meteorology 1993, vol. 32(7), pp.
       1294-1300.
    """

    T = temp_air + 273.15  # Convert to Kelvin                  # noqa: N806
    RH = relative_humidity                                      # noqa: N806

    theta = T / 273.15

    # Eq. 1 from Keogh and Blakers
    pw = (
        0.1 *
        (0.4976 + 1.5265*theta + np.exp(13.6897*theta - 14.9188*(theta)**3)) *
        (216.7*RH/(100*T)*np.exp(22.330 - 49.140*(100/T) -
         10.922*(100/T)**2 - 0.39015*T/100)))

    pw = np.maximum(pw, 0.1)

    return pw


def rh_from_tdew(temp_air, temp_dew, coeff=(6.112, 17.62, 243.12)):
    """
    Calculate relative humidity from dewpoint temperature using the Magnus
    equation.

    Parameters
    ----------
    temp_air : numeric
        Air temperature (dry-bulb temperature). [°C]
    temp_dew : numeric
        Dew-point temperature. [°C]
    coeff : tuple, default (6.112, 17.62, 243.12)
        Magnus equation coefficients (A, B, C).  The default values are those
        recommended by the WMO [1]_.

    Returns
    -------
    numeric
        Relative humidity (0.0-100.0). [%]

    References
    ----------
    .. [1] "Guide to Instruments and Methods of Observation",
       World Meteorological Organization, WMO-No. 8, 2023.
       https://library.wmo.int/idurl/4/68695
    """

    # Calculate vapor pressure (e) and saturation vapor pressure (es)
    e = coeff[0] * np.exp((coeff[1] * temp_air) / (coeff[2] + temp_air))
    es = coeff[0] * np.exp((coeff[1] * temp_dew) / (coeff[2] + temp_dew))

    # Calculate relative humidity as percentage
    relative_humidity = 100 * (es / e)

    return relative_humidity


def tdew_from_rh(temp_air, relative_humidity, coeff=(6.112, 17.62, 243.12)):
    """
    Calculate dewpoint temperature using the Magnus equation.
    This is a reversal of the calculation in :py:func:`rh_from_tdew`.

    Parameters
    ----------
    temp_air : numeric
        Air temperature (dry-bulb temperature). [°C]
    relative_humidity : numeric
        Relative humidity (0-100). [%]
    coeff: tuple, default (6.112, 17.62, 243.12)
        Magnus equation coefficients (A, B, C).  The default values are those
        recommended by the WMO [1]_.

    Returns
    -------
    numeric
        Dewpoint temperature. [°C]

    References
    ----------
    .. [1] "Guide to Instruments and Methods of Observation",
       World Meteorological Organization, WMO-No. 8, 2023.
       https://library.wmo.int/idurl/4/68695
    """
    # Calculate the term inside the log
    # From RH = 100 * (es/e), we get es = (RH/100) * e
    # Substituting the Magnus equation and solving for dewpoint

    # First calculate ln(es/A)
    ln_term = (
        (coeff[1] * temp_air) / (coeff[2] + temp_air)
        + np.log(relative_humidity/100)
    )

    # Then solve for dewpoint
    dewpoint = coeff[2] * ln_term / (coeff[1] - ln_term)

    return dewpoint


def bird_hulstrom80_aod_bb(aod380, aod500):
    """
    Approximate broadband aerosol optical depth.

    Bird and Hulstrom developed a correlation for broadband aerosol optical
    depth (AOD) using two wavelengths, 380 nm and 500 nm.

    Parameters
    ----------
    aod380 : numeric
        AOD measured at 380 nm. [unitless]
    aod500 : numeric
        AOD measured at 500 nm. [unitless]

    Returns
    -------
    aod_bb : numeric
        Broadband AOD.  [unitless]

    See also
    --------
    pvlib.atmosphere.kasten96_lt

    References
    ----------
    .. [1] Bird and Hulstrom, "Direct Insolation Models" (1980)
       `SERI/TR-335-344 <http://www.nrel.gov/docs/legosti/old/344.pdf>`_

    .. [2] R. E. Bird and R. L. Hulstrom, "Review, Evaluation, and Improvement
       of Direct Irradiance Models", Journal of Solar Energy Engineering
       103(3), pp. 182-192 (1981)
       :doi:`10.1115/1.3266239`
    """
    # approximate broadband AOD using (Bird-Hulstrom 1980)
    return 0.27583 * aod380 + 0.35 * aod500


def kasten96_lt(airmass_absolute, precipitable_water, aod_bb):
    """
    Calculate Linke turbidity  using Kasten pyrheliometric formula.

    Note that broadband aerosol optical depth (AOD) can be approximated by AOD
    measured at 700 nm according to Molineaux [4] . Bird and Hulstrom offer an
    alternate approximation using AOD measured at 380 nm and 500 nm.

    Based on original implementation by Armel Oumbe.

    .. warning::
        These calculations are only valid for airmass less than 5 and
        precipitable water less than 5 cm.

    Parameters
    ----------
    airmass_absolute : numeric
        Pressure-adjusted airmass. [unitless]
    precipitable_water : numeric
        Precipitable water. [cm]
    aod_bb : numeric
        broadband AOD. [unitless]

    Returns
    -------
    lt : numeric
        Linke turbidity. [unitless]

    See also
    --------
    pvlib.atmosphere.bird_hulstrom80_aod_bb
    pvlib.atmosphere.angstrom_aod_at_lambda

    References
    ----------
    .. [1] F. Linke, "Transmissions-Koeffizient und Trubungsfaktor", Beitrage
       zur Physik der Atmosphare, Vol 10, pp. 91-103 (1922)

    .. [2] F. Kasten, "A simple parameterization of the pyrheliometric formula
       for determining the Linke turbidity factor", Meteorologische Rundschau
       33, pp. 124-127 (1980)

    .. [3] Kasten, "The Linke turbidity factor based on improved values of the
       integral Rayleigh optical thickness", Solar Energy, Vol. 56, No. 3,
       pp. 239-244 (1996)
       :doi:`10.1016/0038-092X(95)00114-7`

    .. [4] B. Molineaux, P. Ineichen, N. O'Neill, "Equivalence of
       pyrheliometric and monochromatic aerosol optical depths at a single key
       wavelength", Applied Optics Vol. 37, issue 10, 7008-7018 (1998)
       :doi:`10.1364/AO.37.007008`

    .. [5] P. Ineichen, "Conversion function between the Linke turbidity and
       the atmospheric water vapor and aerosol content", Solar Energy 82,
       pp. 1095-1097 (2008)
       :doi:`10.1016/j.solener.2008.04.010`

    .. [6] P. Ineichen and R. Perez, "A new airmass independent formulation for
       the Linke Turbidity coefficient", Solar Energy, Vol. 73, no. 3,
       pp. 151-157 (2002)
       :doi:`10.1016/S0038-092X(02)00045-2`
    """
    # "From numerically integrated spectral simulations done with Modtran
    # (Berk, 1989), Molineaux (1998) obtained for the broadband optical depth
    # of a clean and dry atmospshere (fictitious atmosphere that comprises only
    # the effects of Rayleigh scattering and absorption by the atmosphere gases
    # other than the water vapor) the following expression"
    # - P. Ineichen (2008)
    delta_cda = -0.101 + 0.235 * airmass_absolute ** (-0.16)
    # "and the broadband water vapor optical depth where pwat is the integrated
    # precipitable water vapor content of the atmosphere expressed in cm and am
    # the optical air mass. The precision of these fits is better than 1% when
    # compared with Modtran simulations in the range 1 < am < 5 and
    # 0 < pwat < 5 cm at sea level" - P. Ineichen (2008)
    delta_w = 0.112 * airmass_absolute ** (-0.55) * precipitable_water ** 0.34
    # broadband AOD
    delta_a = aod_bb
    # "Then using the Kasten pyrheliometric formula (1980, 1996), the Linke
    # turbidity at am = 2 can be written. The extension of the Linke turbidity
    # coefficient to other values of air mass was published by Ineichen and
    # Perez (2002)" - P. Ineichen (2008)
    lt = -(9.4 + 0.9 * airmass_absolute) * np.log(
        np.exp(-airmass_absolute * (delta_cda + delta_w + delta_a))
    ) / airmass_absolute
    # filter out of extrapolated values
    return lt


def angstrom_aod_at_lambda(aod0, lambda0, alpha=1.14, lambda1=700.0):
    r"""
    Get AOD at specified wavelength using Angstrom turbidity model.

    Parameters
    ----------
    aod0 : numeric
        Aerosol optical depth (AOD) measured at wavelength ``lambda0``.
        [unitless]
    lambda0 : numeric
        Wavelength corresponding to ``aod0``. [nm]
    alpha : numeric, default 1.14
        Angstrom :math:`\alpha` exponent corresponding to ``aod0``. [unitless]
    lambda1 : numeric, default 700
        Desired wavelength. [nm]

    Returns
    -------
    aod1 : numeric
        AOD at desired wavelength ``lambda1``. [unitless]

    See also
    --------
    pvlib.atmosphere.angstrom_alpha

    References
    ----------
    .. [1] Anders Angstrom, "On the Atmospheric Transmission of Sun Radiation
       and On Dust in the Air", Geografiska Annaler Vol. 11, pp. 156-166 (1929)
       JSTOR
       :doi:`10.2307/519399`

    .. [2] Anders Angstrom, "Techniques of Determining the Turbidity of the
       Atmosphere", Tellus 13:2, pp. 214-223 (1961) Taylor & Francis
       :doi:`10.3402/tellusa.v13i2.9493` and Co-Action Publishing
       :doi:`10.1111/j.2153-3490.1961.tb00078.x`
    """
    return aod0 * ((lambda1 / lambda0) ** (-alpha))


def angstrom_alpha(aod1, lambda1, aod2, lambda2):
    r"""
    Calculate Angstrom alpha exponent.

    Parameters
    ----------
    aod1 : numeric
        Aerosol optical depth at wavelength ``lambda1``. [unitless]
    lambda1 : numeric
        Wavelength corresponding to ``aod1``. [nm]
    aod2 : numeric
        Aerosol optical depth  at wavelength ``lambda2``. [unitless]
    lambda2 : numeric
        Wavelength corresponding to ``aod2``. [nm]

    Returns
    -------
    alpha : numeric
        Angstrom :math:`\alpha` exponent for wavelength in
        ``(lambda1, lambda2)``. [unitless]

    See also
    --------
    pvlib.atmosphere.angstrom_aod_at_lambda
    """
    return - np.log(aod1 / aod2) / np.log(lambda1 / lambda2)


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


def windspeed_powerlaw(wind_speed_reference, height_reference,
                       height_desired, exponent=None,
                       surface_type=None):
    r"""
    Estimate wind speed for different heights.

    The model is based on the power law equation by Hellmann [1]_ [2]_.

    Parameters
    ----------
    wind_speed_reference : numeric
        Measured wind speed. [m/s]

    height_reference : float
        The height above ground at which the wind speed is measured. [m]

    height_desired : float
        The height above ground at which the wind speed will be estimated. [m]

    exponent : float, optional
        Exponent based on the surface type. [unitless]

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
        If both ``exponent`` and a ``surface_type`` is given. These parameters
        are mutually exclusive.

    KeyError
        If the specified ``surface_type`` is invalid.

    Notes
    -----
    Module temperature functions often require wind speeds at a height of 10 m
    and not the wind speed at the module height.

    For example, the following temperature functions require the input wind
    speed to be 10 m: :py:func:`~pvlib.temperature.sapm_cell`, and
    :py:func:`~pvlib.temperature.sapm_module` whereas the
    :py:func:`~pvlib.temperature.fuentes` model requires wind speed at 9.144 m.

    Additionally, the heat loss coefficients of some models have been developed
    for wind speed measurements at 10 m (e.g.,
    :py:func:`~pvlib.temperature.pvsyst_cell`,
    :py:func:`~pvlib.temperature.faiman`, and
    :py:func:`~pvlib.temperature.faiman_rad`).

    The equation for calculating the wind speed at a height of :math:`h` is
    given by the following power law equation [1]_ [2]_:

    .. math::
       :label: wind speed

        WS_{h} = WS_{ref} \cdot \left( \frac{h}{h_{ref}} \right)^a

    where :math:`h` [m] is the height at which we would like to calculate the
    wind speed, :math:`h_{ref}` [m] is the reference height at which the wind
    speed is known, and :math:`WS_{h}` [m/s] and :math:`WS_{ref}`
    [m/s] are the corresponding wind speeds at these heights. The exponent
    :math:`a` [unitless] depends on the surface type. Some values found in the
    literature [1]_ for :math:`a` are:

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

    In a report by Sandia [3]_, the equation was experimentally tested for a
    height of 30 ft (:math:`h_{ref} = 9.144` [m]) at their test site in
    Albuquerque for a period of six weeks where a coefficient of
    :math:`a = 0.219` was calculated.

    It should be noted that the equation returns a value of NaN if the
    reference heights or wind speed are negative.

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
       Accessed at:
       https://web.archive.org/web/20230418202422/https://www2.jpl.nasa.gov/adv_tech/photovol/2016CTR/SNL%20-%20Est%20Wind%20Speed%20vs%20Height_1985.pdf
    """  # noqa:E501
    if surface_type is not None and exponent is None:
        # use the Hellmann exponent from dictionary
        exponent = HELLMANN_SURFACE_EXPONENTS[surface_type]
    elif surface_type is None and exponent is not None:
        # use the provided exponent
        pass
    else:
        raise ValueError(
            "Either a 'surface_type' or an 'exponent' parameter must be given")

    wind_speed = wind_speed_reference * (
        (height_desired / height_reference) ** exponent)

    # if wind speed is negative or complex return NaN
    wind_speed = np.where(np.iscomplex(wind_speed) | (wind_speed < 0),
                          np.nan, wind_speed)

    if isinstance(wind_speed_reference, pd.Series):
        wind_speed = pd.Series(wind_speed, index=wind_speed_reference.index)

    return wind_speed
