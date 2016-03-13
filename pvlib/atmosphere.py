"""
The ``atmosphere`` module contains methods to calculate relative and
absolute airmass and to determine pressure from altitude or vice versa.
"""

from __future__ import division

import numpy as np

APPARENT_ZENITH_MODELS = ('simple', 'kasten1966', 'kastenyoung1989',
                          'gueymard1993', 'pickering2002')
TRUE_ZENITH_MODELS = ('youngirvine1967', 'young1994')
AIRMASS_MODELS = APPARENT_ZENITH_MODELS + TRUE_ZENITH_MODELS


def pres2alt(pressure):
    '''
    Determine altitude from site pressure.

    Parameters
    ----------
    pressure : scalar or Series
        Atmospheric pressure (Pascals)

    Returns
    -------
    altitude : scalar or Series
        Altitude in meters above sea level

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
    Gas constant for air           287.053 J/(kgK)
    Relative Humidity              0%
    ============================   ================

    References
    -----------

    "A Quick Derivation relating altitude to air pressure" from Portland
    State Aerospace Society, Version 1.03, 12/22/2004.
    '''

    alt = 44331.5 - 4946.62 * pressure ** (0.190263)
    return alt


def alt2pres(altitude):
    '''
    Determine site pressure from altitude.

    Parameters
    ----------
    Altitude : scalar or Series
        Altitude in meters above sea level

    Returns
    -------
    Pressure : scalar or Series
        Atmospheric pressure (Pascals)

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
    Gas constant for air           287.053 J/(kgK)
    Relative Humidity              0%
    ============================   ================

    References
    -----------

    "A Quick Derivation relating altitude to air pressure" from Portland
    State Aerospace Society, Version 1.03, 12/22/2004.
    '''

    press = 100 * ((44331.514 - altitude) / 11880.516) ** (1 / 0.1902632)

    return press


def absoluteairmass(airmass_relative, pressure=101325.):
    '''
    Determine absolute (pressure corrected) airmass from relative
    airmass and pressure

    Gives the airmass for locations not at sea-level (i.e. not at
    standard pressure). The input argument "AMrelative" is the relative
    airmass. The input argument "pressure" is the pressure (in Pascals)
    at the location of interest and must be greater than 0. The
    calculation for absolute airmass is

    .. math::
        absolute airmass = (relative airmass)*pressure/101325

    Parameters
    ----------

    airmass_relative : scalar or Series
        The airmass at sea-level.

    pressure : scalar or Series
        The site pressure in Pascal.

    Returns
    -------
    scalar or Series
        Absolute (pressure corrected) airmass

    References
    ----------
    [1] C. Gueymard, "Critical analysis and performance assessment of
    clear sky solar irradiance models using theoretical and measured
    data," Solar Energy, vol. 51, pp. 121-138, 1993.

    '''

    airmass_absolute = airmass_relative * pressure / 101325.

    return airmass_absolute


def relativeairmass(zenith, model='kastenyoung1989'):
    '''
    Gives the relative (not pressure-corrected) airmass.

    Gives the airmass at sea-level when given a sun zenith angle (in
    degrees). The ``model`` variable allows selection of different
    airmass models (described below). If ``model`` is not included or is
    not valid, the default model is 'kastenyoung1989'.

    Parameters
    ----------

    zenith : float or Series
        Zenith angle of the sun in degrees. Note that some models use
        the apparent (refraction corrected) zenith angle, and some
        models use the true (not refraction-corrected) zenith angle. See
        model descriptions to determine which type of zenith angle is
        required. Apparent zenith angles must be calculated at sea level.

    model : String
        Available models include the following:

        * 'simple' - secant(apparent zenith angle) -
          Note that this gives -inf at zenith=90
        * 'kasten1966' - See reference [1] -
          requires apparent sun zenith
        * 'youngirvine1967' - See reference [2] -
          requires true sun zenith
        * 'kastenyoung1989' - See reference [3] -
          requires apparent sun zenith
        * 'gueymard1993' - See reference [4] -
          requires apparent sun zenith
        * 'young1994' - See reference [5] -
          requries true sun zenith
        * 'pickering2002' - See reference [6] -
          requires apparent sun zenith

    Returns
    -------
    airmass_relative : float or Series
        Relative airmass at sea level.  Will return NaN values for any
        zenith angle greater than 90 degrees.

    References
    ----------

    [1] Fritz Kasten. "A New Table and Approximation Formula for the
    Relative Optical Air Mass". Technical Report 136, Hanover, N.H.:
    U.S. Army Material Command, CRREL.

    [2] A. T. Young and W. M. Irvine, "Multicolor Photoelectric
    Photometry of the Brighter Planets," The Astronomical Journal, vol.
    72, pp. 945-950, 1967.

    [3] Fritz Kasten and Andrew Young. "Revised optical air mass tables
    and approximation formula". Applied Optics 28:4735-4738

    [4] C. Gueymard, "Critical analysis and performance assessment of
    clear sky solar irradiance models using theoretical and measured
    data," Solar Energy, vol. 51, pp. 121-138, 1993.

    [5] A. T. Young, "AIR-MASS AND REFRACTION," Applied Optics, vol. 33,
    pp. 1108-1110, Feb 1994.

    [6] Keith A. Pickering. "The Ancient Star Catalog". DIO 12:1, 20,

    [7] Matthew J. Reno, Clifford W. Hansen and Joshua S. Stein, "Global
    Horizontal Irradiance Clear Sky Models: Implementation and Analysis"
    Sandia Report, (2012).
    '''

    z = zenith
    zenith_rad = np.radians(z)

    model = model.lower()

    if 'kastenyoung1989' == model:
        am = (1.0 / (np.cos(zenith_rad) +
              0.50572*(((6.07995 + (90 - z)) ** - 1.6364))))
    elif 'kasten1966' == model:
        am = 1.0 / (np.cos(zenith_rad) + 0.15*((93.885 - z) ** - 1.253))
    elif 'simple' == model:
        am = 1.0 / np.cos(zenith_rad)
    elif 'pickering2002' == model:
        am = (1.0 / (np.sin(np.radians(90 - z +
              244.0 / (165 + 47.0 * (90 - z) ** 1.1)))))
    elif 'youngirvine1967' == model:
        am = ((1.0 / np.cos(zenith_rad)) *
              (1 - 0.0012*((1.0 / np.cos(zenith_rad)) ** 2) - 1))
    elif 'young1994' == model:
        am = ((1.002432*((np.cos(zenith_rad)) ** 2) +
              0.148386*(np.cos(zenith_rad)) + 0.0096467) /
              (np.cos(zenith_rad) ** 3 +
              0.149864*(np.cos(zenith_rad) ** 2) +
              0.0102963*(np.cos(zenith_rad)) + 0.000303978))
    elif 'gueymard1993' == model:
        am = (1.0 / (np.cos(zenith_rad) +
              0.00176759*(z)*((94.37515 - z) ** - 1.21563)))
    else:
        raise ValueError('%s is not a valid model for relativeairmass', model)

    try:
        am[z > 90] = np.nan
    except TypeError:
        am = np.nan if z > 90 else am

    return am
