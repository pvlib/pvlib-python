"""
The ``atmosphere`` module contains methods to calculate relative and
absolute airmass and to determine pressure from altitude or vice versa.
"""

from __future__ import division

import numpy as np
import pandas as pd
from warnings import warn

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
        Atmospheric pressure (Pascals)

    Returns
    -------
    altitude : numeric
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
    [1] "A Quick Derivation relating altitude to air pressure" from
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
        Altitude in meters above sea level

    Returns
    -------
    pressure : numeric
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
    [1] "A Quick Derivation relating altitude to air pressure" from
    Portland State Aerospace Society, Version 1.03, 12/22/2004.
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
    airmass_relative : numeric
        The airmass at sea-level.

    pressure : numeric
        The site pressure in Pascal.

    Returns
    -------
    airmass_absolute : numeric
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
    zenith : numeric
        Zenith angle of the sun in degrees. Note that some models use
        the apparent (refraction corrected) zenith angle, and some
        models use the true (not refraction-corrected) zenith angle. See
        model descriptions to determine which type of zenith angle is
        required. Apparent zenith angles must be calculated at sea level.

    model : string
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
    airmass_relative : numeric
        Relative airmass at sea level. Will return NaN values for any
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

    # need to filter first because python 2.7 does not support raising a
    # negative number to a negative power.
    z = np.where(zenith > 90, np.nan, zenith)
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

           w = 0.1 H_v \rho_v

    using Eq. 2 in [2]_

    .. math::

           \rho_v = 216.7 R_H e_s /T

    :math:`H_v` is the apparant water vapor scale height (km). The
    expression for :math:`H_v` is Eq. 4 in [2]_:

    .. math::

           H_v = 0.4976 + 1.5265*T/273.15 + \exp(13.6897*T/273.15 - 14.9188*(T/273.15)^3)

    :math:`\rho_v` is the surface water vapor density (g/m^3). In the
    expression :math:`\rho_v`, :math:`e_s` is the saturation water vapor
    pressure (millibar). The
    expression for :math:`e_s` is Eq. 1 in [3]_

    .. math::

          e_s = \exp(22.330 - 49.140*(100/T) - 10.922*(100/T)^2 - 0.39015*T/100)

    Parameters
    ----------
    temp_air : numeric
        ambient air temperature at the surface (C)
    relative_humidity : numeric
        relative humidity at the surface (%)

    Returns
    -------
    pw : numeric
        precipitable water (cm)

    References
    ----------
    .. [1] W. M. Keogh and A. W. Blakers, Accurate Measurement, Using Natural
       Sunlight, of Silicon Solar Cells, Prog. in Photovoltaics: Res.
       and Appl. 2004, vol 12, pp. 1-19 (DOI: 10.1002/pip.517)

    .. [2] C. Gueymard, Analysis of Monthly Average Atmospheric Precipitable
       Water and Turbidity in Canada and Northern United States,
       Solar Energy vol 53(1), pp. 57-71, 1994.

    .. [3] C. Gueymard, Assessment of the Accuracy and Computing Speed of
       simplified saturation vapor equations using a new reference
       dataset, J. of Applied Meteorology 1993, vol. 32(7), pp.
       1294-1300.
    """

    T = temp_air + 273.15  # Convert to Kelvin
    RH = relative_humidity

    theta = T / 273.15

    # Eq. 1 from Keogh and Blakers
    pw = (
        0.1 *
        (0.4976 + 1.5265*theta + np.exp(13.6897*theta - 14.9188*(theta)**3)) *
        (216.7*RH/(100*T)*np.exp(22.330 - 49.140*(100/T) -
         10.922*(100/T)**2 - 0.39015*T/100)))

    pw = np.maximum(pw, 0.1)

    return pw


def first_solar_spectral_correction(pw, airmass_absolute, module_type=None,
                                    coefficients=None):
    r"""
    Spectral mismatch modifier based on precipitable water and absolute
    (pressure corrected) airmass.

    Estimates a spectral mismatch modifier M representing the effect on
    module short circuit current of variation in the spectral
    irradiance. M is estimated from absolute (pressure currected) air
    mass, AMa, and precipitable water, Pwat, using the following
    function:

    .. math::

        M = c_1 + c_2*AMa  + c_3*Pwat  + c_4*AMa^.5
            + c_5*Pwat^.5 + c_6*AMa/Pwat^.5

    Default coefficients are determined for several cell types with
    known quantum efficiency curves, by using the Simple Model of the
    Atmospheric Radiative Transfer of Sunshine (SMARTS) [1]_. Using
    SMARTS, spectrums are simulated with all combinations of AMa and
    Pwat where:

       * 0.5 cm <= Pwat <= 5 cm
       * 1.0 <= AMa <= 5.0
       * Spectral range is limited to that of CMP11 (280 nm to 2800 nm)
       * spectrum simulated on a plane normal to the sun
       * All other parameters fixed at G173 standard

    From these simulated spectra, M is calculated using the known
    quantum efficiency curves. Multiple linear regression is then
    applied to fit Eq. 1 to determine the coefficients for each module.

    Based on the PVLIB Matlab function ``pvl_FSspeccorr`` by Mitchell
    Lee and Alex Panchula, at First Solar, 2016 [2]_.

    Parameters
    ----------
    pw : array-like
        atmospheric precipitable water (cm).

    airmass_absolute : array-like
        absolute (pressure corrected) airmass.

    module_type : None or string
        a string specifying a cell type. Can be lower or upper case
        letters. Admits values of 'cdte', 'monosi', 'xsi', 'multisi',
        'polysi'. If provided, this input selects coefficients for the
        following default modules:

            * 'cdte' - First Solar Series 4-2 CdTe modules.
            * 'monosi', 'xsi' - First Solar TetraSun modules.
            * 'multisi', 'polysi' - multi-crystalline silicon modules.

        The module used to calculate the spectral correction
        coefficients corresponds to the Mult-crystalline silicon
        Manufacturer 2 Model C from [3]_.

    coefficients : array-like
        allows for entry of user defined spectral correction
        coefficients. Coefficients must be of length 6. Derivation of
        coefficients requires use of SMARTS and PV module quantum
        efficiency curve. Useful for modeling PV module types which are
        not included as defaults, or to fine tune the spectral
        correction to a particular mono-Si, multi-Si, or CdTe PV module.
        Note that the parameters for modules with very similar QE should
        be similar, in most cases limiting the need for module specific
        coefficients.

    Returns
    -------
    modifier: array-like
        spectral mismatch factor (unitless) which is can be multiplied
        with broadband irradiance reaching a module's cells to estimate
        effective irradiance, i.e., the irradiance that is converted to
        electrical current.

    References
    ----------
    .. [1] Gueymard, Christian. SMARTS2: a simple model of the atmospheric
       radiative transfer of sunshine: algorithms and performance
       assessment. Cocoa, FL: Florida Solar Energy Center, 1995.
    .. [2] Lee, Mitchell, and Panchula, Alex. "Spectral Correction for
       Photovoltaic Module Performance Based on Air Mass and Precipitable
       Water." IEEE Photovoltaic Specialists Conference, Portland, 2016
    .. [3] Marion, William F., et al. User's Manual for Data for Validating
       Models for PV Module Performance. National Renewable Energy
       Laboratory, 2014. http://www.nrel.gov/docs/fy14osti/61610.pdf
    """

    # --- Screen Input Data ---

    # *** Pwat ***
    # Replace Pwat Values below 0.1 cm with 0.1 cm to prevent model from
    # diverging"

    if np.min(pw) < 0.1:
        pw = np.maximum(pw, 0.1)
        warn('Exceptionally low Pwat values replaced with 0.1 cm to prevent' +
             ' model divergence')

    # Warn user about Pwat data that is exceptionally high
    if np.max(pw) > 8:
        warn('Exceptionally high Pwat values. Check input data:' +
             ' model may diverge in this range')

    # *** AMa ***
    # Replace Extremely High AM with AM 10 to prevent model divergence
    # AM > 10 will only occur very close to sunset
    if np.max(airmass_absolute) > 10:
        airmass_absolute = np.minimum(airmass_absolute, 10)

    # Warn user about AMa data that is exceptionally low
    if np.min(airmass_absolute) < 0.58:
        warn('Exceptionally low air mass: ' +
             'model not intended for extra-terrestrial use')
        # pvl_absoluteairmass(1,pvl_alt2pres(4340)) = 0.58 Elevation of
        # Mina Pirquita, Argentian = 4340 m. Highest elevation city with
        # population over 50,000.

    _coefficients = {}
    _coefficients['cdte'] = (
       0.86273, -0.038948, -0.012506, 0.098871, 0.084658, -0.0042948)
    _coefficients['monosi'] = (
        0.85914, -0.020880, -0.0058853, 0.12029, 0.026814, -0.0017810)
    _coefficients['xsi'] = _coefficients['monosi']
    _coefficients['polysi'] = (
        0.84090, -0.027539, -0.0079224, 0.13570, 0.038024, -0.0021218)
    _coefficients['multisi'] = _coefficients['polysi']

    if module_type is not None and coefficients is None:
        coefficients = _coefficients[module_type.lower()]
    elif module_type is None and coefficients is not None:
        pass
    else:
        raise TypeError('ambiguous input, must supply only 1 of ' +
                        'module_type and coefficients')

    # Evaluate Spectral Shift
    coeff = coefficients
    ama = airmass_absolute
    modifier = (
        coeff[0] + coeff[1]*ama + coeff[2]*pw + coeff[3]*np.sqrt(ama) +
        coeff[4]*np.sqrt(pw) + coeff[5]*ama/np.sqrt(pw))

    return modifier
