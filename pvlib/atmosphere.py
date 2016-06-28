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
    temp_air : array-like
        ambient air temperature at the surface (C)
    relative_humidity : array-like
        relative humidity at the surface (%)

    Returns
    -------
    pw : array-like
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
            + c_5*Pwat^.5 + c_6*AMa/Pwat

    Default coefficients are determined for several cell types with
    known quantum efficiency curves, by using the Simple Model of the
    Atmospheric Radiative Transfer of Sunshine (SMARTS) [1]_. Using
    SMARTS, spectrums are simulated with all combinations of AMa and
    Pwat where:

       * 0.5 cm <= Pwat <= 5 cm
       * 0.8 <= AMa <= 4.75 (Pressure of 800 mbar and 1.01 <= AM <= 6)
       * Spectral range is limited to that of CMP11 (280 nm to 2800 nm)
       * spectrum simulated on a plane normal to the sun
       * All other parameters fixed at G173 standard

    From these simulated spectra, M is calculated using the known
    quantum efficiency curves. Multiple linear regression is then
    applied to fit Eq. 1 to determine the coefficients for each module.

    Based on the PVLIB Matlab function ``pvl_FSspeccorr`` by Mitchell
    Lee and Alex Panchula, at First Solar, 2015.

    Parameters
    ----------
    pw : array-like
        atmospheric precipitable water (cm).

    airmass_absolute :
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
        Manufacturer 2 Model C from [2]_.

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

    .. [2] Marion, William F., et al. User's Manual for Data for Validating
       Models for PV Module Performance. National Renewable Energy
       Laboratory, 2014. http://www.nrel.gov/docs/fy14osti/61610.pdf
    """

    _coefficients = {}
    _coefficients['cdte'] = (
        0.87102, -0.040543, -0.00929202, 0.10052, 0.073062, -0.0034187)
    _coefficients['monosi'] = (
        0.86588, -0.021637, -0.0030218, 0.12081, 0.017514, -0.0012610)
    _coefficients['xsi'] = _coefficients['monosi']
    _coefficients['polysi'] = (
        0.84674, -0.028568, -0.0051832, 0.13669, 0.029234, -0.0014207)
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
    AMa = airmass_absolute
    modifier = (
        coeff[0] + coeff[1]*AMa  + coeff[2]*pw  + coeff[3]*np.sqrt(AMa) +
        + coeff[4]*np.sqrt(pw) + coeff[5]*AMa/pw)

    return modifier


def transmittance(cloud_prct):
    '''
    Calculates transmittance.

    Based on observations by Liu and Jordan, 1960 as well as
    Gates 1980.

    Parameters
    ----------
    cloud_prct: float or int
        Percentage of clouds covering the sky.

    Returns
    -------
    value: float
        Shortwave radiation transmittance.

    References
    ----------
    [1] Campbell, G. S., J. M. Norman (1998) An Introduction to
    Environmental Biophysics. 2nd Ed. New York: Springer.

    [2] Gates, D. M. (1980) Biophysical Ecology. New York: Springer Verlag.

    [3] Liu, B. Y., R. C. Jordan, (1960). "The interrelationship and
    characteristic distribution of direct, diffuse, and total solar
    radiation".  Solar Energy 4:1-19
    '''

    return ((100.0 - cloud_prct) / 100.0) * 0.75
