"""
The ``atmosphere`` module contains methods to calculate relative and
absolute airmass and to determine pressure from altitude or vice versa.
"""

from __future__ import division

from warnings import warn

import numpy as np
import pandas as pd

from pvlib._deprecation import deprecated

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


def get_absolute_airmass(airmass_relative, pressure=101325.):
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

    pressure : numeric, default 101325
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


absoluteairmass = deprecated('0.6', alternative='get_absolute_airmass',
                             name='absoluteairmass', removal='0.7')(
                             get_absolute_airmass)


def get_relative_airmass(zenith, model='kastenyoung1989'):
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

    model : string, default 'kastenyoung1989'
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
    else:
        raise ValueError('%s is not a valid model for relativeairmass', model)

    if isinstance(zenith, pd.Series):
        am = pd.Series(am, index=zenith.index)

    return am


relativeairmass = deprecated('0.6', alternative='get_relative_airmass',
                             name='relativeairmass', removal='0.7')(
                             get_relative_airmass)


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

    module_type : None or string, default None
        a string specifying a cell type. Can be lower or upper case
        letters. Admits values of 'cdte', 'monosi', 'xsi', 'multisi',
        'polysi'. If provided, this input selects coefficients for the
        following default modules:

            * 'cdte' - First Solar Series 4-2 CdTe modules.
            * 'monosi', 'xsi' - First Solar TetraSun modules.
            * 'multisi', 'polysi' - multi-crystalline silicon modules.
            * 'cigs' - anonymous copper indium gallium selenide PV module
            * 'asi' - anonymous amorphous silicon PV module

        The module used to calculate the spectral correction
        coefficients corresponds to the Mult-crystalline silicon
        Manufacturer 2 Model C from [3]_. Spectral Response (SR) of CIGS
        and a-Si modules used to derive coefficients can be found in [4]_

    coefficients : None or array-like, default None
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
    .. [4] Schweiger, M. and Hermann, W, Influence of Spectral Effects
        on Energy Yield of Different PV Modules: Comparison of Pwat and
        MMF Approach, TUV Rheinland Energy GmbH report 21237296.003,
        January 2017
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
    _coefficients['cigs'] = (
        0.85252, -0.022314, -0.0047216, 0.13666, 0.013342, -0.0008945)
    _coefficients['asi'] = (
        1.12094, -0.047620, -0.0083627, -0.10443, 0.098382, -0.0033818)

    if module_type is not None and coefficients is None:
        coefficients = _coefficients[module_type.lower()]
    elif module_type is None and coefficients is not None:
        pass
    elif module_type is None and coefficients is None:
        raise TypeError('No valid input provided, both module_type and ' +
                        'coefficients are None')
    else:
        raise TypeError('Cannot resolve input, must supply only one of ' +
                        'module_type and coefficients')

    # Evaluate Spectral Shift
    coeff = coefficients
    ama = airmass_absolute
    modifier = (
        coeff[0] + coeff[1]*ama + coeff[2]*pw + coeff[3]*np.sqrt(ama) +
        coeff[4]*np.sqrt(pw) + coeff[5]*ama/np.sqrt(pw))

    return modifier


def bird_hulstrom80_aod_bb(aod380, aod500):
    """
    Approximate broadband aerosol optical depth.

    Bird and Hulstrom developed a correlation for broadband aerosol optical
    depth (AOD) using two wavelengths, 380 nm and 500 nm.

    Parameters
    ----------
    aod380 : numeric
        AOD measured at 380 nm
    aod500 : numeric
        AOD measured at 500 nm

    Returns
    -------
    aod_bb : numeric
        broadband AOD

    See also
    --------
    kasten96_lt

    References
    ----------
    [1] Bird and Hulstrom, "Direct Insolation Models" (1980)
    `SERI/TR-335-344 <http://www.nrel.gov/docs/legosti/old/344.pdf>`_

    [2] R. E. Bird and R. L. Hulstrom, "Review, Evaluation, and Improvement of
    Direct Irradiance Models", Journal of Solar Energy Engineering 103(3),
    pp. 182-192 (1981)
    :doi:`10.1115/1.3266239`
    """
    # approximate broadband AOD using (Bird-Hulstrom 1980)
    return 0.27583 * aod380 + 0.35 * aod500


def kasten96_lt(airmass_absolute, precipitable_water, aod_bb):
    """
    Calculate Linke turbidity factor using Kasten pyrheliometric formula.

    Note that broadband aerosol optical depth (AOD) can be approximated by AOD
    measured at 700 nm according to Molineaux [4] . Bird and Hulstrom offer an
    alternate approximation using AOD measured at 380 nm and 500 nm.

    Based on original implementation by Armel Oumbe.

    .. warning::
        These calculations are only valid for air mass less than 5 atm and
        precipitable water less than 5 cm.

    Parameters
    ----------
    airmass_absolute : numeric
        airmass, pressure corrected in atmospheres
    precipitable_water : numeric
        precipitable water or total column water vapor in centimeters
    aod_bb : numeric
        broadband AOD

    Returns
    -------
    lt : numeric
        Linke turbidity

    See also
    --------
    bird_hulstrom80_aod_bb
    angstrom_aod_at_lambda

    References
    ----------
    [1] F. Linke, "Transmissions-Koeffizient und Trubungsfaktor", Beitrage
    zur Physik der Atmosphare, Vol 10, pp. 91-103 (1922)

    [2] F. Kasten, "A simple parameterization of the pyrheliometric formula for
    determining the Linke turbidity factor", Meteorologische Rundschau 33,
    pp. 124-127 (1980)

    [3] Kasten, "The Linke turbidity factor based on improved values of the
    integral Rayleigh optical thickness", Solar Energy, Vol. 56, No. 3,
    pp. 239-244 (1996)
    :doi:`10.1016/0038-092X(95)00114-7`

    [4] B. Molineaux, P. Ineichen, N. O'Neill, "Equivalence of pyrheliometric
    and monochromatic aerosol optical depths at a single key wavelength",
    Applied Optics Vol. 37, issue 10, 7008-7018 (1998)
    :doi:`10.1364/AO.37.007008`

    [5] P. Ineichen, "Conversion function between the Linke turbidity and the
    atmospheric water vapor and aerosol content", Solar Energy 82,
    pp. 1095-1097 (2008)
    :doi:`10.1016/j.solener.2008.04.010`

    [6] P. Ineichen and R. Perez, "A new airmass independent formulation for
    the Linke Turbidity coefficient", Solar Energy, Vol. 73, no. 3, pp. 151-157
    (2002)
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
        aerosol optical depth (AOD) measured at known wavelength
    lambda0 : numeric
        wavelength in nanometers corresponding to ``aod0``
    alpha : numeric, default 1.14
        Angstrom :math:`\alpha` exponent corresponding to ``aod0``
    lambda1 : numeric, default 700
        desired wavelength in nanometers

    Returns
    -------
    aod1 : numeric
        AOD at desired wavelength, ``lambda1``

    See also
    --------
    angstrom_alpha

    References
    ----------
    [1] Anders Angstrom, "On the Atmospheric Transmission of Sun Radiation and
    On Dust in the Air", Geografiska Annaler Vol. 11, pp. 156-166 (1929) JSTOR
    :doi:`10.2307/519399`

    [2] Anders Angstrom, "Techniques of Determining the Turbidity of the
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
        first aerosol optical depth
    lambda1 : numeric
        wavelength in nanometers corresponding to ``aod1``
    aod2 : numeric
        second aerosol optical depth
    lambda2 : numeric
        wavelength in nanometers corresponding to ``aod2``

    Returns
    -------
    alpha : numeric
        Angstrom :math:`\alpha` exponent for AOD in ``(lambda1, lambda2)``

    See also
    --------
    angstrom_aod_at_lambda
    """
    return - np.log(aod1 / aod2) / np.log(lambda1 / lambda2)
