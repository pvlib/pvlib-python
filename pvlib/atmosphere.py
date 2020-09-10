"""
The ``atmosphere`` module contains methods to calculate relative and
absolute airmass and to determine pressure from altitude or vice versa.
"""

from warnings import warn

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
          requries true sun zenith
        * 'pickering2002' - See reference [6] -
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

    .. [7] Matthew J. Reno, Clifford W. Hansen and Joshua S. Stein, "Global
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


def first_solar_spectral_correction(pw, airmass_absolute,
                                    module_type=None, coefficients=None,
                                    min_pw=0.1, max_pw=8):
    r"""
    Spectral mismatch modifier based on precipitable water and absolute
    (pressure-adjusted) airmass.

    Estimates a spectral mismatch modifier :math:`M` representing the effect on
    module short circuit current of variation in the spectral
    irradiance. :math:`M`  is estimated from absolute (pressure currected) air
    mass, :math:`AM_a`, and precipitable water, :math:`Pw`, using the following
    function:

    .. math::

        M = c_1 + c_2 AM_a  + c_3 Pw  + c_4 AM_a^{0.5}
            + c_5 Pw^{0.5} + c_6 \frac{AM_a} {Pw^{0.5}}

    Default coefficients are determined for several cell types with
    known quantum efficiency curves, by using the Simple Model of the
    Atmospheric Radiative Transfer of Sunshine (SMARTS) [1]_. Using
    SMARTS, spectrums are simulated with all combinations of AMa and
    Pw where:

       * :math:`0.5 \textrm{cm} <= Pw <= 5 \textrm{cm}`
       * :math:`1.0 <= AM_a <= 5.0`
       * Spectral range is limited to that of CMP11 (280 nm to 2800 nm)
       * spectrum simulated on a plane normal to the sun
       * All other parameters fixed at G173 standard

    From these simulated spectra, M is calculated using the known
    quantum efficiency curves. Multiple linear regression is then
    applied to fit Eq. 1 to determine the coefficients for each module.

    Based on the PVLIB Matlab function ``pvl_FSspeccorr`` by Mitchell
    Lee and Alex Panchula of First Solar, 2016 [2]_.

    Parameters
    ----------
    pw : array-like
        atmospheric precipitable water. [cm]

    airmass_absolute : array-like
        absolute (pressure-adjusted) airmass. [unitless]

    min_pw : float, default 0.1
        minimum atmospheric precipitable water. Any pw value lower than min_pw
        is set to min_pw to avoid model divergence. [cm]

    max_pw : float, default 8
        maximum atmospheric precipitable water. Any pw value higher than max_pw
        is set to NaN to avoid model divergence. [cm]

    module_type : None or string, default None
        a string specifying a cell type. Values of 'cdte', 'monosi', 'xsi',
        'multisi', and 'polysi' (can be lower or upper case). If provided,
        module_type selects default coefficients for the following modules:

            * 'cdte' - First Solar Series 4-2 CdTe module.
            * 'monosi', 'xsi' - First Solar TetraSun module.
            * 'multisi', 'polysi' - anonymous multi-crystalline silicon module.
            * 'cigs' - anonymous copper indium gallium selenide module.
            * 'asi' - anonymous amorphous silicon module.

        The module used to calculate the spectral correction
        coefficients corresponds to the Multi-crystalline silicon
        Manufacturer 2 Model C from [3]_. The spectral response (SR) of CIGS
        and a-Si modules used to derive coefficients can be found in [4]_

    coefficients : None or array-like, default None
        Allows for entry of user-defined spectral correction
        coefficients. Coefficients must be of length 6. Derivation of
        coefficients requires use of SMARTS and PV module quantum
        efficiency curve. Useful for modeling PV module types which are
        not included as defaults, or to fine tune the spectral
        correction to a particular PV module. Note that the parameters for
        modules with very similar quantum efficiency should be similar,
        in most cases limiting the need for module specific coefficients.

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

    # *** Pw ***
    # Replace Pw Values below 0.1 cm with 0.1 cm to prevent model from
    # diverging"
    pw = np.atleast_1d(pw)
    pw = pw.astype('float64')
    if np.min(pw) < min_pw:
        pw = np.maximum(pw, min_pw)
        warn(f'Exceptionally low pw values replaced with {min_pw} cm to '
             'prevent model divergence')

    # Warn user about Pw data that is exceptionally high
    if np.max(pw) > max_pw:
        pw[pw > max_pw] = np.nan
        warn('Exceptionally high pw values replaced by np.nan: '
             'check input data.')

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
