"""
The ``atmosphere`` module contains methods to calculate relative and
absolute airmass and to determine pressure from altitude or vice versa.
"""

import numpy as np
import pandas as pd
import pvlib

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


first_solar_spectral_correction = deprecated(
    since='0.10.0',
    alternative='pvlib.spectrum.spectral_factor_firstsolar'
)(pvlib.spectrum.spectral_factor_firstsolar)


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


def caballero_spectral_correction(airmass_absolute, aod500, pw,
                                  module_type=None, coefficients=None,
                                  aod500_ref=0.084, pw_ref=1.42):
    r"""
    Spectral mismatch modifier based on absolute (pressure-adjusted)
    airmass (AM), aerosol optical depth (AOD) at 500 nm and
    precipitable water (PW).

    Estimates a spectral mismatch modifier :math:`M` representing
    the effect on module short circuit current of variation in the
    spectral irradiance. :math:`M` is estimated from absolute
    (pressure-adjusted) AM, :math:`ama`, AOD at 500 nm, :math:`aod500`
    and PW, :math:`pw` [1]_.

    The best fit polynomial for each atmospheric parameter (AM, AOD, PW)
    and PV technology under study has been obtained from synthetic spectra
    generated with SMARTS [2]_, considering the following boundary
    conditions:

       * :math:`1.0 <= ama <= 5.0`
       * :math:`0.05 <= aod500 <= 0.6`
       * :math:`0.25 \textrm{cm} <= pw <= 4 \textrm{cm}`
       * Spectral range is limited to that of CMP11 (280 nm to 2800 nm)
       * All other parameters fixed at G173 standard

    Elevation (deg), AOD and PW data were recorded in the city of Jaén,
    Spain for one year synchronously with both, broadband and
    spectroradiometric measurements of 30º tilted global irradiance
    south-facing logged in 5-min intervals. AM was estimated through
    elevation data.

    Finally, the spectral mismatch factor was calculated for each
    of the PV technologies and a multivariable regression adjustment
    as a function of AM, AOD and PW was performed according to [3] and [1]_.
    As such, the polynomial adjustment coefficients included in [1]
    were obtained.


    Parameters
    ----------
    airmass_absolute : array-like
        absolute (pressure-adjusted) airmass. [unitless]

    aod500 : array-like
        atmospheric aerosol optical depth at 500 nm. [unitless]

    pw : array-like
        atmospheric precipitable water. [cm]

    module_type : None or string, default None
        a string specifying a cell type. Values of 'cdte', 'monosi', 'cigs',
        'multisi','asi' and 'perovskite'. If provided,
        module_type selects default coefficients for the following modules:

            * 'cdte' - anonymous CdTe module.
            * 'monosi', - anonymous sc-si module.
            * 'multisi', - anonymous mc-si- module.
            * 'cigs' - anonymous copper indium gallium selenide module.
            * 'asi' - anonymous amorphous silicon module.
            * 'perovskite' - anonymous pervoskite module.

    coefficients : None or array-like, optional
        the coefficients employed have been obtained with experimental
        data in the city of Jaén, Spain. It is pending to verify if such
        coefficients vary in places with extreme climates where AOD and
        pw values are frequently high.

    aod500_ref : numeric, default 0.084
        atmospheric aerosol optical depth at 500nm value related to the
        AM1.5G ASTMG-173-03 reference spectrum. [unitless]

    pw_ref : numeric, default 1.42
        atmospheric precipitable water value related to the AM1.5G ASTMG-173-03
        reference spectrum. [cm]

    Returns
    -------
    modifier: array-like
        spectral mismatch factor (unitless) which is can be multiplied
        with broadband irradiance reaching a module's cells to estimate
        effective irradiance, i.e., the irradiance that is converted to
        electrical current.

    References
    ----------
    .. [1] Caballero, J.A., Fernández, E., Theristis, M.,
        Almonacid, F., and Nofuentes, G. "Spectral Corrections Based on
        Air Mass, Aerosol Optical Depth and Precipitable Water
        for PV Performance Modeling."
        IEEE Journal of Photovoltaics 2018, 8(2), 552-558.
        https://doi.org/10.1109/jphotov.2017.2787019
    .. [2] Gueymard, Christian. SMARTS2: a simple model of the
        atmospheric radiative transfer of sunshine: algorithms
        and performance assessment. Cocoa, FL:
        Florida Solar Energy Center, 1995.
    .. [3] Theristis, M., Fernández, E., Almonacid, F., and
       Pérez-Higueras, Pedro. "Spectral Corrections Based
       on Air Mass, Aerosol Optical Depth and Precipitable
       Water for CPV Performance Modeling."
       IEEE Journal of Photovoltaics 2016, 6(6), 1598-1604.
       https://doi.org/10.1109/jphotov.2016.2606702

        """

    # Experimental coefficients

    _coefficients = {}
    _coefficients['cdte'] = (
        1.0044, 0.0095, -0.0037, 0.0002, 0.0000, -0.0046,
        -0.0182, 0, 0.0095, 0.0068, 0, 1)
    _coefficients['monosi'] = (
        0.9706, 0.0377, -0.0123, 0.0025, -0.0002, 0.0159,
        -0.0165, 0, -0.0016, -0.0027, 1, 0)
    _coefficients['multisi'] = (
        0.9836, 0.0254, -0.0085, 0.0016, -0.0001, 0.0094,
        -0.0132, 0, -0.0002, -0.0011, 1, 0)
    _coefficients['cigs'] = (
        0.9801, 0.0283, -0.0092, 0.0019, -0.0001, 0.0117,
        -0.0126, 0, -0.0011, -0.0019, 1, 0)
    _coefficients['asi'] = (
        1.1060, -0.0848, 0.0302, -0.0076, 0.0006, -0.1283,
        0.0986, -0.0254, 0.0156, 0.0146, 1, 0)
    _coefficients['perovskite'] = (
        1.0637, -0.0491, 0.0180, -0.0047, 0.0004, -0.0773,
        0.0583, -0.0159, 0.01251, 0.0109, 1, 0)

    if module_type is None and coefficients is None:
        raise ValueError('Must provide either `module_type` or `coefficients`')
    if module_type is not None and coefficients is not None:
        raise ValueError('Only one of `module_type` and `coefficients` should '
                         'be provided')
    if module_type is not None:
        coeff = _coefficients[module_type]
    else:
        coeff = coefficients

    # Evaluate spectral correction factor
    ama = airmass_absolute
    modifier = (
        coeff[0] + (ama) * coeff[1] + (ama * ama) * coeff[2]
        + (ama * ama * ama) * coeff[3] + (ama * ama * ama * ama) * coeff[4]
        + (aod500 - aod500_ref) * coeff[5]
        + ((aod500 - aod500_ref) * (ama) * coeff[6]) * coeff[10]
        + ((aod500 - aod500_ref) * (np.log(ama)) * coeff[6]) * coeff[11]
        + (aod500 - aod500_ref) * (ama * ama) * coeff[7]
        + (pw - pw_ref) * coeff[8] + (pw - pw_ref) * (np.log(ama)) * coeff[9])

    return modifier
