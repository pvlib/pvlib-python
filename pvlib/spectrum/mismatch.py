"""
The ``mismatch`` module in the ``spectrum`` package provides functions for
spectral mismatch calculations. Spectral mismatch models quantify the effect on
a device's photocurrent (or its short-circuit current) of changes in the solar
spectrum due to the atmosphere.
"""
import pvlib
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from warnings import warn


def calc_spectral_mismatch_field(sr, e_sun, e_ref=None):
    """
    Calculate spectral mismatch between a test device and broadband reference
    device under specified solar spectral irradiance conditions.

    Parameters
    ----------
    sr: pandas.Series
        The relative spectral response of one (photovoltaic) test device.
        The index of the Series must contain wavelength values in nm. [-]

    e_sun: pandas.DataFrame or pandas.Series
        One or more measured solar irradiance spectra in a pandas.DataFrame
        having wavelength in nm as column index. A single spectrum may be
        be given as a pandas.Series having wavelength in nm as index.
        [(W/m^2)/nm]

    e_ref: pandas.Series, optional
        The reference spectrum to use for the mismatch calculation.
        The index of the Series must contain wavelength values in nm.
        The default is the ASTM G173-03 global tilted spectrum. [(W/m^2)/nm]

    Returns
    -------
    smm: pandas.Series or float if a single measured spectrum is provided. [-]

    Notes
    -----
    Measured solar spectral irradiance usually covers a wavelength range
    that is smaller than the range considered as broadband irradiance.
    The infrared limit for the former typically lies around 1100 or 1600 nm,
    whereas the latter extends to around 2800 or 4000 nm.  To avoid imbalance
    between the magnitudes of the integrated spectra (the broadband values)
    this function truncates the reference spectrum to the same range as the
    measured (or simulated) field spectra. The assumption implicit in this
    truncation is that the energy in the unmeasured wavelength range
    is the same fraction of the broadband energy for both the measured
    spectra and the reference spectrum.

    If the default reference spectrum is used it is linearly interpolated
    to the wavelengths of the measured spectrum, but if a reference spectrum
    is provided via the parameter ``e_ref`` it is used without change. This
    makes it possible to avoid interpolation, or to use a different method of
    interpolation, or to avoid truncation.

    The spectral response is linearly interpolated to the wavelengths of each
    spectrum with which is it multiplied internally (``e_sun`` and ``e_ref``).
    If the wavelengths of the spectral response already match one or both
    of these spectra interpolation has no effect; therefore, another type of
    interpolation could be used to process ``sr`` before calling this function.

    The standards describing mismatch calculations focus on indoor laboratory
    applications, but are applicable to outdoor performance as well.
    The 2016 version of ASTM E973 [1]_ is somewhat more difficult to
    read than the 2010 version [2]_ because it includes adjustments for
    the temperature dependency of spectral response, which led to a
    formulation using quantum efficiency (QE).
    IEC 60904-7 is clearer and also discusses the use of a broadband
    reference device. [3]_

    References
    ----------
    .. [1] ASTM "E973-16 Standard Test Method for Determination of the
       Spectral Mismatch Parameter Between a Photovoltaic Device and a
       Photovoltaic Reference Cell" :doi:`10.1520/E0973-16R20`
    .. [2] ASTM "E973-10 Standard Test Method for Determination of the
       Spectral Mismatch Parameter Between a Photovoltaic Device and a
       Photovoltaic Reference Cell" :doi:`10.1520/E0973-10`
    .. [3] IEC 60904-7 "Computation of the spectral mismatch correction
       for measurements of photovoltaic devices"
    """
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Aug. 2022

    # get the reference spectrum at wavelengths matching the measured spectra
    if e_ref is None:
        e_ref = pvlib.spectrum.get_reference_spectra(
            wavelengths=e_sun.T.index)["global"]

    # interpolate the sr at the wavelengths of the spectra
    # reference spectrum wavelengths may differ if e_ref is from caller
    sr_sun = np.interp(e_sun.T.index, sr.index, sr, left=0.0, right=0.0)
    sr_ref = np.interp(e_ref.T.index, sr.index, sr, left=0.0, right=0.0)

    # a helper function to make usable fraction calculations more readable
    def integrate(e):
        return trapezoid(e, x=e.T.index, axis=-1)

    # calculate usable fractions
    uf_sun = integrate(e_sun * sr_sun) / integrate(e_sun)
    uf_ref = integrate(e_ref * sr_ref) / integrate(e_ref)

    # mismatch is the ratio or quotient of the usable fractions
    smm = uf_sun / uf_ref

    if isinstance(e_sun, pd.DataFrame):
        smm = pd.Series(smm, index=e_sun.index)

    return smm


def spectral_factor_firstsolar(precipitable_water, airmass_absolute,
                               module_type=None, coefficients=None,
                               min_precipitable_water=0.1,
                               max_precipitable_water=8,
                               min_airmass_absolute=0.58,
                               max_airmass_absolute=10):
    r"""
    Spectral mismatch modifier based on precipitable water and absolute
    (pressure-adjusted) air mass.

    Estimates the spectral mismatch modifier, :math:`M`, representing the
    effect of variation in the spectral irradiance on the module short circuit
    current :math:`M`  is estimated from absolute (pressure-corrected) air
    mass, :math:`AM_a`, and precipitable water, :math:`Pw`.

    Default coefficients are determined for several cell types with
    known quantum efficiency curves, by using the Simple Model of the
    Atmospheric Radiative Transfer of Sunshine (SMARTS) [1]_. Using
    SMARTS, spectrums are simulated with all combinations of AMa and
    Pw where:

    * :math:`0.5 \textrm{cm} <= Pw <= 5 \textrm{cm}`
    * :math:`1.0 <= AM_a <= 5.0`
    * Spectral range is limited to that of CMP11 (280 nm to 2800 nm)
    * Spectrum simulated on an equatorial facing surface with 37° tilt
    * All other parameters fixed at G173 standard

    From these simulated spectra, :math:`M` is calculated using the known
    quantum efficiency curves. Multiple linear regression is then
    applied to fit Eq. 1 to determine the coefficients for each module. More
    details on the model can be found in [2]_.

    Parameters
    ----------
    precipitable_water : numeric
        atmospheric precipitable water. [cm]

    airmass_absolute : numeric
        absolute (pressure-adjusted) air mass. [unitless]

    module_type : str, optional
        a string specifying a cell type. Values of 'cdte', 'monosi', 'xsi',
        'multisi', and 'polysi' (can be lower or upper case). If provided,
        module_type selects default coefficients for the following modules:

        * ``'cdte'`` - First Solar Series 4-2 CdTe module.
        * ``'monosi'``, ``'xsi'`` - First Solar TetraSun module.
        * ``'multisi'``, ``'polysi'`` - anonymous multi-crystalline silicon
          module.
        * ``'cigs'`` - anonymous copper indium gallium selenide module.
        * ``'asi'`` - anonymous amorphous silicon module.

        The module used to calculate the spectral correction
        coefficients corresponds to the Multi-crystalline silicon
        Manufacturer 2 Model C from [3]_. The spectral response (SR) of CIGS
        and a-Si modules used to derive coefficients can be found in [4]_

    coefficients : array-like, optional
        Allows for entry of user-defined spectral correction
        coefficients. Coefficients must be of length 6. Derivation of
        coefficients requires use of SMARTS and PV module quantum
        efficiency curve. Useful for modeling PV module types which are
        not included as defaults, or to fine tune the spectral
        correction to a particular PV module. Note that the parameters for
        modules with very similar quantum efficiency should be similar,
        in most cases limiting the need for module specific coefficients.

    min_precipitable_water : float, default 0.1
        minimum atmospheric precipitable water. Any ``precipitable_water``
        value lower than ``min_precipitable_water``
        is set to ``min_precipitable_water``. [cm]

    max_precipitable_water : float, default 8
        maximum atmospheric precipitable water. Any ``precipitable_water``
        value greater than ``max_precipitable_water``
        is set to ``np.nan``. [cm]

    min_airmass_absolute : float, default 0.58
        minimum absolute airmass. Any ``airmass_absolute`` value lower than
        ``min_airmass_absolute`` is set to ``min_airmass_absolute``. [unitless]

    max_airmass_absolute : float, default 10
        minimum absolute airmass. Any ``airmass_absolute`` value greater than
        ``max_airmass_absolute`` is set to ``max_airmass_absolute``. [unitless]

    Returns
    -------
    modifier: array-like
        spectral mismatch factor (unitless) which can be multiplied
        with broadband irradiance reaching a module's cells to estimate
        effective irradiance, i.e., the irradiance that is converted to
        electrical current.

    Notes
    ----
    The ``spectral_factor_firstsolar`` model takes the following form:

    .. math::

        M = c_1 + c_2 AM_a  + c_3 Pw  + c_4 AM_a^{0.5}
            + c_5 Pw^{0.5} + c_6 \frac{AM_a} {Pw^{0.5}}.

    The default values for the limits applied to :math:`AM_a` and :math:`Pw`
    via the ``min_precipitable_water``, ``max_precipitable_water``,
    ``min_airmass_absolute``, and ``max_airmass_absolute`` are set to prevent
    divergence of the model presented above. These default values were
    determined by the publication authors in the original pvlib-python
    implementation (:pull:`208`).

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
    pw = np.atleast_1d(precipitable_water)
    pw = pw.astype('float64')
    if np.min(pw) < min_precipitable_water:
        pw = np.maximum(pw, min_precipitable_water)
        warn('Low precipitable water values replaced with '
             f'{min_precipitable_water} cm in the calculation of spectral '
             'mismatch.')

    if np.max(pw) > max_precipitable_water:
        pw[pw > max_precipitable_water] = np.nan
        warn('High precipitable water values replaced with np.nan in '
             'the calculation of spectral mismatch.')

    airmass_absolute = np.minimum(airmass_absolute, max_airmass_absolute)

    if np.min(airmass_absolute) < min_airmass_absolute:
        airmass_absolute = np.maximum(airmass_absolute, min_airmass_absolute)
        warn('Low airmass values replaced with 'f'{min_airmass_absolute} in '
             'the calculation of spectral mismatch.')
        # pvlib.atmosphere.get_absolute_airmass(1,
        # pvlib.atmosphere.alt2pres(4340)) = 0.58 Elevation of
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

    coeff = coefficients
    ama = airmass_absolute
    modifier = (
        coeff[0] + coeff[1]*ama + coeff[2]*pw + coeff[3]*np.sqrt(ama) +
        coeff[4]*np.sqrt(pw) + coeff[5]*ama/np.sqrt(pw))

    return modifier


def spectral_factor_sapm(airmass_absolute, module):
    """
    Calculates the spectral mismatch factor, :math:`f_1`,
    using the Sandia Array Performance Model approach.

    The SAPM spectral factor function is part of the broader Sandia Array
    Performance Model, which defines five points on an IV curve using empirical
    module-specific coefficients. Module coefficients for the SAPM are
    available in the SAPM database and can be retrieved for use in the
    ``module`` parameter through
    :py:func:`pvlib.pvsystem.retrieve_sam()`. More details on the
    SAPM can be found in [1]_, while a full description of the procedure to
    determine the empirical model coefficients, including those for the SAPM
    spectral correction, can be found in [2]_.

    Parameters
    ----------
    airmass_absolute : numeric
        Absolute airmass [unitless]

        Note: ``np.nan`` airmass values will result in 0 output.

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM parameters.
        Must contain keys `'A0'` through `'A4'`.
        See the :py:func:`pvlib.pvsystem.sapm` notes section for more details.

    Returns
    -------
    f1 : numeric
        The spectral mismatch factor. [unitless]

    Notes
    -----
    The SAPM spectral correction functions parameterises :math:`f_1` as a
    fourth order polynomial function of absolute air mass:

    .. math::

        f_1 = a_0 + a_1 AM_a + a_2 AM_a^2 + a_3 AM_a^3 + a_4 AM_a^4,

    where :math:`f_1` is the spectral mismatch factor, :math:`a_{0-4}` are
    the module-specific coefficients, and :math:`AM_a` is the absolute airmass,
    which is calculated by applying a pressure correction to the relative
    airmass. More detail on how this spectral correction function was developed
    can be found in [3]_.

    References
    ----------
    .. [1] King, D., Kratochvil, J., and Boyson W. (2004), "Sandia
           Photovoltaic Array Performance Model", (No. SAND2004-3535), Sandia
           National Laboratories, Albuquerque, NM (United States).
           :doi:`10.2172/919131`
    .. [2] King, B., Hansen, C., Riley, D., Robinson, C., and Pratt, L.
           (2016). Procedure to determine coefficients for the Sandia Array
           Performance Model (SAPM) (No. SAND2016-5284). Sandia National
           Laboratories, Albuquerque, NM (United States).
           :doi:`10.2172/1256510`
    .. [3] King, D., Kratochvil, J., and Boyson, W. "Measuring solar spectral
           and angle-of-incidence effects on photovoltaic modules and solar
           irradiance sensors." Conference Record of the 26th IEEE Potovoltaic
           Specialists Conference (PVSC). IEEE, 1997.
           :doi:`10.1109/PVSC.1997.654283`

    """

    am_coeff = [module['A4'], module['A3'], module['A2'], module['A1'],
                module['A0']]

    spectral_loss = np.polyval(am_coeff, airmass_absolute)

    spectral_loss = np.where(np.isnan(spectral_loss), 0, spectral_loss)

    spectral_loss = np.maximum(0, spectral_loss)

    if isinstance(airmass_absolute, pd.Series):
        spectral_loss = pd.Series(spectral_loss, airmass_absolute.index)

    return spectral_loss


def spectral_factor_caballero(precipitable_water, airmass_absolute, aod500,
                              module_type=None, coefficients=None):
    r"""
    Estimate a technology-specific spectral mismatch modifier from
    airmass, aerosol optical depth, and atmospheric precipitable water,
    using the Caballero model.

    The model structure was motivated by examining the effect of these three
    atmospheric parameters on simulated irradiance spectra and spectral
    modifiers.  However, the coefficient values reported in [1]_ and
    available here via the ``module_type`` parameter were determined
    by fitting the model equations to spectral factors calculated from
    global tilted spectral irradiance measurements taken in the city of
    Jaén, Spain. See [1]_ for details.

    Parameters
    ----------
    precipitable_water : numeric
        atmospheric precipitable water. [cm]

    airmass_absolute : numeric
        absolute (pressure-adjusted) airmass. [unitless]

    aod500 : numeric
        atmospheric aerosol optical depth at 500 nm. [unitless]

    module_type : str, optional
        One of the following PV technology strings from [1]_:

        * ``'cdte'`` - anonymous CdTe module.
        * ``'monosi'`` - anonymous sc-si module.
        * ``'multisi'`` - anonymous mc-si- module.
        * ``'cigs'`` - anonymous copper indium gallium selenide module.
        * ``'asi'`` - anonymous amorphous silicon module.
        * ``'perovskite'`` - anonymous pervoskite module.

    coefficients : array-like, optional
        user-defined coefficients, if not using one of the default coefficient
        sets via the ``module_type`` parameter.

    Returns
    -------
    modifier: numeric
        spectral mismatch factor (unitless) which is multiplied
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
        :doi:`10.1109/jphotov.2017.2787019`
    """

    if module_type is None and coefficients is None:
        raise ValueError('Must provide either `module_type` or `coefficients`')
    if module_type is not None and coefficients is not None:
        raise ValueError('Only one of `module_type` and `coefficients` should '
                         'be provided')

    # Experimental coefficients from [1]_.
    # The extra 0/1 coefficients at the end are used to enable/disable
    # terms to match the different equation forms in Table 1.
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

    if module_type is not None:
        coeff = _coefficients[module_type]
    else:
        coeff = coefficients

    # Evaluate spectral correction factor
    ama = airmass_absolute
    aod500_ref = 0.084
    pw_ref = 1.4164

    f_AM = (
        coeff[0]
        + coeff[1] * ama
        + coeff[2] * ama**2
        + coeff[3] * ama**3
        + coeff[4] * ama**4
    )
    # Eq 6, with Table 1
    f_AOD = (aod500 - aod500_ref) * (
        coeff[5]
        + coeff[10] * coeff[6] * ama
        + coeff[11] * coeff[6] * np.log(ama)
        + coeff[7] * ama**2
    )
    # Eq 7, with Table 1
    f_PW = (precipitable_water - pw_ref) * (
        coeff[8]
        + coeff[9] * np.log(ama)
    )
    modifier = f_AM + f_AOD + f_PW  # Eq 5
    return modifier


def spectral_factor_pvspec(airmass_absolute, clearsky_index,
                           module_type=None, coefficients=None):
    r"""
    Estimate a technology-specific spectral mismatch modifier from absolute
    airmass and clear sky index using the PVSPEC model.

    The PVSPEC spectral mismatch model includes the effects of cloud cover on
    the irradiance spectrum. Model coefficients are derived using spectral
    irradiance and other meteorological data from eight locations. Coefficients
    for six module types are available via the ``module_type`` parameter.
    More details on the model can be found in [1]_.

    Parameters
    ----------
    airmass_absolute : numeric
        absolute (pressure-adjusted) airmass. [unitless]

    clearsky_index: numeric
        clear sky index. [unitless]

    module_type : str, optional
        One of the following PV technology strings from [1]_:

        * ``'fs4-1'`` - First Solar series 4-1 and earlier CdTe module.
        * ``'fs4-2'`` - First Solar series 4-2 and later CdTe module.
        * ``'monosi'`` - anonymous monocrystalline Si module.
        * ``'multisi'`` - anonymous multicrystalline Si module.
        * ``'cigs'`` - anonymous copper indium gallium selenide module.
        * ``'asi'`` - anonymous amorphous silicon module.

    coefficients : array-like, optional
        user-defined coefficients, if not using one of the default coefficient
        sets via the ``module_type`` parameter.

    Returns
    -------
    mismatch: numeric
        spectral mismatch factor (unitless) which is multiplied
        with broadband irradiance reaching a module's cells to estimate
        effective irradiance, i.e., the irradiance that is converted to
        electrical current.

    Notes
    -----
    The PVSPEC model parameterises the spectral mismatch factor as a function
    of absolute air mass and the clear sky index as follows:

    .. math::

        M = a_1 k_c^{a_2} AM_a^{a_3},

    where :math:`M` is the spectral mismatch factor, :math:`k_c` is the clear
    sky index, :math:`AM_a` is the absolute air mass, and :math:`a_1, a_2, a_3`
    are module-specific coefficients. In the PVSPEC model publication, absolute
    air mass (denoted as :math:`AM`) is estimated starting from the Kasten and
    Young relative air mass [2]_. The clear sky index, which is the ratio of
    GHI to clear sky GHI, uses the ESRA model [3]_ to estimate the clear sky
    GHI with monthly Linke turbidity values from [4]_ as inputs.

    References
    ----------
    .. [1] Pelland, S., Beswick, C., Thevenard, D., Côté, A., Pai, A. and
       Poissant, Y., 2020. Development and testing of the PVSPEC model of
       photovoltaic spectral mismatch factor. In 2020 47th IEEE Photovoltaic
       Specialists Conference (PVSC) (pp. 1258-1264). IEEE.
       :doi:`10.1109/PVSC45281.2020.9300932`
    .. [2] Kasten, F. and Young, A.T., 1989. Revised optical air mass tables
       and approximation formula. Applied Optics, 28(22), pp.4735-4738.
       :doi:`10.1364/AO.28.004735`
    .. [3] Rigollier, C., Bauer, O. and Wald, L., 2000. On the clear sky model
       of the ESRA—European Solar Radiation Atlas—with respect to the Heliosat
       method. Solar energy, 68(1), pp.33-48.
       :doi:`10.1016/S0038-092X(99)00055-9`
    .. [4] SoDa website monthly Linke turbidity values:
       http://www.soda-pro.com/
    """

    _coefficients = {}
    _coefficients['multisi'] = (0.9847, -0.05237, 0.03034)
    _coefficients['monosi'] = (0.9845, -0.05169, 0.03034)
    _coefficients['fs4-2'] = (1.002, -0.07108, 0.02465)
    _coefficients['fs4-1'] = (0.9981, -0.05776, 0.02336)
    _coefficients['cigs'] = (0.9791, -0.03904, 0.03096)
    _coefficients['asi'] = (1.051, -0.1033, 0.009838)

    if module_type is not None and coefficients is None:
        coefficients = _coefficients[module_type.lower()]
    elif module_type is None and coefficients is not None:
        pass
    elif module_type is None and coefficients is None:
        raise ValueError('No valid input provided, both module_type and ' +
                         'coefficients are None. module_type can be one of ' +
                         ", ".join(_coefficients.keys()))
    else:
        raise ValueError('Cannot resolve input, must supply only one of ' +
                         'module_type and coefficients. module_type can be ' +
                         'one of' ", ".join(_coefficients.keys()))

    coeff = coefficients
    ama = airmass_absolute
    kc = clearsky_index
    mismatch = coeff[0]*np.power(kc, coeff[1])*np.power(ama, coeff[2])

    return mismatch


def spectral_factor_jrc(airmass, clearsky_index, module_type=None,
                        coefficients=None):
    r"""
    Estimate a technology-specific spectral mismatch modifier from
    airmass and clear sky index using the JRC model.

    The JRC spectral mismatch model includes the effects of cloud cover on
    the irradiance spectrum. Model coefficients are derived using measurements
    of irradiance and module performance at the Joint Research Centre (JRC) in
    Ispra, Italy (45.80N, 8.62E). Coefficients for two module types are
    available via the ``module_type`` parameter. More details on the model can
    be found in [1]_.

    Parameters
    ----------
    airmass : numeric
        relative airmass. [unitless]

    clearsky_index: numeric
        clear sky index. [unitless]

    module_type : str, optional
        One of the following PV technology strings from [1]_:

        * ``'cdte'`` - anonymous CdTe module.
        * ``'multisi'`` - anonymous multicrystalline Si module.

    coefficients : array-like, optional
        user-defined coefficients, if not using one of the default coefficient
        sets via the ``module_type`` parameter.

    Returns
    -------
    mismatch: numeric
        spectral mismatch factor (unitless) which is multiplied
        with broadband irradiance reaching a module's cells to estimate
        effective irradiance, i.e., the irradiance that is converted to
        electrical current.

    Notes
    -----
    The JRC model parameterises the spectral mismatch factor as a function
    of air mass and the clear sky index as follows:

    .. math::

        M = 1 + a_1(e^{-k_c}-e^{-1}) + a_2(k_c-1)+a_3(AM-1.5),

    where :math:`M` is the spectral mismatch factor, :math:`k_c` is the clear
    sky index, :math:`AM` is the air mass, :math:`e` is Euler's number, and
    :math:`a_1, a_2, a_3` are module-specific coefficients. The :math:`a_n`
    coefficients available via the ``coefficients`` parameter differ from the
    :math:`k_n` coefficients documented in [1]_ in that they are normalised by
    the specific short-circuit current value, :math:`I_{sc0}^*`, which is the
    expected short-circuit current at standard test conditions indoors. The
    model used to estimate the air mass (denoted as :math:`AM`) is not stated
    in the original publication. The authors of [1]_ used the ESRA model [2]_
    to estimate the clear sky GHI for the clear sky index, which is the ratio
    of GHI to clear sky GHI. Also, prior to the calculation of :math:`k_c`, the
    irradiance measurements were corrected for angle of incidence using the
    Martin and Ruiz model [3]_.

    References
    ----------
    .. [1] Huld, T., Sample, T., and Dunlop, E., 2009. A simple model
       for estimating the influence of spectrum variations on PV performance.
       In Proceedings of the 24th European Photovoltaic Solar Energy
       Conference, Hamburg, Germany pp. 3385-3389. 2009. Accessed at:
       https://www.researchgate.net/publication/256080247
    .. [2] Rigollier, C., Bauer, O., and Wald, L., 2000. On the clear sky model
       of the ESRA—European Solar Radiation Atlas—with respect to the Heliosat
       method. Solar energy, 68(1), pp.33-48.
       :doi:`10.1016/S0038-092X(99)00055-9`
    .. [3] Martin, N. and Ruiz, J. M., 2001. Calculation of the PV modules
       angular losses under field conditions by means of an analytical model.
       Solar Energy Materials and Solar Cells, 70(1), 25-38.
       :doi:`10.1016/S0927-0248(00)00408-6`
    """

    _coefficients = {}
    _coefficients['multisi'] = (0.00172, 0.000508, 0.00000357)
    _coefficients['cdte'] = (0.000643, 0.000130, 0.0000108)
    # normalise coefficients by I*sc0, see [1]
    _coefficients = {
        'multisi': tuple(x / 0.00348 for x in _coefficients['multisi']),
        'cdte': tuple(x / 0.001150 for x in _coefficients['cdte'])
    }
    if module_type is not None and coefficients is None:
        coefficients = _coefficients[module_type.lower()]
    elif module_type is None and coefficients is not None:
        pass
    elif module_type is None and coefficients is None:
        raise ValueError('No valid input provided, both module_type and ' +
                         'coefficients are None. module_type can be one of ' +
                         ", ".join(_coefficients.keys()))
    else:
        raise ValueError('Cannot resolve input, must supply only one of ' +
                         'module_type and coefficients. module_type can be ' +
                         'one of' ", ".join(_coefficients.keys()))

    coeff = coefficients
    mismatch = (
        1
        + coeff[0] * (np.exp(-clearsky_index) - np.exp(-1))
        + coeff[1] * (clearsky_index - 1)
        + coeff[2] * (airmass - 1.5)
    )
    return mismatch
