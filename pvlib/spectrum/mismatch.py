"""
The ``mismatch`` module provides functions for spectral mismatch calculations.
"""

import pvlib
from pvlib._deprecation import deprecated
from pvlib.tools import normalize_max2one
import numpy as np
import pandas as pd
import scipy.constants
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from pathlib import Path
from warnings import warn
from functools import partial


_PLANCK_BY_LIGHT_SPEED_OVER_ELEMENTAL_CHARGE_BY_BILLION = (
    scipy.constants.speed_of_light
    * scipy.constants.Planck
    / scipy.constants.elementary_charge
    * 1e9
)


def get_example_spectral_response(wavelength=None):
    '''
    Generate a generic smooth spectral response (SR) for tests and experiments.

    Parameters
    ----------
    wavelength: 1-D sequence of numeric, optional
        Wavelengths at which spectral response values are generated.
        By default ``wavelength`` is from 280 to 1200 in 5 nm intervals. [nm]

    Returns
    -------
    spectral_response : pandas.Series
        The relative spectral response indexed by ``wavelength`` in nm. [-]

    Notes
    -----
    This spectral response is based on measurements taken on a c-Si cell.
    A small number of points near the measured curve are used to define
    a cubic spline having no undue oscillations, as shown in [1]_.  The spline
    can be interpolated at arbitrary wavelengths to produce a continuous,
    smooth curve , which makes it suitable for experimenting with spectral
    data of different resolutions.

    References
    ----------
    .. [1] Driesse, Anton, and Stein, Joshua. "Global Normal Spectral
       Irradiance in Albuquerque: a One-Year Open Dataset for PV Research".
       United States 2020. :doi:`10.2172/1814068`.
    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Aug. 2022

    SR_DATA = np.array([[290, 0.00],
                        [350, 0.27],
                        [400, 0.37],
                        [500, 0.52],
                        [650, 0.71],
                        [800, 0.88],
                        [900, 0.97],
                        [950, 1.00],
                        [1000, 0.93],
                        [1050, 0.58],
                        [1100, 0.21],
                        [1150, 0.05],
                        [1190, 0.00]]).transpose()

    if wavelength is None:
        resolution = 5.0
        wavelength = np.arange(280, 1200 + resolution, resolution)

    interpolator = interp1d(SR_DATA[0], SR_DATA[1],
                            kind='cubic',
                            bounds_error=False,
                            fill_value=0.0,
                            copy=False,
                            assume_sorted=True)

    sr = pd.Series(data=interpolator(wavelength), index=wavelength)

    sr.index.name = 'wavelength'
    sr.name = 'spectral_response'

    return sr


@deprecated(
    since="0.11",
    removal="0.12",
    name="pvlib.spectrum.get_am15g",
    alternative="pvlib.spectrum.get_reference_spectra",
    addendum=(
        "The new function reads more data. Use it with "
        + "standard='ASTM G173-03' and extract the 'global' column."
    ),
)
def get_am15g(wavelength=None):
    r"""
    Read the ASTM G173-03 AM1.5 global spectrum on a 37-degree tilted surface,
    optionally interpolated to the specified wavelength(s).

    Global (tilted) irradiance includes direct and diffuse irradiance from sky
    and ground reflections, and is more formally called hemispherical
    irradiance (on a tilted surface).  In the context of photovoltaic systems
    the irradiance on a flat receiver is frequently called plane-of-array (POA)
    irradiance.

    Parameters
    ----------
    wavelength: 1-D sequence of numeric, optional
        Wavelengths at which the spectrum is interpolated.
        By default the 2002 wavelengths of the standard are returned. [nm].

    Returns
    -------
    am15g: pandas.Series
        The AM1.5g standard spectrum indexed by ``wavelength``. [W/(m²nm)].

    Notes
    -----
    If ``wavelength`` is specified this function uses linear interpolation.

    If the values in ``wavelength`` are too widely spaced, the integral of the
    spectrum may deviate from the standard value of 1000.37 W/m².

    The values in the data file provided with pvlib-python are copied from an
    Excel file distributed by NREL, which is found here:
    https://www.nrel.gov/grid/solar-resource/assets/data/astmg173.xls

    More information about reference spectra is found here:
    https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html

    See Also
    --------
    pvlib.spectrum.get_reference_spectra : reads also the direct and
      extraterrestrial components of the spectrum.

    References
    ----------
    .. [1] ASTM "G173-03 Standard Tables for Reference Solar Spectral
       Irradiances: Direct Normal and Hemispherical on 37° Tilted Surface."
    """  # noqa: E501
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Aug. 2022
    # modified by @echedey-ls, as a wrapper of spectrum.get_reference_spectra
    standard = get_reference_spectra(wavelength, standard="ASTM G173-03")
    return standard["global"]


def get_reference_spectra(wavelengths=None, standard="ASTM G173-03"):
    r"""
    Read a standard spectrum specified by ``standard``, optionally
    interpolated to the specified wavelength(s).

    Defaults to ``ASTM G173-03`` AM1.5 standard [1]_, which returns
    ``extraterrestrial``, ``global`` and ``direct`` spectrum on a 37-degree
    tilted surface, optionally interpolated to the specified wavelength(s).

    Parameters
    ----------
    wavelengths : numeric, optional
        Wavelengths at which the spectrum is interpolated. [nm].
        If not provided, the original wavelengths from the specified standard
        are used. Values outside that range are filled with zeros.

    standard : str, default "ASTM G173-03"
        The reference standard to be read. Only the reference
        ``"ASTM G173-03"`` is available at the moment.

    Returns
    -------
    standard_spectra : pandas.DataFrame
        The standard spectrum by ``wavelength [nm]``. [W/(m²nm)].
        Column names are ``extraterrestrial``, ``direct`` and ``global``.

    Notes
    -----
    If ``wavelength`` is specified, linear interpolation is used.

    If the values in ``wavelength`` are too widely spaced, the integral of each
    spectrum may deviate from its standard value.
    For global spectra, it is about 1000.37 W/m².

    The values of the ASTM G173-03 provided with pvlib-python are copied from
    an Excel file distributed by NREL, which is found here [2]_:
    https://www.nrel.gov/grid/solar-resource/assets/data/astmg173.xls

    Examples
    --------
    >>> from pvlib import spectrum
    >>> am15 = spectrum.get_reference_spectra()
    >>> am15_extraterrestrial, am15_global, am15_direct = \
    >>>     am15['extraterrestrial'], am15['global'], am15['direct']
    >>> print(am15.head())
                extraterrestrial        global        direct
    wavelength
    280.0                  0.082  4.730900e-23  2.536100e-26
    280.5                  0.099  1.230700e-21  1.091700e-24
    281.0                  0.150  5.689500e-21  6.125300e-24
    281.5                  0.212  1.566200e-19  2.747900e-22
    282.0                  0.267  1.194600e-18  2.834600e-21

    >>> am15 = spectrum.get_reference_spectra([300, 500, 800, 1100])
    >>> print(am15)
                extraterrestrial   global    direct
    wavelength
    300                  0.45794  0.00102  0.000456
    500                  1.91600  1.54510  1.339100
    800                  1.12480  1.07250  0.988590
    1100                 0.60000  0.48577  0.461130

    References
    ----------
    .. [1] ASTM "G173-03 Standard Tables for Reference Solar Spectral
       Irradiances: Direct Normal and Hemispherical on 37° Tilted Surface."
    .. [2] “Reference Air Mass 1.5 Spectra,” www.nrel.gov.
       https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html
    """  # Contributed by Echedey Luis, inspired by Anton Driesse (get_am15g)
    SPECTRA_FILES = {
        "ASTM G173-03": "ASTMG173.csv",
    }
    pvlib_datapath = Path(pvlib.__path__[0]) / "data"

    try:
        filepath = pvlib_datapath / SPECTRA_FILES[standard]
    except KeyError:
        raise ValueError(
            f"Invalid standard identifier '{standard}'. Available "
            + "identifiers are: "
            + ", ".join(SPECTRA_FILES.keys())
        )

    standard = pd.read_csv(
        filepath,
        header=1,  # expect first line of description, then column names
        index_col=0,  # first column is "wavelength"
        dtype=float,
    )

    if wavelengths is not None:
        interpolator = partial(
            np.interp, xp=standard.index, left=0.0, right=0.0
        )
        standard = pd.DataFrame(
            index=wavelengths,
            data={
                col: interpolator(x=wavelengths, fp=standard[col])
                for col in standard.columns
            },
        )

    return standard


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
        e_ref = get_reference_spectra(wavelengths=e_sun.T.index)["global"]

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
                               max_precipitable_water=8):
    r"""
    Spectral mismatch modifier based on precipitable water and absolute
    (pressure-adjusted) air mass.

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
    precipitable_water : numeric
        atmospheric precipitable water. [cm]

    airmass_absolute : numeric
        absolute (pressure-adjusted) air mass. [unitless]

    module_type : str, optional
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
        is set to ``min_precipitable_water`` to avoid model divergence. [cm]

    max_precipitable_water : float, default 8
        maximum atmospheric precipitable water. Any ``precipitable_water``
        value greater than ``max_precipitable_water``
        is set to ``np.nan`` to avoid model divergence. [cm]

    Returns
    -------
    modifier: array-like
        spectral mismatch factor (unitless) which can be multiplied
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
    pw = np.atleast_1d(precipitable_water)
    pw = pw.astype('float64')
    if np.min(pw) < min_precipitable_water:
        pw = np.maximum(pw, min_precipitable_water)
        warn('Exceptionally low pw values replaced with '
             f'{min_precipitable_water} cm to prevent model divergence')

    # Warn user about Pw data that is exceptionally high
    if np.max(pw) > max_precipitable_water:
        pw[pw > max_precipitable_water] = np.nan
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


def spectral_factor_sapm(airmass_absolute, module):
    """
    Calculates the SAPM spectral loss coefficient, F1.

    Parameters
    ----------
    airmass_absolute : numeric
        Absolute airmass

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM performance
        parameters. See the :py:func:`sapm` notes section for more
        details.

    Returns
    -------
    F1 : numeric
        The SAPM spectral loss coefficient.

    Notes
    -----
    nan airmass values will result in 0 output.
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
    Jaén, Spain.  See [1]_ for details.

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
        * ``'monosi'``, - anonymous sc-si module.
        * ``'multisi'``, - anonymous mc-si- module.
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
        * ``'fs4-2'`` - First Solar 4-2 and later CdTe module.
        * ``'monosi'``, - anonymous monocrystalline Si module.
        * ``'multisi'``, - anonymous multicrystalline Si module.
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
    _coefficients['fs-2'] = (1.002, -0.07108, 0.02465)
    _coefficients['fs-4'] = (0.9981, -0.05776, 0.02336)
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


def sr_to_qe(sr, wavelength=None, normalize=False):
    """
    Convert spectral responsivities to quantum efficiencies.
    If ``wavelength`` is not provided, the spectral responsivity ``sr`` must be
    a :py:class:`pandas.Series` or :py:class:`pandas.DataFrame`, with the
    wavelengths in the index.

    Provide wavelengths in nanometers, [nm].

    Conversion is described in [1]_.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    sr : numeric, pandas.Series or pandas.DataFrame
        Spectral response, [A/W].
        Index must be the wavelength in nanometers, [nm].

    wavelength : numeric, optional
        Points where spectral response is measured, in nanometers, [nm].

    normalize : bool, default False
        If True, the quantum efficiency is normalized so that the maximum value
        is 1.
        For ``pandas.DataFrame``, normalization is done for each column.
        For 2D arrays, normalization is done for each sub-array.

    Returns
    -------
    quantum_efficiency : numeric, same type as ``sr``
        Quantum efficiency, in the interval [0, 1].

    Notes
    -----
    - If ``sr`` is of type ``pandas.Series`` or ``pandas.DataFrame``,
      column names will remain unchanged in the returned object.
    - If ``wavelength`` is provided it will be used independently of the
      datatype of ``sr``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pvlib import spectrum
    >>> wavelengths = np.array([350, 550, 750])
    >>> spectral_response = np.array([0.25, 0.40, 0.57])
    >>> quantum_efficiency = spectrum.sr_to_qe(spectral_response, wavelengths)
    >>> print(quantum_efficiency)
    array([0.88560142, 0.90170326, 0.94227991])

    >>> spectral_response_series = pd.Series(spectral_response, index=wavelengths, name="dataset")
    >>> qe = spectrum.sr_to_qe(spectral_response_series)
    >>> print(qe)
    350    0.885601
    550    0.901703
    750    0.942280
    Name: dataset, dtype: float64

    >>> qe = spectrum.sr_to_qe(spectral_response_series, normalize=True)
    >>> print(qe)
    350    0.939850
    550    0.956938
    750    1.000000
    Name: dataset, dtype: float64

    References
    ----------
    .. [1] “Spectral Response,” PV Performance Modeling Collaborative (PVPMC).
        https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/effective-irradiance/spectral-response/
    .. [2] “Spectral Response | PVEducation,” www.pveducation.org.
        https://www.pveducation.org/pvcdrom/solar-cell-operation/spectral-response

    See Also
    --------
    pvlib.spectrum.qe_to_sr
    """  # noqa: E501
    if wavelength is None:
        if hasattr(sr, "index"):  # true for pandas objects
            # use reference to index values instead of index alone so
            # sr / wavelength returns a series with the same name
            wavelength = sr.index.array
        else:
            raise TypeError(
                "'sr' must have an '.index' attribute"
                + " or 'wavelength' must be provided"
            )
    quantum_efficiency = (
        sr
        / wavelength
        * _PLANCK_BY_LIGHT_SPEED_OVER_ELEMENTAL_CHARGE_BY_BILLION
    )

    if normalize:
        quantum_efficiency = normalize_max2one(quantum_efficiency)

    return quantum_efficiency


def qe_to_sr(qe, wavelength=None, normalize=False):
    """
    Convert quantum efficiencies to spectral responsivities.
    If ``wavelength`` is not provided, the quantum efficiency ``qe`` must be
    a :py:class:`pandas.Series` or :py:class:`pandas.DataFrame`, with the
    wavelengths in the index.

    Provide wavelengths in nanometers, [nm].

    Conversion is described in [1]_.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    qe : numeric, pandas.Series or pandas.DataFrame
        Quantum efficiency.
        If pandas subtype, index must be the wavelength in nanometers, [nm].

    wavelength : numeric, optional
        Points where quantum efficiency is measured, in nanometers, [nm].

    normalize : bool, default False
        If True, the spectral response is normalized so that the maximum value
        is 1.
        For ``pandas.DataFrame``, normalization is done for each column.
        For 2D arrays, normalization is done for each sub-array.

    Returns
    -------
    spectral_response : numeric, same type as ``qe``
        Spectral response, [A/W].

    Notes
    -----
    - If ``qe`` is of type ``pandas.Series`` or ``pandas.DataFrame``,
      column names will remain unchanged in the returned object.
    - If ``wavelength`` is provided it will be used independently of the
      datatype of ``qe``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pvlib import spectrum
    >>> wavelengths = np.array([350, 550, 750])
    >>> quantum_efficiency = np.array([0.86, 0.90, 0.94])
    >>> spectral_response = spectrum.qe_to_sr(quantum_efficiency, wavelengths)
    >>> print(spectral_response)
    array([0.24277287, 0.39924442, 0.56862085])

    >>> quantum_efficiency_series = pd.Series(quantum_efficiency, index=wavelengths, name="dataset")
    >>> sr = spectrum.qe_to_sr(quantum_efficiency_series)
    >>> print(sr)
    350    0.242773
    550    0.399244
    750    0.568621
    Name: dataset, dtype: float64

    >>> sr = spectrum.qe_to_sr(quantum_efficiency_series, normalize=True)
    >>> print(sr)
    350    0.426950
    550    0.702128
    750    1.000000
    Name: dataset, dtype: float64

    References
    ----------
    .. [1] “Spectral Response,” PV Performance Modeling Collaborative (PVPMC).
        https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/effective-irradiance/spectral-response/
    .. [2] “Spectral Response | PVEducation,” www.pveducation.org.
        https://www.pveducation.org/pvcdrom/solar-cell-operation/spectral-response

    See Also
    --------
    pvlib.spectrum.sr_to_qe
    """  # noqa: E501
    if wavelength is None:
        if hasattr(qe, "index"):  # true for pandas objects
            # use reference to index values instead of index alone so
            # sr / wavelength returns a series with the same name
            wavelength = qe.index.array
        else:
            raise TypeError(
                "'qe' must have an '.index' attribute"
                + " or 'wavelength' must be provided"
            )
    spectral_responsivity = (
        qe
        * wavelength
        / _PLANCK_BY_LIGHT_SPEED_OVER_ELEMENTAL_CHARGE_BY_BILLION
    )

    if normalize:
        spectral_responsivity = normalize_max2one(spectral_responsivity)

    return spectral_responsivity
