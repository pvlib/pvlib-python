"""
The ``mismatch`` module provides functions for spectral mismatch calculations.
"""

import pvlib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os


def get_example_spectral_response(wavelength=None):
    '''
    Generate a generic smooth spectral response (SR) for tests and experiments.

    Parameters
    ----------
    wavelength: 1-D sequence of numeric, optional
        Wavelengths at which the spectral response should be interpolated.
        By default the wavelengths are from 280 to 1200 in 5 nm intervals. [nm]

    Returns
    -------
    sr: pandas.Series
        The relative spectral response indexed by wavelength in nm. [-]

    Notes
    -----
    This spectral response is based on measurements taken on a c-Si cell.
    The measured data points have been adjusted by PV Performance Labs so that
    standard cubic spline interpolation produces a curve without oscillations
    as shown in [1]_, which makes it suitable for experimenting with spectral
    data of different resolutions.

    References
    ----------
    .. [1] Driesse, Anton, and Stein, Joshua. "Global Normal Spectral
       Irradiance in Albuquerque: a One-Year Open Dataset for PV Research".
       United States 2020. :doi:`10.2172/1814068`.
    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Aug. 2022

    SR_DATA = np.array([[ 290, 0.00],
                        [ 350, 0.27],
                        [ 400, 0.37],
                        [ 500, 0.52],
                        [ 650, 0.71],
                        [ 800, 0.88],
                        [ 900, 0.97],
                        [ 950, 1.00],
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
    sr.name = 'sr'

    return sr


def get_am15g(wavelength=None):
    '''
    Read the ASTM G173-03 AM1.5 global tilted spectrum, optionally interpolated
    to the specified wavelength(s).

    Global (tilted) irradiance includes direct and diffuse irradiance from sky
    and ground reflections, and is more formally called hemispherical
    irradiance (on a tilted surface).

    Parameters
    ----------
    wavelength: 1-D sequence of numeric, optional
        The wavelengths at which the spectrum should be interpolated.
        By default the 2002 wavelengths of the standard are returned. [nm]

    Returns
    -------
    am15g: pandas.Series
        The AM1.5g standard spectrum indexed by wavelength in nm. [(W/m^2)/nm]

    Notes
    -----
    This function uses linear interpolation.  If the values in ``wavelength``
    are too widely spaced, the integral of the spectrum may deviate from the
    standard value of 1000.37 W/m^2.

    The values in the data file provided with pvlib-python are copied from an
    Excel file distributed by NREL, which is found here:
    https://www.nrel.gov/grid/solar-resource/assets/data/astmg173.xls

    More information about reference spectra is found here:
    https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html

    References
    ----------
    .. [1] ASTM "G173-03 Standard Tables for Reference Solar Spectral
        Irradiances: Direct Normal and Hemispherical on 37° Tilted Surface."
    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Aug. 2022

    pvlib_path = pvlib.__path__[0]
    filepath = os.path.join(pvlib_path, 'data', 'astm_g173_am15g.csv')

    am15g = pd.read_csv(filepath, index_col=0).squeeze()

    if wavelength is not None:
        interpolator = interp1d(am15g.index, am15g,
                                kind='linear',
                                bounds_error=False,
                                fill_value=0.0,
                                copy=False,
                                assume_sorted=True)

        am15g = pd.Series(data=interpolator(wavelength), index=wavelength)

    am15g.index.name = 'wavelength'
    am15g.name = 'am15g'

    return am15g


def calc_spectral_mismatch(sr, e_sun, e_ref=None):
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
        having wavelength in nm as column index.  A single spectrum may be
        be given as a pandas.Series having wavelength in nm as index.
        [W/m^2/nm]

    e_ref: pandas.Series, optional
        The reference spectrum to use for the mismatch calculation.
        The index of the Series must contain wavelength values in nm.
        The default is the ASTM G173-03 global tilted spectrum. [W/m^2/nm]

    Returns
    -------
    smm: pandas.Series or float if a single measured spectrum is provided. [-]

    Notes
    -----
    If the default reference spectrum is used it is linearly interpolated
    to the wavelengths of the measured spectrum.  To achieve alternate
    behavior e_ref can be transformed before calling this function and
    provided as an argument.

    The spectral response is linearly interpolated to the wavelengths of the
    spectrum with which is it multiplied internally (e_sun and e_ref). To
    achieve alternate behavior the spectral response can be transformed
    before calling this function.

    The standards describing mismatch calculations focus on indoor laboratory
    applications, but are applicable to outdoor performance as well.
    The most recent version of ASTM E973 [1]_ is somewhat more difficult to
    read because it includes adjustments for temperature dependency of the
    spectral response, which led to a formulation using quantum efficiency
    (QE). IEC 60904-7 is clearer and also discusses the use of a
    broadband reference device.

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
        e_ref = get_am15g(wavelength=e_sun.T.index)

    # interpolate the sr at the wavelengths of the spectra
    # reference spectrum wavelengths may differ if e_ref is from caller
    sr_sun = np.interp(e_sun.T.index, sr.index, sr, left=0.0, right=0.0)
    sr_ref = np.interp(e_ref.T.index, sr.index, sr, left=0.0, right=0.0)

    # a helper function to make usable fraction calculations more readable
    def integrate(e):
        return np.trapz(e, x=e.T.index, axis=-1)

    # calculate usable fractions
    uf_sun = integrate(e_sun * sr_sun) / integrate(e_sun)
    uf_ref = integrate(e_ref * sr_ref) / integrate(e_ref)

    # mismatch is the ratio or quotient of the usable fractions
    smm = uf_sun / uf_ref

    if isinstance(e_sun, pd.DataFrame):
        smm = pd.Series(smm, index=e_sun.index)

    return smm
