"""
The ``mismatch`` module provides functions for spectral mismatch calculations.
"""

import pvlib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os

def get_sample_sr(wavelength=None):
    '''
    Generate a generic smooth c-si spectral response for tests and experiments.

    Notes
    -----
    This sr is based on measurements taken on a reference cell.
    The measured data points have been adjusted by PV Performance Labs so that
    standard cubic spline interpolation produces a curve without oscillations.
    '''
    SR_DATA = np.array([[ 290, 0.00],
                        [ 350, 0.27],
                        [ 400, 0.37],
                        [ 500, 0.52],
                        [ 650, 0.71],
                        [ 800, 0.88],
                        [ 900, 0.97],
                        [ 957, 1.00],
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

    Notes
    -----
    More information about reference spectra is found here:
    https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html

    The original file containing the reference spectra can be found here:
    https://www.nrel.gov/grid/solar-resource/assets/data/astmg173.xls
    '''

    pvlib_path = pvlib.__path__[0]
    filepath = os.path.join(pvlib_path, 'data', 'astmg173.xls')

    g173 = pd.read_excel(filepath, index_col=0, skiprows=1)
    am15g = g173['Global tilt  W*m-2*nm-1']

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
    Calculate the spectral mismatch under a given measured spectrum with
    respect to a reference spectrum.

    Parameters
    ----------
    sr: pandas.Series
        The spectral response of one (photovoltaic) device.
    e_sun: pandas.DataFrame or pandase.Series
        One or more irradiance spectra.  Usually a pandas.DataFrame with
        wavelength as column index.  A single spectrum may be given as a
        pandas.Series having a wavelength index.
    e_ref: pandas.Series, optional
        The reference spectrum to use for the mismatch calculation. The default
        is the ASTM G173-03 global tilted spectrum.

    Returns
    -------
    smm: pandas.Series or float

    Notes
    -----
    If the default reference spectrum is used, it is reindexed
    to the wavelengths of the measured spectrum.

    If e_ref is provided as an argument, it is used without modification.

    The spectral response is always interpolated to the wavelengths of the
    spectrum with which is it multiplied.
    """
    # get the reference spectrum at wavelengths matching the measured spectra
    if e_ref is None:
        e_ref = get_am15g(wavelength=e_sun.T.index)

    # interpolate the sr at the wavelengths of the spectra
    # reference spectrum wavelengths may differ if e_ref is from caller
    sr_sun = np.interp(e_sun.T.index, sr.index, sr, left=0.0, right=0.0)
    sr_ref = np.interp(e_ref.T.index, sr.index, sr, left=0.0, right=0.0)

    # a helper function to make usable fraction calculations more readable
    integrate = lambda e: np.trapz(e, x=e.T.index, axis=-1)

    # calculate usable fractions
    uf_sun = integrate(e_sun * sr_sun) / integrate(e_sun)
    uf_ref = integrate(e_ref * sr_ref) / integrate(e_ref)

    # mismatch is the ratio or quotient of the usable fractions
    smm = uf_sun / uf_ref

    if isinstance(e_sun, pd.DataFrame):
        smm = pd.Series(smm, index=e_sun.index)

    return smm
