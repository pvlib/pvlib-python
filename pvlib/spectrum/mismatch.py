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
    Read the ASTM G173 AM1.5 global spectrum, optionally interpolated to the
    specified wavelength(s).

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
