"""
The ``spectral_irradiance`` module provides functions for calculations related
to spectral irradiance  data.
"""

import pvlib
from pvlib._deprecation import deprecated
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from pathlib import Path
from warnings import warn
from functools import partial
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
