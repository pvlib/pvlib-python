"""
The ``irradiance`` module in the ``spectrum`` package provides functions for
calculations related to spectral irradiance data.
"""

import pvlib
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from scipy import constants
from scipy.integrate import trapezoid


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


def average_photon_energy(spectra):
    r"""
    Calculate the average photon energy of one or more spectral irradiance
    distributions.

    Parameters
    ----------
    spectra : pandas.Series or pandas.DataFrame

        Spectral irradiance, must be positive [Wm⁻²nm⁻¹].
        See :term:`spectra`.

        A single spectrum must be a :py:class:`pandas.Series` with wavelength
        [nm] as the index, while multiple spectra must be rows in a
        :py:class:`pandas.DataFrame` with column headers as wavelength [nm].

    Returns
    -------
    ape : numeric or pandas.Series
        Average Photon Energy [eV].
        Note: returns ``np.nan`` in the case of all-zero spectral irradiance
        input.

    Notes
    -----
    The average photon energy (APE) is an index used to characterise the solar
    spectrum. It has been used widely in the physics literature since the
    1900s, but its application for solar spectral irradiance characterisation
    in the context of PV performance modelling was proposed in 2002 [1]_. The
    APE is calculated based on the principle that a photon's energy is
    inversely proportional to its wavelength:

    .. math::

        E_\gamma = \frac{hc}{\lambda},

    where :math:`E_\gamma` is the energy of a photon with wavelength
    :math:`\lambda`, :math:`h` is the Planck constant, and :math:`c` is the
    speed of light. Therefore, the average energy of all photons within a
    single spectral irradiance distribution provides an indication of the
    general shape of the spectrum. A higher average photon energy
    (shorter wavelength) indicates a blue-shifted spectrum, while a lower
    average photon energy (longer wavelength) would indicate a red-shifted
    spectrum. This value of the average photon energy can be calculated by
    dividing the total energy in the spectrum by the total number of photons
    in the spectrum as follows [1]_:

    .. math::

        \overline{E_\gamma} = \frac{1}{q} \cdot \frac{\int G(\lambda) \,
                                                       d\lambda}
        {\int \Phi(\lambda) \, d\lambda}.

    :math:`\Phi(\lambda)` is the photon flux density as a function of
    wavelength, :math:`G(\lambda)` is the spectral irradiance, :math:`q` is the
    elementary charge used here so that the average photon energy,
    :math:`\overline{E_\gamma}`, is expressed in electronvolts (eV). The
    integrals are computed over the full wavelength range of the ``spectra``
    parameter.

    References
    ----------
    .. [1] Jardine, C., et al., 2002, January. Influence of spectral effects on
       the performance of multijunction amorphous silicon cells. In Proc.
       Photovoltaic in Europe Conference (pp. 1756-1759).
    """

    if not isinstance(spectra, (pd.Series, pd.DataFrame)):
        raise TypeError('`spectra` must be either a'
                        ' pandas Series or DataFrame')

    if (spectra < 0).any().any():
        raise ValueError('Spectral irradiance data must be positive')

    hclambda = pd.Series((constants.h*constants.c)/(spectra.T.index*1e-9))
    hclambda.index = spectra.T.index
    pfd = spectra.div(hclambda)

    def integrate(e):
        return trapezoid(e, x=e.T.index, axis=-1)

    int_spectra = integrate(spectra)
    int_pfd = integrate(pfd)

    with np.errstate(invalid='ignore'):
        ape = (1/constants.elementary_charge)*int_spectra/int_pfd

    if isinstance(spectra, pd.DataFrame):
        ape = pd.Series(ape, index=spectra.index)

    return ape
