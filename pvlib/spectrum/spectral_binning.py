"""
The ``spectral_binning`` module in the ``spectrum`` package provides functions
for calculations related to reducing (binning) large spectral irradiance
datasets into smaller, representative datasets.
"""

import pvlib
import numpy as np
import pandas as pd

from scipy.integrate import trapezoid
from pvlib.spectrum.irradiance import average_photon_energy


def spectral_binning(spectra, n_bins):
    r"""
    Reduce a set of spectra to a few representative spectra by equal-energy
    binning.

    Spectra are sorted from red-rich to blue-rich by average photon energy
    (APE) and split into bins that each hold an equal share of the total
    incident energy. Each bin is represented by its mean "proxy" spectrum.

    Parameters
    ----------
    spectra : pandas.DataFrame
        Spectral irradiance, one spectrum per row with wavelength [nm] as
        column headers. Must be positive. [Wm⁻²nm⁻¹]
        See :term:`spectra`.
    n_bins : int
        Number of bins.

    Returns
    -------
    proxy_spectra : pandas.DataFrame
        Mean spectrum of each bin, indexed by bin number. [Wm⁻²nm⁻¹]
    bins : pandas.DataFrame
        Bin properties: ``ape_min``, ``ape_max`` and ``ape`` (APE of the
        proxy spectrum) [eV], ``n_spectra``, and ``energy_fraction`` (share
        of the total incident energy).
    labels : pandas.Series
        Bin number of each input spectrum. ``<NA>`` for spectra excluded
        because their irradiance is zero or NaN.

    Notes
    -----
    This function implements the spectral binning method of [1]_, adapted
    from [2]_. Spectra are sorted by APE (see
    :py:func:`~pvlib.spectrum.average_photon_energy`) and their cumulative
    broadband irradiance :math:`G = \int \Phi(\lambda) \, d\lambda` is split
    into ``n_bins`` equal parts. Bin :math:`k` is represented by the mean of
    its :math:`n_k` spectra,

    .. math::

        \Phi_k^\mathrm{p} = \frac{1}{n_k} \sum_{l=1}^{n_k} \Phi_{kl},

    which conserves the total incident energy. Fewer than ``n_bins`` bins are
    returned if a single spectrum holds more than ``1/n_bins`` of the total.

    Each spectrum is assumed to represent an equal time interval, so data with
    unequal intervals should be resampled first. Night-time gaps have no
    effect, but missing daylight data bias the result towards the conditions
    present.

    References
    ----------
    .. [1] R. Witteck, J. F. Geisz, E. L. Warren, and W. E. McMahon, "Spectral
       Effects on the Energy Harvesting Efficiency of Two- and Four-Terminal
       Tandem Photovoltaics," Solar RRL, vol. 8, no. 3, p. 2300782, 2024,
       :doi:`10.1002/solr.202300782`.
    .. [2] I. García, W. E. McMahon, A. Habte, J. F. Geisz, M. A. Steiner,
       M. Sengupta, and D. J. Friedman, "Spectral binning for energy
       production calculations and multijunction solar cell design," Progress
       in Photovoltaics: Research and Applications, vol. 26, no. 1, pp. 48-54,
       2018, :doi:`10.1002/pip.2943`.
    """
