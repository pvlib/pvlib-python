"""
The ``spectral_binning`` module in the ``spectrum`` package provides functions
for calculations related to reducing (binning) large spectral irradiance
datasets into smaller, representative datasets.
"""

import numpy as np
import pandas as pd

from scipy.integrate import trapezoid
from pvlib.spectrum.irradiance import average_photon_energy


def _assign_equal_energy_bins(ape, energy, n_bins):
    """
    Assign each spectrum to one of ``n_bins`` equal-energy bins.

    Parameters
    ----------
    ape : pandas.Series
        Average photon energy of each spectrum. [eV]
    energy : pandas.Series
        Energy of each spectrum, aligned with ``ape``. Any consistent unit.
        Must be positive.
    n_bins : int
        Number of bins.

    Returns
    -------
    labels : pandas.Series of int
        Bin index of each spectrum, with the same index as ``ape``. Bins may
        be empty if a single spectrum holds more than ``1/n_bins`` of the
        total energy.

    Notes
    -----
    Spectra are sorted by ``ape`` and assigned to bin ``k`` (0-indexed) if
    their cumulative energy lies in ``(k*E/n_bins, (k+1)*E/n_bins]``, where
    ``E`` is the total energy.
    """
    order = np.argsort(ape.to_numpy(), kind='stable')
    cum_energy = np.cumsum(energy.to_numpy()[order])
    thresholds = cum_energy[-1] * np.arange(1, n_bins) / n_bins
    labels = np.empty(len(ape), dtype=int)
    labels[order] = np.searchsorted(thresholds, cum_energy, side='left')

    return pd.Series(labels, index=ape.index, name='bin')


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
        Bin number of each input spectrum. ``np.nan`` for spectra whose APE
        is undefined, e.g. all-zero or NaN spectra.

    Notes
    -----
    This function implements the spectral binning method of [1]_, adapted
    from [2]_. First, the spectra are sorted by APE (see
    :py:func:`~pvlib.spectrum.average_photon_energy`). Next, they are divided
    into ``n_bins`` bins of equal energy, using the cumulative sum of their
    broadband irradiance :math:`G = \int \Phi(\lambda) \, d\lambda`. Finally,
    each bin :math:`k` is represented by the mean of its :math:`n_k` spectra,

    .. math::

        \Phi_k^\mathrm{p} = \frac{1}{n_k} \sum_{l=1}^{n_k} \Phi_{kl},

    which conserves the total incident energy.

    A spectrum cannot be split between bins, so a spectrum holding more than
    ``1/n_bins`` of the total energy can span an entire bin and leave it
    empty. Empty bins are omitted, so fewer than ``n_bins`` bins can be
    returned, e.g. when ``n_bins`` exceeds the number of spectra.

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
    ape = average_photon_energy(spectra)
    wavelength = spectra.columns.to_numpy(dtype=float)
    irradiance = pd.Series(trapezoid(spectra, x=wavelength, axis=1),
                           index=spectra.index)

    # APE is NaN for all-zero or NaN spectra, which can't be sorted
    valid = ape.notna()
    ape, irradiance = ape[valid], irradiance[valid]

    # equal time intervals, so energy is proportional to irradiance
    labels = _assign_equal_energy_bins(ape, irradiance, n_bins)

    proxy_spectra = spectra[valid].groupby(labels).mean()

    bins = pd.DataFrame({
        'ape_min': ape.groupby(labels).min(),
        'ape_max': ape.groupby(labels).max(),
        'ape': average_photon_energy(proxy_spectra),
        'n_spectra': labels.value_counts().sort_index(),
        'energy_fraction': irradiance.groupby(labels).sum() / irradiance.sum(),
    })

    labels = labels.reindex(spectra.index)
    return proxy_spectra, bins, labels
