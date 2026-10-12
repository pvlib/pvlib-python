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
    Bin spectral irradiance data into a specified number of bins.

    Parameters
    ----------
    spectra : pandas.Series or pandas.DataFrame

        Spectral irradiance, must be positive [Wm⁻²nm⁻¹].
        See :term:`spectra`.

        A single spectrum must be a :py:class:`pandas.Series` with wavelength
        [nm] as the index, while multiple spectra must be rows in a
        :py:class:`pandas.DataFrame` with column headers as wavelength [nm]

    n_bins : int
        The number of bins to reduce the spectral data into.

    Returns
    -------
    binned_spectra : pandas.Series or pandas.DataFrame
        The binned spectral irradiance data. The structure matches the input
        ``spectra``: a single spectrum is returned as a
        :py:class:`pandas.Series`, while multiple spectra are returned as a
        :py:class:`pandas.DataFrame`.
    bins           : DataFrame with ape_min, ape_max, ape (APE of the proxy
                     spectrum), duration [s], energy [Wh m-2], n_spectra
    labels         : Series, the bin number of each input spectrum
                     (users can group temperature etc. by this)

    Notes
    -----
    This function implements the equal-energy spectral binning method of
    Witteck et al. [1]_, following the approach of Garcia et al. [2]_. It
    reduces a large set of spectra (for example, a year of measurements) to a
    small number of representative "proxy" spectra while conserving the total
    incident energy. This lowers the cost of energy yield simulations and
    highlights which spectra contribute most of the energy.

    Each spectrum is described by two quantities. The first is its average
    photon energy (APE, :math:`E_\mathrm{ape}`; see
    :py:func:`~pvlib.spectrum.average_photon_energy`), a device-independent
    measure of spectral shape, where a higher APE indicates a bluer spectrum.
    The second is the energy it contributes,

    .. math::

        E = G \cdot \Delta t,

    where :math:`G = \int \Phi(\lambda) \, d\lambda` is the broadband
    irradiance of the spectrum :math:`\Phi`, and :math:`\Delta t` is the time
    interval the spectrum represents.

    The spectra are sorted by APE and the cumulative energy is partitioned into
    ``n_bins`` bins that each contain an equal share of the total energy.
    Unlike binning by equal APE width or by equal spectrum count, equal-energy
    binning concentrates the bins where most of the energy is delivered, so the
    representation is most accurate for the spectra that matter most to the
    energy yield.

    Each bin :math:`k` groups :math:`n_k` spectra :math:`\Phi_{kl}`, where
    spectrum :math:`l` represents a time :math:`t_{kl}`. The bin is summarised
    by a single proxy spectrum :math:`\Phi_k^\mathrm{p}`, the time-weighted
    mean of its members,

    .. math::

        \Phi_k^\mathrm{p} = \frac{1}{t_k} \sum_{l=1}^{n_k} \Phi_{kl} \, t_{kl},
        \qquad t_k = \sum_{l=1}^{n_k} t_{kl}.

    Because the mean is energy-conserving, the proxy spectra carry the same
    total energy as the original dataset. The APE depends on the wavelength
    range of the input, so all spectra must share a common wavelength grid.

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
