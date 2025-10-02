"""
Batzelis's method for estimating single-diode model parameters from
datasheet values.
"""

import numpy as np
from scipy.special import lambertw


def fit_desoto_batzelis(isc0, voc0, imp0, vmp0, alpha_sc, beta_voc):
    """
    Determine De Soto single-diode model parameters from datasheet values
    using Batzelis's method.

    This method is described in Section II.C of [1]_.

    Parameters
    ----------
    isc0 : float
        Short-circuit current at STC. [A]
    voc0 : float
        Open-circuit voltage at STC. [V]
    imp0 : float
        Maximum power point current at STC. [A]
    vmp0 : float
        Maximum power point voltage at STC. [V]
    alpha_sc : float
        Short-circuit current temperature coefficient at STC. [1/K]
    beta_voc : float
        Open-circuit voltage temperature coefficient at STC. [1/K]    

    Returns
    -------
    dict
        The returned dict contains the keys:

        * ``alpha_sc`` [A/K]
        * ``a_ref`` [V]
        * ``I_L_ref`` [A]
        * ``I_o_ref`` [A]
        * ``R_sh_ref`` [Ohm]
        * ``R_s`` [Ohm]

    References
    ----------
    .. [1] E. I. Batzelis, "Simple PV Performance Equations Theoretically Well
       Founded on the Single-Diode Model," Journal of Photovoltaics vol. 7, 
       no. 5, pp. 1400-1409, Sep 2017, :doi:`10.1109/JPHOTOV.2017.2711431`
    """
    t0 = 298.15  # K
    del0 = (1 - beta_voc * t0) / (50.1 - alpha_sc * t0)  # Eq 9
    w0 = lambertw(np.exp(1/del0 + 1)).real

    # Eqs 11-15
    a0 = del0 * voc0
    Rs0 = (a0 * (w0 - 1) - vmp0) / imp0
    Rsh0 = a0 * (w0 - 1) / (isc0 * (1 - 1/w0) - imp0)
    Iph0 = (1 + Rs0 / Rsh0) * isc0
    Isat0 = Iph0 * np.exp(-1/del0)

    return {
        'alpha_sc': alpha_sc * isc0,  # convert 1/K to A/K
        'a_ref': a0,
        'I_L_ref': Iph0,
        'I_o_ref': Isat0,
        'R_sh_ref': Rsh0,
        'R_s': Rs0,
    }
