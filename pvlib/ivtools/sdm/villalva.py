"""Functions for fitting the Villalva single-diode model."""

import numpy as np
import pandas as pd

from scipy import constants

from pvlib import pvsystem


def _villalva_params_at_rs(
    resistance_series,
    v_mp,
    i_mp,
    v_oc,
    i_sc,
    a_ref,
):
    """Calculate ``I_L``, ``I_o`` and ``R_sh`` for one candidate ``R_s``."""
    exp_voc = np.expm1(v_oc / a_ref)
    exp_vmp = np.expm1((v_mp + i_mp * resistance_series) / a_ref)

    p_mp = v_mp * i_mp

    # Algebraic combination of Villalva's equations:
    #
    # I_L = (R_sh + R_s) / R_sh * I_sc
    #
    # I_o = (I_L - V_oc / R_sh) / (exp(V_oc / a_ref) - 1)
    #
    # and the R_sh(R_s) relation obtained at the MPP.
    a_term = (
        v_mp * i_sc
        - v_mp * i_sc * exp_vmp / exp_voc
        - p_mp
    )

    b_term = (
        v_mp * i_sc * resistance_series
        - v_mp
        * (i_sc * resistance_series - v_oc)
        * exp_vmp
        / exp_voc
    )

    resistance_shunt = (
        v_mp * (v_mp + i_mp * resistance_series) - b_term
    ) / a_term

    photocurrent = (
        (resistance_shunt + resistance_series)
        / resistance_shunt
        * i_sc
    )

    saturation_current = (
        photocurrent - v_oc / resistance_shunt
    ) / exp_voc

    return photocurrent, saturation_current, resistance_shunt


def fit_villalva(
    v_mp,
    i_mp,
    v_oc,
    i_sc,
    alpha_sc,
    beta_voc,
    cells_in_series,
    diode_factor,
    temp_ref=25.0,
    irrad_ref=1000.0,
    rs_step=1e-4,
    rs_max=None,
):
    r"""Fit Villalva single-diode model parameters from datasheet values.

    Villalva's basic parameter-extraction method increments the series
    resistance from zero. For each candidate :math:`R_s`, the corresponding
    :math:`R_{sh}`, :math:`I_L`, and :math:`I_0` are calculated and the
    maximum power of the resulting single-diode model is evaluated. The
    selected solution minimizes the absolute difference between modeled and
    reference maximum power.

    The dimensionless diode ideality factor :math:`n` is supplied by the user.
    The modified ideality factor at reference conditions is calculated as

    .. math::

       a_{ref} = n N_s k T_{ref} / q.

    Parameters
    ----------
    v_mp : float
        Maximum-power voltage at reference conditions. [V]
    i_mp : float
        Maximum-power current at reference conditions. [A]
    v_oc : float
        Open-circuit voltage at reference conditions. [V]
    i_sc : float
        Short-circuit current at reference conditions. [A]
    alpha_sc : float
        Short-circuit current temperature coefficient. [A/K]
    beta_voc : float
        Open-circuit voltage temperature coefficient. [V/K]
    cells_in_series : int
        Effective number of cells or junctions connected in series.
    diode_factor : float
        Dimensionless diode ideality factor used by the Villalva model.
    temp_ref : float, default 25
        Reference cell temperature. [°C]
    irrad_ref : float, default 1000
        Reference irradiance. [W/m²]
    rs_step : float, default 1e-4
        Increment used for the series-resistance sweep. [ohm]
    rs_max : float, optional
        Maximum series resistance considered. If ``None``, ``v_oc / i_sc``
        is used. [ohm]

    Returns
    -------
    params : dict
        Fitted reference-condition Villalva parameters. The dictionary
        contains ``I_L_ref``, ``I_o_ref``, ``R_s``, ``R_sh_ref``, ``a_ref``,
        ``alpha_sc``, ``beta_voc``, ``i_sc_ref``, ``v_oc_ref``,
        ``irrad_ref``, and ``temp_ref``.
    history : pandas.DataFrame
        Full fitting history for all valid values of ``R_s`` evaluated by the
        algorithm.

    Raises
    ------
    ValueError
        If the minimum Villalva shunt resistance is not positive.
    RuntimeError
        If no valid Villalva solution is found.

    Notes
    -----
    The returned reference parameters are intended for use with
    :py:func:`pvlib.pvsystem.calcparams_villalva`. I-V curve points can then
    be calculated with :py:func:`pvlib.pvsystem.singlediode` or
    :py:func:`pvlib.pvsystem.i_from_v`.

    References
    ----------
    .. [1] M. G. Villalva, "Three-phase electronic power converter for a grid-connected photovoltaic system," 
       PhD Thesis, Unicamp, 2010. :doi:`10.47749/T/UNICAMP.2010.781324`
    .. [2] M. G. Villalva, J. R. Gazoli, and E. Ruppert Filho, 
       "Comprehensive Approach to Modeling and Simulation of Photovoltaic Arrays," 
       IEEE Transactions on Power Electronics, 2009. :doi:`10.1109/TPEL.2009.2013862`
    """
    temp_ref_k = temp_ref + 273.15

    # a_ref = a * Ns * kT/q
    a_ref = (
        diode_factor
        * cells_in_series
        * constants.k
        * temp_ref_k
        / constants.e
    )

    p_mp_ref = v_mp * i_mp

    # Villalva minimum shunt resistance.
    r_sh_min = v_mp / (i_sc - i_mp) - (v_oc - v_mp) / i_mp

    if r_sh_min <= 0:
        raise ValueError("Villalva R_sh_min must be positive.")

    if rs_max is None:
        rs_max = v_oc / i_sc

    rows = []
    rs_values = np.arange(0.0, rs_max + 0.5 * rs_step, rs_step)

    for resistance_series in rs_values:
        try:
            photocurrent, saturation_current, resistance_shunt = (
                _villalva_params_at_rs(
                    resistance_series,
                    v_mp,
                    i_mp,
                    v_oc,
                    i_sc,
                    a_ref,
                )
            )
        except (FloatingPointError, ZeroDivisionError):
            continue

        if (
            not np.isfinite(resistance_shunt)
            or resistance_shunt < r_sh_min
            or photocurrent <= 0
            or saturation_current <= 0
        ):
            continue

        try:
            mpp = pvsystem.max_power_point(
                photocurrent=photocurrent,
                saturation_current=saturation_current,
                resistance_series=resistance_series,
                resistance_shunt=resistance_shunt,
                nNsVth=a_ref,
                method="brentq",
            )
        except (ValueError, RuntimeError):
            continue

        p_mp_model = float(np.asarray(mpp["p_mp"]))
        power_error = p_mp_model - p_mp_ref

        rows.append(
            {
                "R_s": resistance_series,
                "R_sh_ref": resistance_shunt,
                "I_L_ref": photocurrent,
                "I_o_ref": saturation_current,
                "v_mp_model": float(np.asarray(mpp["v_mp"])),
                "i_mp_model": float(np.asarray(mpp["i_mp"])),
                "p_mp_model": p_mp_model,
                "p_mp_ref": p_mp_ref,
                "power_error": power_error,
                "abs_power_error": abs(power_error),
            }
        )

    if not rows:
        raise RuntimeError("No valid Villalva solution was found.")

    history = pd.DataFrame(rows)
    best = history.loc[history["abs_power_error"].idxmin()]

    params = {
        "I_L_ref": float(best["I_L_ref"]),
        "I_o_ref": float(best["I_o_ref"]),
        "R_s": float(best["R_s"]),
        "R_sh_ref": float(best["R_sh_ref"]),
        "a_ref": float(a_ref),
        "alpha_sc": alpha_sc,
        "beta_voc": beta_voc,
        "i_sc_ref": i_sc,
        "v_oc_ref": v_oc,
        "irrad_ref": irrad_ref,
        "temp_ref": temp_ref,
    }

    return params, history
