"""
Villalva single-diode model
===========================

This example extracts single-diode model (SDM) parameters from manufacturer
specifications using Villalva's iterative fitting method, validates the fitted
STC I-V curve, recovers a known synthetic SDM parameter set, and calculates
I-V curves at several irradiance and cell-temperature conditions.

The implementation follows Villalva's thesis, Chapter 3 and Appendix A, and
Villalva, Gazoli, and Ruppert Filho (2009).
"""

# %%
# Imports
# -------
# The fitting function is exposed through ``pvlib.ivtools.sdm`` and the
# operating-condition auxiliary equations are exposed through
# ``pvlib.pvsystem``.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import constants

from pvlib import pvsystem
from pvlib.ivtools.sdm import fit_villalva


# %%
# Example module
# --------------
# The JA Solar JAM78D40-625/MB datasheet values are used at STC. The module
# has 156 half-cells, represented here as 78 effective series junctions.
# Villalva's basic fitting method also requires a user-specified dimensionless
# diode ideality factor.

module = {
    "Name": "JA Solar JAM78D40-625/MB",
    "P_mp_ref": 625.0,
    "V_oc_ref": 55.49,
    "I_sc_ref": 14.36,
    "V_mp_ref": 46.37,
    "I_mp_ref": 13.48,
    "cells_in_series": 78,
    "alpha_sc_rel": 0.00046,
    "beta_voc_rel": -0.00260,
    "irrad_ref": 1000.0,
    "temp_ref": 25.0,
}

module["alpha_sc"] = module["I_sc_ref"] * module["alpha_sc_rel"]
module["beta_voc"] = module["V_oc_ref"] * module["beta_voc_rel"]
module["P_mp_fit"] = module["V_mp_ref"] * module["I_mp_ref"]

print(pd.Series(module))


# %%
# Fit the STC parameters
# ----------------------
# The algorithm increments ``R_s`` and selects the valid iteration that
# minimizes the absolute difference between modeled and reference maximum
# power, where the reference target is ``V_mp * I_mp``.

diode_factor = 1.0

params, history = fit_villalva(
    v_mp=module["V_mp_ref"],
    i_mp=module["I_mp_ref"],
    v_oc=module["V_oc_ref"],
    i_sc=module["I_sc_ref"],
    alpha_sc=module["alpha_sc"],
    beta_voc=module["beta_voc"],
    cells_in_series=module["cells_in_series"],
    diode_factor=diode_factor,
    temp_ref=module["temp_ref"],
    irrad_ref=module["irrad_ref"],
    rs_step=1e-4,
    rs_max=0.5,
)

parameter_table = pd.DataFrame(
    {
        "Value": [
            diode_factor,
            params["a_ref"],
            params["I_L_ref"],
            params["I_o_ref"],
            params["R_s"],
            params["R_sh_ref"],
        ],
        "Unit": ["-", "V", "A", "A", "ohm", "ohm"],
    },
    index=[
        "diode_factor",
        "a_ref",
        "I_L_ref",
        "I_o_ref",
        "R_s",
        "R_sh_ref",
    ],
)

print(parameter_table)


# %%
# Evolution of maximum power with series resistance
# -------------------------------------------------
# This plot shows the complete valid fitting history and the selected value of
# ``R_s``.

best_idx = history["abs_power_error"].idxmin()
best = history.loc[best_idx]

print(f"Best R_s = {best['R_s']:.6f} ohm")
print(f"Modeled P_mp = {best['p_mp_model']:.6f} W")
print(f"Reference P_mp = {best['p_mp_ref']:.6f} W")
print(f"Absolute error = {best['abs_power_error']:.6e} W")

plt.figure(figsize=(9, 5))
plt.plot(history["R_s"], history["p_mp_model"], label="Modeled $P_{mp}$")
plt.axhline(
    module["P_mp_fit"],
    linestyle="--",
    label="Reference $V_{mp}I_{mp}$",
)
plt.scatter(
    [best["R_s"]],
    [best["p_mp_model"]],
    zorder=3,
    label=f"Best $R_s$ = {best['R_s']:.4f} $\\Omega$",
)
plt.axvline(best["R_s"], linestyle=":")
plt.xlim(0, 0.26)
plt.xlabel("$R_s$ [$\\Omega$]")
plt.ylabel("$P_{mp}$ [W]")
plt.title("Villalva fitting evolution: $P_{mp}$ vs. $R_s$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %%
# STC I-V curve and datasheet validation
# --------------------------------------
# The fitted parameters are passed directly to pvlib's existing single-diode
# equation solvers. The key modeled points are compared with the datasheet
# values used by the fitting procedure.

stc_sde = {
    "photocurrent": params["I_L_ref"],
    "saturation_current": params["I_o_ref"],
    "resistance_series": params["R_s"],
    "resistance_shunt": params["R_sh_ref"],
    "nNsVth": params["a_ref"],
}

stc = pvsystem.singlediode(method="lambertw", **stc_sde)

datasheet = pd.Series(
    {
        "i_sc": module["I_sc_ref"],
        "v_oc": module["V_oc_ref"],
        "i_mp": module["I_mp_ref"],
        "v_mp": module["V_mp_ref"],
        "p_mp": module["P_mp_fit"],
    }
)

model = pd.Series(
    {
        key: float(np.asarray(stc[key]))
        for key in ["i_sc", "v_oc", "i_mp", "v_mp", "p_mp"]
    }
)

validation = pd.DataFrame({"Datasheet": datasheet, "Villalva": model})
validation["Error"] = validation["Villalva"] - validation["Datasheet"]
validation["Error [%]"] = (
    100 * validation["Error"] / validation["Datasheet"]
)

print(validation)

v_stc = np.linspace(0.0, float(stc["v_oc"]), 200)
i_stc = pvsystem.i_from_v(
    voltage=v_stc,
    method="lambertw",
    **stc_sde,
)

plt.figure(figsize=(9, 6))
plt.plot(v_stc, i_stc, label="Villalva model")
plt.scatter(
    [0.0, module["V_mp_ref"], module["V_oc_ref"]],
    [module["I_sc_ref"], module["I_mp_ref"], 0.0],
    label="Datasheet key points",
    zorder=3,
)
plt.xlabel("Module voltage [V]")
plt.ylabel("Module current [A]")
plt.title(module["Name"] + "\nSTC I-V curve")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %%
# Recover a known SDM parameter set
# ---------------------------------
# A second validation starts from a complete known SDM parameter set. Synthetic
# STC key points are generated with ``pvsystem.singlediode`` and then supplied
# to ``fit_villalva``. This checks how closely the fitting method recovers the
# original parameters when its inputs are internally consistent with the SDE.

known = {
    "Name": "Canadian Solar CS5P-220M",
    "N_s": 96,
    "alpha_sc": 0.004539,
    "beta_voc": -0.22216,
    "a_ref": 2.6373,
    "I_L_ref": 5.114,
    "I_o_ref": 8.196e-10,
    "R_s": 1.065,
    "R_sh_ref": 381.68,
    "temp_ref": 25.0,
    "irrad_ref": 1000.0,
}

T_ref_K = known["temp_ref"] + 273.15

diode_factor_known = known["a_ref"] / (
    known["N_s"] * constants.k * T_ref_K / constants.e
)

synthetic = pvsystem.singlediode(
    photocurrent=known["I_L_ref"],
    saturation_current=known["I_o_ref"],
    resistance_series=known["R_s"],
    resistance_shunt=known["R_sh_ref"],
    nNsVth=known["a_ref"],
    method="lambertw",
)

synthetic_points = {
    key: float(np.asarray(synthetic[key]))
    for key in ["i_sc", "v_oc", "i_mp", "v_mp", "p_mp"]
}

print("Synthetic STC key points")
print(pd.Series(synthetic_points))

recovered, recovery_history = fit_villalva(
    v_mp=synthetic_points["v_mp"],
    i_mp=synthetic_points["i_mp"],
    v_oc=synthetic_points["v_oc"],
    i_sc=synthetic_points["i_sc"],
    alpha_sc=known["alpha_sc"],
    beta_voc=known["beta_voc"],
    cells_in_series=known["N_s"],
    diode_factor=diode_factor_known,
    temp_ref=known["temp_ref"],
    irrad_ref=known["irrad_ref"],
    rs_step=1e-4,
    rs_max=1.5,
)

parameter_names = ["I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "a_ref"]
recovery_table = pd.DataFrame(
    {
        "Starting value": [known[name] for name in parameter_names],
        "Recovered value": [recovered[name] for name in parameter_names],
    },
    index=parameter_names,
)
recovery_table["Absolute error"] = (
    recovery_table["Recovered value"] - recovery_table["Starting value"]
)
recovery_table["Relative error [%]"] = (
    100
    * recovery_table["Absolute error"]
    / recovery_table["Starting value"]
)

print(f"Known Villalva diode ideality factor: {diode_factor_known:.9f}")
print(recovery_table)

v_validation = np.linspace(0.0, synthetic_points["v_oc"], 300)

i_original = pvsystem.i_from_v(
    voltage=v_validation,
    photocurrent=known["I_L_ref"],
    saturation_current=known["I_o_ref"],
    resistance_series=known["R_s"],
    resistance_shunt=known["R_sh_ref"],
    nNsVth=known["a_ref"],
    method="lambertw",
)

i_recovered = pvsystem.i_from_v(
    voltage=v_validation,
    photocurrent=recovered["I_L_ref"],
    saturation_current=recovered["I_o_ref"],
    resistance_series=recovered["R_s"],
    resistance_shunt=recovered["R_sh_ref"],
    nNsVth=recovered["a_ref"],
    method="lambertw",
)

plt.figure(figsize=(9, 6))
plt.plot(v_validation, i_original, label="Starting SDM")
plt.plot(
    v_validation,
    i_recovered,
    linestyle="--",
    label="Recovered Villalva SDM",
)
plt.scatter(
    [0.0, synthetic_points["v_mp"], synthetic_points["v_oc"]],
    [synthetic_points["i_sc"], synthetic_points["i_mp"], 0.0],
    label="Synthetic key points",
    zorder=3,
)
plt.xlabel("Module voltage [V]")
plt.ylabel("Module current [A]")
plt.title("Villalva parameter-recovery validation")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %%
# I-V curves under operating conditions
# -------------------------------------
# The fitted reference parameters are converted to operating-condition SDM
# parameters with ``pvsystem.calcparams_villalva``. pvlib's existing
# ``singlediode`` and ``i_from_v`` functions then solve the corresponding
# I-V curves.

cases = [
    (1000, 55),
    (800, 55),
    (600, 55),
    (400, 25),
    (400, 40),
    (400, 55),
]

conditions = pd.DataFrame(cases, columns=["Geff", "Tcell"])

IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_villalva(
    effective_irradiance=conditions["Geff"],
    temp_cell=conditions["Tcell"],
    alpha_sc=params["alpha_sc"],
    beta_voc=params["beta_voc"],
    a_ref=params["a_ref"],
    I_L_ref=params["I_L_ref"],
    R_sh_ref=params["R_sh_ref"],
    R_s=params["R_s"],
    i_sc_ref=params["i_sc_ref"],
    v_oc_ref=params["v_oc_ref"],
    irrad_ref=params["irrad_ref"],
    temp_ref=params["temp_ref"],
)

SDE_params = {
    "photocurrent": IL,
    "saturation_current": I0,
    "resistance_series": Rs,
    "resistance_shunt": Rsh,
    "nNsVth": nNsVth,
}

curve_info = pvsystem.singlediode(method="lambertw", **SDE_params)

v = pd.DataFrame(np.linspace(0.0, curve_info["v_oc"], 100))
i = pd.DataFrame(
    pvsystem.i_from_v(
        voltage=v,
        method="lambertw",
        **SDE_params,
    )
)

plt.figure(figsize=(9, 6))

for idx, case in conditions.iterrows():
    label = (
        "$G_{eff}$ "
        + f"{case['Geff']:.0f} W/m$^2$\n"
        "$T_{cell}$ "
        + f"{case['Tcell']:.0f} °C"
    )
    plt.plot(v[idx], i[idx], label=label)
    plt.plot(
        [curve_info["v_mp"][idx]],
        [curve_info["i_mp"][idx]],
        linestyle="",
        marker="o",
    )

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel("Module voltage [V]")
plt.ylabel("Module current [A]")
plt.title(module["Name"])
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

operating_results = pd.DataFrame(
    {
        "Geff [W/m2]": conditions["Geff"],
        "Tcell [degC]": conditions["Tcell"],
        "i_sc [A]": curve_info["i_sc"],
        "v_oc [V]": curve_info["v_oc"],
        "i_mp [A]": curve_info["i_mp"],
        "v_mp [V]": curve_info["v_mp"],
        "p_mp [W]": curve_info["p_mp"],
    }
)

print(operating_results)


# %%
# API placement
# -------------
# The proposed public functions are ``pvlib.ivtools.sdm.fit_villalva`` for
# reference-condition parameter extraction and
# ``pvlib.pvsystem.calcparams_villalva`` for the operating-condition
# auxiliary equations. The actual I-V solution remains in pvlib's existing
# ``singlediode`` and ``i_from_v`` functions.
