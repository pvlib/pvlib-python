"""
Single-diode equation
=====================

Examples of modeling IV curves using a single-diode circuit equivalent model.
"""

#%%
# Calculating a module IV curve for certain operating conditions is a two-step
# process.  Multiple methods exist for both parts of the process.  Here we use
# the De Soto 5-parameter model to calculate the electrical characteristics
# of a PV module at a certain irradiance and temperature using the module's
# base characteristics at reference conditions.  Those parameters are then used
# to calculate the module's IV curve by solving the single-diode equation using
# the Lambert W method.
#
# The single-diode equation is a circuit-equivalent model of a PV
# cell and has five electrical parameters that depend on the operating
# conditions.  For more details on the single-diode equation and the five
# parameters, see the `PVPMC single diode page
# <https://pvpmc.sandia.gov/modeling-steps/2-dc-module-iv/diode-equivalent-circuit-models/>`_.
#
# Calculating IV Curves
# -----------------------
# This example uses :py:meth:`pvlib.pvsystem.calcparams_desoto` to calculate
# the 5 electrical parameters needed to solve the single-diode equation.
# :py:meth:`pvlib.pvsystem.singlediode` is then used to generate the IV curves.

from pvlib import pvsystem
import pandas as pd
import matplotlib.pyplot as plt

# Example module parameters for the Canadian Solar CS5P-220M:
parameters = {
    'Name': 'Canadian Solar CS5P-220M',
    'BIPV': 'N',
    'Date': '10/5/2009',
    'T_NOCT': 42.4,
    'A_c': 1.7,
    'N_s': 96,
    'I_sc_ref': 5.1,
    'V_oc_ref': 59.4,
    'I_mp_ref': 4.69,
    'V_mp_ref': 46.9,
    'alpha_sc': 0.004539,
    'beta_oc': -0.22216,
    'a_ref': 2.6373,
    'I_L_ref': 5.114,
    'I_o_ref': 8.196e-10,
    'R_s': 1.065,
    'R_sh_ref': 381.68,
    'Adjust': 8.7,
    'gamma_r': -0.476,
    'Version': 'MM106',
    'PTC': 200.1,
    'Technology': 'Mono-c-Si',
}

times = pd.date_range(start='2015-06-01 10:00', periods=3, freq='h')
effective_irradiance = pd.Series([800, 600.0, 400.0], index=times)
temp_cell = pd.Series([60, 40, 20], index=times)

# adjust the reference parameters according to the operating
# conditions using the De Soto model:
IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
    effective_irradiance,
    temp_cell,
    alpha_sc=parameters['alpha_sc'],
    a_ref=parameters['a_ref'],
    I_L_ref=parameters['I_L_ref'],
    I_o_ref=parameters['I_o_ref'],
    R_sh_ref=parameters['R_sh_ref'],
    R_s=parameters['R_s'],
    EgRef=1.121,
    dEgdT=-0.0002677
)

# plug the parameters into the SDE and solve for IV curves:
curve_info = pvsystem.singlediode(
    photocurrent=IL,
    saturation_current=I0,
    resistance_series=Rs,
    resistance_shunt=Rsh,
    nNsVth=nNsVth,
    ivcurve_pnts=100,
    method='lambertw'
)

# plot the calculated curves:
plt.figure()
for i in range(len(times)):
    label = (
        "$G_{eff}=" + f"{effective_irradiance[i]}$ $W/m^2$; "
        "$T_{cell}=" + f"{temp_cell[i]}$ C"
    )
    plt.plot(curve_info['v'][i], curve_info['i'][i], label=label)
    v_mp = curve_info['v_mp'][i]
    i_mp = curve_info['i_mp'][i]
    plt.plot([v_mp], [i_mp], ls='', marker='o', c='k')

plt.legend()
plt.xlabel('Module voltage [V]')
plt.ylabel('Module current [A]')
plt.title(parameters['Name'])
plt.show()

print(pd.DataFrame({
    'i_sc': curve_info['i_sc'],
    'v_oc': curve_info['v_oc'],
    'i_mp': curve_info['i_mp'],
    'v_mp': curve_info['v_mp'],
    'p_mp': curve_info['p_mp'],
}))

#%%
# Interactive Demo
# ----------------
#
# The plot below shows the main IV curve points from sweeping across irradiance
# and temperature.  Change the dropdowns to scatter different variables against
# each other.
#
#  .. bokeh-plot:: ../../examples/interactive_examples/interactive_singlediode.py
