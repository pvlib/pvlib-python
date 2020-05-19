"""
Calculating power loss from partial module shading
==================================================

Example of modeling cell-to-cell mismatch loss from partial module shading.
"""

# %%
# Even though the PV cell is the primary power generation unit, PV modeling is
# often done at the module level for simplicity because module-level parameters
# are much more available and it significantly reduces the computational scope
# of the simulation.  However, module-level simulations are too coarse to be
# able to model effects like cell to cell mismatch or partial shading.  This
# example calculates cell-level IV curves and combines them to reconstruct
# the module-level IV curve.  It uses this approach to find the maximum power
# under various shading and irradiance conditions.
#
# The primary functions used here are:
#
# - :py:meth:`pvlib.pvsystem.calcparams_desoto` to estimate the SDE parameters
#   at the specified operating conditions.
# - :py:meth:`pvlib.singlediode.bishop88` to calculate the full cell IV curve,
#   including the reverse bias region.
#
# .. note::
#
#     This example requires the reverse bias functionality added in pvlib 0.7.2
#
# .. warning::
#
#     Modeling partial module shading is complicated and depends significantly
#     on the module's electrical topology.  This example makes some simplifying
#     assumptions that are not generally applicable.  For instance, it assumes
#     that all of the module's cell strings perform identically, making it
#     possible to ignore the effect of bypass diodes.  It also assumes that
#     shading only applies to beam irradiance, *i.e.* all cells receive the
#     same amount of diffuse irradiance.

from pvlib import pvsystem, singlediode
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

kb = 1.380649e-23  # J/K
qe = 1.602176634e-19  # C
Vth = kb * (273.15+25) / qe

cell_parameters = {
    'I_L_ref': 8.24,
    'I_o_ref': 2.36e-9,
    'a_ref': 1.3*Vth,
    'R_sh_ref': 1000,
    'R_s': 0.00181,
    'alpha_sc': 0.0042,
    'breakdown_factor': 2e-3,
    'breakdown_exp': 3,
    'breakdown_voltage': -15,
}


def simulate_full_curve(parameters, Geff, Tcell, method='brentq',
                        ivcurve_pnts=1000):
    """
    Use De Soto and Bishop to simulate a full IV curve with both
    forward and reverse bias regions.
    """

    # adjust the reference parameters according to the operating
    # conditions using the De Soto model:
    sde_args = pvsystem.calcparams_desoto(
        Geff,
        Tcell,
        alpha_sc=parameters['alpha_sc'],
        a_ref=parameters['a_ref'],
        I_L_ref=parameters['I_L_ref'],
        I_o_ref=parameters['I_o_ref'],
        R_sh_ref=parameters['R_sh_ref'],
        R_s=parameters['R_s'],
    )
    # sde_args has values:
    # (photocurrent, saturation_current, resistance_series,
    # resistance_shunt, nNsVth)

    # Use Bishop's method to calculate points on the IV curve with V ranging
    # from the reverse breakdown voltage to open circuit
    kwargs = {
        'breakdown_factor': parameters['breakdown_factor'],
        'breakdown_exp': parameters['breakdown_exp'],
        'breakdown_voltage': parameters['breakdown_voltage'],
    }
    v_oc = singlediode.bishop88_v_from_i(
        0.0, *sde_args, method=method, **kwargs
    )
    vd = np.linspace(0.99*kwargs['breakdown_voltage'], v_oc, ivcurve_pnts)

    ivcurve_i, ivcurve_v, _ = singlediode.bishop88(vd, *sde_args, **kwargs)
    return pd.DataFrame({
        'i': ivcurve_i,
        'v': ivcurve_v,
    })


def interpolate(df, i):
    """convenience wrapper around scipy.interpolate.interp1d"""
    f_interp = interp1d(np.flipud(df['i']), np.flipud(df['v']), kind='linear',
                        fill_value='extrapolate')
    return f_interp(i)


def combine_series(dfs):
    """
    Combine IV curves in series by aligning currents and summing voltages.
    The current range is based on the first curve's forward bias region.
    """
    df1 = dfs[0]
    imin = df1['i'].min()
    imax = df1.loc[df1['v'] > 0, 'i'].max()
    i = np.linspace(imin, imax, 1000)
    v = 0
    for df2 in dfs:
        v_cell = interpolate(df2, i)
        v += v_cell
    return pd.DataFrame({'i': i, 'v': v})


def simulate_module(cell_parameters, poa_direct, poa_diffuse, Tcell,
                    shaded_fraction, cells_per_string=24, strings=3):
    """
    Simulate the IV curve for a partially shaded module.
    The shade is assumed to be coming up from the bottom of the module when in
    portrait orientation, so it affects all substrings equally.
    Substrings are assumed to be "down and back", so the number of cells per
    string is divided between two columns of cells.
    """
    # find the number of cells per column that are in full shadow
    nrow = cells_per_string//2
    nrow_full_shade = int(shaded_fraction * nrow)
    # find the fraction of shade in the border row
    partial_shade_fraction = 1 - (shaded_fraction * nrow - nrow_full_shade)

    df_lit = simulate_full_curve(
        cell_parameters,
        poa_diffuse + poa_direct,
        Tcell)
    df_partial = simulate_full_curve(
        cell_parameters,
        poa_diffuse + partial_shade_fraction * poa_direct,
        Tcell)
    df_shaded = simulate_full_curve(
        cell_parameters,
        poa_diffuse,
        Tcell)
    # build a list of IV curves for a single column of cells (half a substring)
    include_partial_cell = (shaded_fraction < 1)
    half_substring_curves = (
        [df_lit] * (nrow - nrow_full_shade - 1) +
        ([df_partial] if include_partial_cell else []) +
        [df_shaded] * nrow_full_shade
    )
    df = combine_series(half_substring_curves)
    # all substrings perform equally, so can just scale voltage directly
    df['v'] *= strings*2
    return df


def find_pmp(df):
    """simple function to find Pmp on an IV curve"""
    return df.product(axis=1).max()


# find Pmp under different shading conditions
data = []
for diffuse_fraction in np.linspace(0, 1, 11):
    for shaded_fraction in np.linspace(0, 1, 51):

        df = simulate_module(cell_parameters,
                             poa_direct=(1-diffuse_fraction)*1000,
                             poa_diffuse=diffuse_fraction*1000,
                             Tcell=40,
                             shaded_fraction=shaded_fraction)
        data.append({
            'fd': diffuse_fraction,
            'fs': shaded_fraction,
            'pmp': find_pmp(df)
        })

results = pd.DataFrame(data)
results['pmp'] /= results['pmp'].max()  # normalize power to 0-1
results_pivot = results.pivot('fd', 'fs', 'pmp')
plt.imshow(results_pivot, origin='lower', aspect='auto')
plt.xlabel('shaded fraction')
plt.ylabel('diffuse fraction')
xlabels = ["{:0.02f}".format(fs) for fs in results_pivot.columns[::5]]
ylabels = ["{:0.02f}".format(fd) for fd in results_pivot.index]
plt.xticks(range(0, 5*len(xlabels), 5), xlabels)
plt.yticks(range(0, len(ylabels)), ylabels)
plt.title('Module P_mp across shading conditions')
plt.colorbar()
plt.show()

# %%
#
# This heatmap shows the module maximum power under different partial shade
# conditions, where "diffuse fraction" refers to the ratio
# :math:`poa_{diffuse} / poa_{global}` and "shaded fraction" refers to the
# fraction of the module that receives only diffuse irradiance.
#
# The heatmap makes a few things evident:
#
# - When diffuse fraction is equal to 1, there is no beam irradiance to lose,
#   so shading has no effect on production.
# - When shaded fraction is equal to 0, no irradiance is blocked, so module
#   output does not change with the diffuse fraction.
# - Under sunny conditions (diffuse fraction < 0.5), module output is
#   significantly reduced after just the first cell is shaded
#   (1/12 = ~8% shaded fraction).
