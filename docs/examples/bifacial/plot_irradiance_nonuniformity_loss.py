"""
Plot Irradiance Non-uniformity Loss
===================================

Calculate the incident irradiance lost to non-uniformity in a bifacial PV array
"""

# %%
# The incident irradiance on the backside of a bifacial PV module is
# not uniform due to neighboring rows, the ground albedo and site conditions.
# When each cell works at different irradiance levels, the power produced by
# the module is less than the sum of the power produced by each cell since the
# maximum power point of each cell is different, but cells connected in series
# will operate at the same current. In that case, a deviation is found
# between the MPP and the working point of the cells.
# This is known as irradiance non-uniformity loss.
#
# Calculating the IV curve of each cell and then matching the working point of
# the whole module is computationally expensive, so a model to account for this
# loss is of interest. Deline et al. [1]_ proposed a model based on the
# Relative Mean Absolute Difference (RMAD) of the irradiance of each cell.
# They did also use the standard deviation of the cells' irradiances, but they
# found that the RMAD was a better predictor of the mismatch loss.
#
# This example demonstrates how to model the irradiance non-uniformity loss
# from the irradiance levels of each cell in a PV module.
#
# The function
# :py:func:`pvlib.bifacial.nonuniform_irradiance_deline_power_loss` is
# used to transform the Relative Mean Absolute Difference (RMAD) of the
# irradiance into a power loss percentage. Down below you will find a
# numpy-based implementation of the RMAD function.
#
# References
# ----------
# .. [1] C. Deline, S. Ayala Pelaez, S. MacAlpine, and C. Olalla, 'Estimating
#    and parameterizing mismatch power loss in bifacial photovoltaic
#    systems', Progress in Photovoltaics: Research and Applications, vol. 28,
#    no. 7, pp. 691-703, 2020, :doi:`10.1002/pip.3259`.
#
# .. sectionauthor:: Echedey Luis <echelual (at) gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pvlib.bifacial import nonuniform_irradiance_deline_power_loss

# %%
# Problem description
# -------------------
# Let's set a fixed irradiance to each cell row of the PV array with the values
# described in Figure 1 (A), [1]_. We will cover this case for educational
# purposes, although it can be achieved with the packages
# :ref:`solarfactors <https://github.com/pvlib/solarfactors/>` and
# :ref:`bifacial_radiance <https://github.com/NREL/bifacial_radiance>`.
#
# Here we set and plot the global irradiance level of each cell.

x = np.arange(12, 0, -1)
y = np.arange(6, 0, -1)
cells_irrad = np.repeat([1059, 976, 967, 986, 1034, 1128], len(x)).reshape(
    len(y), len(x)
)

color_map = "gray"
color_norm = Normalize(930, 1150)

fig, ax = plt.subplots()
fig.suptitle("Global Irradiance Levels of Each Cell")
fig.colorbar(
    ScalarMappable(cmap=color_map, norm=color_norm),
    ax=ax,
    orientation="vertical",
    label="$[W/m^2]$",
)
ax.set_aspect("equal")
ax.pcolormesh(
    x,
    y,
    cells_irrad,
    shading="nearest",
    edgecolors="black",
    cmap=color_map,
    norm=color_norm,
)

# %%
# Relative Mean Absolute Difference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Calculate the Relative Mean Absolute Difference (RMAD) of the cells'
# irradiance with the following function, Eq. (4) of [1]_:
#
# .. math::
#
#    \Delta \left[ \% \right] = \frac{1}{n^2 \bar{G}_{total}}
#    \sum_{i=1}^{n} \sum_{j=1}^{n} \lvert G_{total,i} - G_{total,j} \rvert
#


def rmad(data, axis=None):
    """
    Relative Mean Absolute Difference.
    https://stackoverflow.com/a/19472336/19371110
    """
    mad = np.mean(np.absolute(data - np.mean(data, axis)), axis)
    return mad / np.mean(data, axis)


rmad_cells = rmad(cells_irrad)

# this is the same as a column's RMAD!
print(rmad_cells == rmad(cells_irrad[:, 0]))

# %%
# Mismatch Loss
# ^^^^^^^^^^^^^
# Calculate the power loss percentage due to the irradiance non-uniformity
# with the function
# :py:func:`pvlib.bifacial.nonuniform_irradiance_deline_power_loss`.

mismatch_loss = nonuniform_irradiance_deline_power_loss(rmad_cells)

print(f"RMAD of the cells' irradiance: {rmad_cells:.3} [unitless]")
print(f"Power loss due to the irradiance non-uniformity: {mismatch_loss:.3%}")
