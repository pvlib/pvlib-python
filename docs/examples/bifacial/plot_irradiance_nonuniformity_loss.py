"""
Plot Irradiance Non-uniformity Loss
===================================

Calculate the DC power lost to irradiance non-uniformity in a bifacial PV
array.
"""

# %%
# The incident irradiance on the backside of a bifacial PV module is
# not uniform due to neighboring rows, the ground albedo, and site conditions.
# When each cell works at different irradiance levels, the power produced by
# the module is less than the sum of the power produced by each cell since the
# maximum power point (MPP) of each cell is different, but cells connected in
# series will operate at the same current.
# This is known as irradiance non-uniformity loss.
#
# Calculating the IV curve of each cell and then matching the working point of
# the whole module is computationally expensive, so a simple model to account
# for this loss is of interest. Deline et al. [1]_ proposed a model based on
# the Relative Mean Absolute Difference (RMAD) of the irradiance of each cell.
# They considered the standard deviation of the cells' irradiances, but they
# found that the RMAD was a better predictor of the mismatch loss.
#
# This example demonstrates how to model the irradiance non-uniformity loss
# from the irradiance levels of each cell in a PV module.
#
# The function
# :py:func:`pvlib.bifacial.power_mismatch_deline` is
# used to transform the Relative Mean Absolute Difference (RMAD) of the
# irradiance into a power loss mismatch. Down below you will find a
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

# %%
# Problem description
# -------------------
# Let's set a fixed irradiance to each cell row of the PV array with the values
# described in Figure 1 (A), [1]_. We will cover this case for educational
# purposes, although it can be achieved with the packages
# `solarfactors <https://github.com/pvlib/solarfactors/>`_ and
# `bifacial_radiance <https://github.com/NREL/bifacial_radiance>`_.
#
# Here we set and plot the global irradiance level of each cell.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pvlib.bifacial import power_mismatch_deline

x = np.arange(12, 0, -1)
y = np.arange(6, 0, -1)
cells_irrad = np.repeat([1059, 976, 967, 986, 1034, 1128], len(x)).reshape(
    len(y), len(x)
)

# plot the irradiance levels of each cell
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
# irradiances with the following function, Eq. (4) of [1]_:
#
# .. math::
#
#    \Delta \left[ unitless \right] = \frac{1}{n^2 \bar{G}_{total}}
#    \sum_{i=1}^{n} \sum_{j=1}^{n} \lvert G_{total,i} - G_{total,j} \rvert
#


def rmad(data):
    """
    Relative Mean Absolute Difference. Output is [Unitless]. Eq. (4) of [1]_.
    """
    mean = np.mean(data)
    mad = np.mean(np.absolute(np.subtract.outer(data, data)))
    return mad / mean


rmad_cells = rmad(cells_irrad)

# this is the same as a column's RMAD!
print(rmad_cells == rmad(cells_irrad[:, 0]))

# %%
# Mismatch Loss
# ^^^^^^^^^^^^^
# Calculate the power loss ratio due to the irradiance non-uniformity
# with :py:func:`pvlib.bifacial.power_mismatch_deline`.

mismatch_loss = power_mismatch_deline(rmad_cells)

print(f"RMAD of the cells' irradiance: {rmad_cells:.3} [unitless]")
print(
    "Power loss due to the irradiance non-uniformity: "
    + f"{mismatch_loss:.3} [unitless]"
)
