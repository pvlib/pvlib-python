"""
Plot Irradiance Non-uniformity Loss
===================================

Calculate the incident irradiance lost to non-uniformity in a bifacial PV array
"""

# %%
# The incident irradiance on the backside of a bifacial PV module is
# not uniform due to neighboring rows, the ground albedo and site conditions.
# Cells with different irradiance levels produce less power that the sum of
# the power produced by each cell individually. This is known as irradiance
# non-uniformity loss.
#
# This example demonstrates how to model the irradiance non-uniformity loss
# due to different global irradiance levels on a bifacial PV module.
#
# The function :py:func:`pvlib.pvsystem.nonuniform_irradiance_loss` will be
# used to transform the Relative Mean Absolute Deviation (RMAD) of the
# irradiance into a power loss percentage.
#
# References
# ----------
# .. [1] Ayala Pelaez, S., Deline, C., MacAlpine, S., & Olalla, C. (2019).
#    Bifacial PV System Mismatch Loss Estimation.
#    https://www.nrel.gov/docs/fy19osti/74831.pdf
#
# .. sectionauthor:: Echedey Luis <echelual (at) gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pvlib.pvsystem import nonuniform_irradiance_loss


# Define the Relative Mean Absolute Deviation (RMAD) function, (Eq. 4) in [1]_.
# https://stackoverflow.com/a/19472336/19371110
def rmad(data, axis=None):
    mad = np.mean(np.absolute(data - np.mean(data, axis)), axis)
    return mad / np.mean(data, axis)


# %%
# Theoretical and straightforward problem
# =======================================
# Let's set a fixed irradiance to each cell row of the PV array with the values
# described in Figure 1 (a), [1]_. We will cover this case for educational
# purposes.
# Here we set and plot the global irradiance levels of each cell.

x = np.arange(6, 0, -1)
y = np.arange(6, 0, -1)
cells_irrad = np.repeat([1059, 976, 967, 986, 1034, 1128], 6).reshape(6, 6)

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
# Relative Mean Absolute Deviation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Calculate the Relative Mean Absolute Deviation (RMAD) of the cells'
# irradiance with the help of the function defined on the top of this example.

rmad_cells = rmad(cells_irrad)

# this is the same as a column's RMAD!
assert rmad_cells == rmad(cells_irrad[:, 0])

# %%
# Mismatch Loss
# ^^^^^^^^^^^^^
# Calculate the power loss percentage due to the irradiance non-uniformity
# with the function :py:func:`pvlib.pvsystem.nonuniform_irradiance_loss`.

mismatch_loss = nonuniform_irradiance_loss(rmad_cells)

# %%
# Total power incident on the module cells
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# It is the sum of the irradiance of each cell

total_irrad = np.sum(cells_irrad)

# %%
# Mismatch-corrected irradiance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The power incident on the module cells is multiplied by the mismatch loss
# as follows:

mismatch_corrected_irrad = total_irrad * (1 - mismatch_loss)

# %%
# Results
# ^^^^^^^

print(f"Total power incident on the module cells: {total_irrad}")
print(f"RMAD of the cells' irradiance: {rmad_cells}")
print(f"Power loss % due to the irradiance non-uniformity: {mismatch_loss}")
print(f"Effective power after mismatch correction: {mismatch_corrected_irrad}")
