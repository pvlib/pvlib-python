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
# will operate at the same current. This is known as irradiance non-uniformity
# loss.
#
# Calculating the IV curve of each cell and then matching the working point of
# the whole module is computationally expensive, so a model to account for this
# loss is of interest. Deline et al. [1]_ proposed a model based on the
# Relative Mean Absolute Difference (RMAD) of the irradiance of each cell.
# They did also use the standard deviation of the cells' irradiances, but they
# found that the RMAD was a better predictor of the mismatch loss.
#
# This example demonstrates how to model the irradiance non-uniformity loss
# due to different global irradiance levels on a bifacial PV module through
# two different approaches:
#
# - Given the irradiance levels of each cell in an instant,
#   in the first section.
# - Modelling the irradiance non-uniformity RMAD through the day thanks to a
#   mock-up horizontal axis tracker system, in the second section.
#   See [2]_ and [3]_ for more information.
#
# The function
# :py:func:`pvlib.pvsystem.nonuniform_irradiance_deline_power_loss` is
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
# .. [2] C. Domínguez, J. Marcos, S. Ures, S. Askins, and I. Antón, 'A
#    Horizontal Single-Axis Tracker Mock-Up to Quickly Assess the Influence
#    of Geometrical Factors on Bifacial Energy Gain', in 2023 IEEE 50th
#    Photovoltaic Specialists Conference (PVSC), Jun. 2023, pp. 1–3.
#    :doi:`10.1109/PVSC48320.2023.10359580`.
# .. [3] C. Domínguez, J. Marcos, S. Ures, S. Askins, I. Antón, A Horizontal
#    Single-Axis Tracker Mock-Up to Quickly Assess the Influence of Geometrical
#    Factors on Bifacial Energy Gain. Zenodo, 2023.
#    :doi:`10.5281/zenodo.11125039`.
#
# .. sectionauthor:: Echedey Luis <echelual (at) gmail.com>

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pvlib.pvsystem import nonuniform_irradiance_deline_power_loss

# %%
# Theoretical and straightforward problem
# ---------------------------------------
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
# :py:func:`pvlib.pvsystem.nonuniform_irradiance_deline_power_loss`.

mismatch_loss = nonuniform_irradiance_deline_power_loss(rmad_cells)

print(f"RMAD of the cells' irradiance: {rmad_cells}")
print(f"Power loss % due to the irradiance non-uniformity: {mismatch_loss}")

# %%
# A practical approach
# --------------------
# Instead of simulating or measuring the irradiance of each cell, RMAD trend
# through the day can be modelled. The remaining section of this example will
# cover the modelling of the global RMAD if the backside irradiance RMAD is
# known. Note these two RMADs are inherently different but are related.
#
# Here we will base our calculations on a horizontal single-axis tracking
# mockup developed by Domínguez et al. [2]_.
# The following image in [3]_ shows the backside irradiance non-uniformity of a
# HSAT mock-up system:
#
# .. figure:: ../../_images/Dominguez_et_al_PVSC2023.png
#    :alt: Plot of backside reference cells of an HSAT mock-up and their RMAD.
#    :align: center
#
#    Blue dots represent the backside irradiance non-uniformity.
#    *BE* stands for *backside east*.


def hsat_backside_rmad_model_through_day(hour):  # TODO: REMOVE THIS
    """Model of the blue dots in the image above."""
    # For demonstration purposes only. Model roughly fit to show an example of
    # the RMAD variation through the day without including all the data.
    # fmt: off
    morning_polynom = [6.71787833e-02, -4.50442998e+00, 1.18114757e+02,
                       -1.51725679e+03, 9.56439547e+03, -2.36835920e+04]
    afternoon_polynom = [7.14947943e-01, -6.02541075e+01, 2.02789031e+03,
                         -3.40677727e+04, 2.85671091e+05, -9.56469320e+05]
    # fmt: on
    day_rmad = np.where(
        hour < 14.75,
        np.polyval(morning_polynom, hour),
        np.polyval(afternoon_polynom, hour),
    )
    return day_rmad / 100  # RMAD is a percentage


# %%
# Calculating Global RMAD from Backside RMAD
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. note::
#    The global irradiance RMAD is different from the backside irradiance RMAD.
#
# The global irradiance is the sum of the front irradiance and the backside
# irradiance by the bifaciality factor, see equation (2) in [1]_.
#
# .. math::
#
#    G_{total,i} = G_{front,i} + \phi_{Bifi} G_{rear,i}
#
# where :math:`\phi_{Bifi}` is the bifaciality factor.
#
# Here we will model front and backside irradiances with normal distributions
# for simplicity, then calculate the global RMAD and plot the results.
#
# The backside irradiance is the one that presents the most significant
# non-uniformity. The front irradiance is way more uniform, so it will be
# neglected in this example.
#
# Let's calculate the **global RMAD** through the day - it's **different** from
# the backside RMAD since
#
# .. math::
#
#    RMAD(k \cdot X + c) = RMAD(X) \cdot k \frac{k \bar{X}}{k \bar{X} + c}
#    = RMAD(X) \cdot k \frac{1}{1 + \frac{c}{k \bar{X}}}
#
# where :math:`X` is a random variable and :math:`k>0, c \neq \bar{X}` are
# constants (`source
# <https://en.wikipedia.org/wiki/Mean_absolute_difference#Properties>`_).

times = pd.date_range("2023-06-06T09:30", "2023-06-06T18:30", freq="30min")
hours = times.hour + times.minute / 60
bifaciality = 0.65
front_irrad = scipy.stats.norm.pdf(hours, loc=12.5, scale=1600)
backside_irrad = scipy.stats.norm.pdf(hours, loc=12.5, scale=180)

global_irrad = front_irrad + bifaciality * backside_irrad
# See RMAD properties above
# Here we calculate RMAD(global_irrad)
# backside_irrad := X, bifaciality := k, front_irrad := c
backside_rmad = hsat_backside_rmad_model_through_day(hours)
global_rmad = (
    backside_rmad
    * bifaciality
    / (1 + front_irrad / backside_irrad / bifaciality)
)

# Get the mismatch loss
mismatch_loss = nonuniform_irradiance_deline_power_loss(global_rmad)

# Plot results
fig, ax1 = plt.subplots()
fig.suptitle("Irradiance RMAD and Mismatch Losses")

ax1.plot(hours, global_rmad, label="RMAD: global", color="k")
ax1.plot(
    hours, backside_rmad, label="RMAD: backside", color="b", linestyle="--"
)
ax1.set_xlabel("Hour of the day")
ax1.set_ylabel("RMAD")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(
    hours, mismatch_loss, label="Mismatch loss", color="red", linestyle=":"
)
ax2.grid()
ax2.legend(loc="upper right")
ax2.set_ylabel("Mismatch loss")

fig.show()
