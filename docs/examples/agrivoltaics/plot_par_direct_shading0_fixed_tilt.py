"""
Fixed Tilt System impact on crop shading
========================================

This example shows how to calculate the shading of a crop field by a fixed tilt
system.
"""

# %%
# This is the first of a series of examples that will show how to calculate the
# shading of a crop field by a fixed tilt system, a single-axis tracker, and a
# two-axis tracker. The examples will show how to calculate the shading in 3D
# and 2D space, and how to calculate the shading fraction with the help of the
# :py:mod:`~pvlib.spatial` module and its classes.
#
# Paper Examples
# ==============
#
# +--------------------+----------+-------------+----------+-------+
# | Input parameters   | Vertical | Single-axis | Two-axis | Units |
# +====================+==========+=============+==========+=======+
# | Panel width        |        1 |           1 |        1 |   [m] |
# +--------------------+----------+-------------+----------+-------+
# | Panel length       |        2 |           2 |        2 |   [m] |
# +--------------------+----------+-------------+----------+-------+
# | Number of panels   |       40 |          40 |       40 |   [-] |
# +--------------------+----------+-------------+----------+-------+
# | Total panel area   |       80 |          80 |       80 |  [m²] |
# +--------------------+----------+-------------+----------+-------+
# | Number of rows     |        2 |           2 |        2 |   [-] |
# +--------------------+----------+-------------+----------+-------+
# | Row spacing        |       10 |          10 |       10 |   [m] |
# +--------------------+----------+-------------+----------+-------+
# | Row length         |       20 |          20 |       20 |   [m] |
# +--------------------+----------+-------------+----------+-------+
# | Crop area          |      200 |         200 |      200 |  [m²] |
# +--------------------+----------+-------------+----------+-------+
# | Pitch              |        - |           - |        2 |   [m] |
# +--------------------+----------+-------------+----------+-------+
# | Height             |        0 |           3 |        3 |   [m] |
# +--------------------+----------+-------------+----------+-------+
# | Fixed tilt angle   |       90 |           - |        - |   [°] |
# +--------------------+----------+-------------+----------+-------+
# | Azimuth angle      |        0 |           0 |        0 |   [°] |
# +--------------------+----------+-------------+----------+-------+
# | Maximum tilt angle |        - |          60 |       60 |   [°] |
# +--------------------+----------+-------------+----------+-------+
# | Minimum tilt angle |        - |         -60 |      -60 |   [°] |
# +--------------------+----------+-------------+----------+-------+
#
#

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.dates import DateFormatter
import shapely
try:
    from shapely import Polygon  # shapely >= 2
except ImportError:
    from shapely.geometry import Polygon  # shapely < 2
import pandas as pd
import numpy as np
from functools import partial
from pvlib.spatial import RectangularSurface
from pvlib import solarposition

# Kärrbo Prästgård, Västerås, Sweden
latitude, longitude, altitude = 59.6099, 16.5448, 20  # °N, °E, m

spring_equinox = pd.date_range("2021-03-20", periods=24, freq="H")
summer_solstice = pd.date_range("2021-06-21", periods=24, freq="H")
fall_equinox = pd.date_range("2021-09-22", periods=24, freq="H")
winter_solstice = pd.date_range("2021-12-21", periods=24, freq="H")
dates = (
    spring_equinox.union(summer_solstice)
    .union(fall_equinox)
    .union(winter_solstice)
)
# dates = spring_equinox
solar_position = solarposition.get_solarposition(
    dates, latitude, longitude, altitude
)
solar_zenith = solar_position["apparent_zenith"]
solar_azimuth = solar_position["azimuth"]
N = len(solar_zenith)

# %%
# Fixed Tilt
# ----------
# The fixed tilt system is composed of a crop field and two vertical rows of
# PV panels. Rows are placed at the long sides of the field, with the panels
# facing east and west. The field is 20 m long and 10 m wide, and the panels
# are 2 m wide (height since they are vertical) and 20 m long.

field = RectangularSurface(  # crops surface
    center=[5, 10, 0],
    azimuth=90,
    tilt=0,
    axis_tilt=0,
    width=10,
    length=20,
)
pv_rows = (
    RectangularSurface(  # west-most row (lowest X-coordinate)
        center=[-10 / 2 + 5, 10, 2 / 2],
        azimuth=90,
        tilt=90,
        axis_tilt=0,
        width=2,
        length=20,
    ),
    RectangularSurface(  # east-most row (highest X-coordinate)
        center=[10 / 2 + 5, 10, 2 / 2],
        azimuth=90,
        tilt=90,
        axis_tilt=0,
        width=2,
        length=20,
    ),
)


# %%
# Run the simulation
# ------------------
# The shading fraction is calculated at each instant in time, and the results
# are stored in the `fixed_tilt_shaded_fraction` array. This is done because
# the shading calculation API does not allow for vectorized calculations.

# Allocate space for the shading results
fixed_tilt_shaded_fraction = np.zeros((N,), dtype=float)


# Shades callback
def simulation_and_plot_callback(
    timestamp_index, *, shade_3d_artists, shade_2d_artists, sun_annotation
):
    # Calculate the shades at an specific instant in time
    solar_zenith_instant = solar_zenith.iloc[timestamp_index]
    solar_azimuth_instant = solar_azimuth.iloc[timestamp_index]
    # Update the sun position text
    sun_annotation.set_text(
        f"Sun at γ={solar_zenith_instant:.2f}, β={solar_azimuth_instant:.2f}"
    )
    # skip this instant if the sun is below the horizon
    if solar_zenith_instant < 0:
        fixed_tilt_shaded_fraction[timestamp_index] = 0
        return *shade_3d_artists, *shade_2d_artists
    # Calculate the shades, both in 3D and 2D
    shades_3d = field.get_3D_shades_from(
        solar_zenith_instant, solar_azimuth_instant, *pv_rows
    )
    shades_2d = field.get_2D_shades_from(
        solar_zenith_instant, solar_azimuth_instant, shades_3d=shades_3d
    )
    # Plot the calculated shades
    for index, shade in enumerate(shades_3d.geoms):  # 3D
        if not shade.is_empty:
            shade_3d_artists[index].set_verts(
                [shade.exterior.coords],
                closed=False,  # polygon is already closed
            )
    for index, shade in enumerate(shades_2d.geoms):  # 2D
        if not shade.is_empty:
            shade_2d_artists[index].set_path(
                shapely.plotting._path_from_polygon(shade)
            )

    # Calculate the shaded fraction
    fixed_tilt_shaded_fraction[timestamp_index] = (
        sum(shade.area for shade in shades_2d.geoms) / field2d.area
    )
    return *shade_3d_artists, *shade_2d_artists


# %%
# Plot both the 3D and 2D shades
# ------------------------------

fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(10, 1)

# Plotting styles
field_style = {"color": "forestgreen", "alpha": 0.5}
row_style = {"color": "darkblue", "alpha": 0.5}
shade_style = {"color": "dimgrey", "alpha": 0.8}
field_style_2d = {**field_style, "add_points": False}
shade_style_2d = {**shade_style, "add_points": False}

ax1 = fig.add_subplot(gs[0:6, 0], projection="3d")
ax2 = fig.add_subplot(gs[8:, 0])

# Upper plot, 3D
ax1.axis("equal")
ax1.view_init(
    elev=45,
    azim=-60,  # matplotlib's azimuth is right-handed to Z+, measured from X+
)
ax1.set_xlim(-0, 15)
ax1.set_ylim(0, 20)
ax1.set_zlim(0, 10)
ax1.set_xlabel("West(-) <X> East(+) [m]")
ax1.set_ylabel("South(-) <Y> North(+) [m]")
field.plot(ax=ax1, **field_style)
for pv_row in pv_rows:
    pv_row.plot(ax=ax1, **row_style)

# Lower plot, 2D
field2d = field.representation_in_2D_space()
shapely.plotting.plot_polygon(field2d, ax=ax2, **field_style_2d)

# Add empty shade artists for each shading object, in this case each of the
# PV rows. Artists will be updated in the animation callback later.
shade3d_artists = (
    ax1.add_collection3d(Poly3DCollection([], **shade_style)),
) * len(pv_rows)
shade2d_artists = (
    shapely.plotting.plot_polygon(
        Polygon([[0, 0]] * 4), ax=ax2, **shade_style_2d
    ),
) * len(pv_rows)
sun_text_artist = fig.text(0.5, 0.95, "Sun at γ=--, β=--", ha="center")

ani = animation.FuncAnimation(
    fig,
    partial(
        simulation_and_plot_callback,
        shade_3d_artists=shade3d_artists,
        shade_2d_artists=shade2d_artists,
        sun_annotation=sun_text_artist,
    ),
    frames=np.arange(N),
    interval=200,
    blit=True,
)

# uncomment to run and save animation locally
# ani.save("fixed_tilt_shading.gif", writer="pillow")

plt.show()

# %%
# Shaded Fraction vs. Time
# ------------------------
# Create a handy pandas series to plot the shaded fraction vs. time.

fixed_tilt_shaded_fraction = pd.Series(fixed_tilt_shaded_fraction, index=dates)

fig, axs = plt.subplots(ncols=4, sharey=True, figsize=(20, 5))
fig.suptitle("Shaded Fraction vs. Time")
fig.subplots_adjust(wspace=0)

for ax, day_datetimes, title in zip(
    axs,
    (spring_equinox, summer_solstice, fall_equinox, winter_solstice),
    ("Spring Equinox", "Summer Solstice", "Autumn Equinox", "Winter Solstice"),
):
    fixed_tilt_shaded_fraction[day_datetimes].plot(ax=ax)
    ax.xaxis.set_major_formatter(DateFormatter("%H"))
    ax.set_title(title)
    ax.grid(True)
for ax_a, ax_b in zip(axs[:-1], axs[1:]):
    ax_a.spines.right.set_visible(False)
    ax_b.spines.left.set_visible(False)
    ax_a.tick_params(labelright=False)
    ax_b.tick_params(labelleft=False)

axs[0].set_ylabel("Shaded Fraction [Unitless]")
axs[0].set_ylim(0, 1)

plt.show()


# %%
# References
# ----------
# .. [1] S. Zainali et al., 'Direct and diffuse shading factors modelling
#    for the most representative agrivoltaic system layouts', Applied
#    Energy, vol. 339, p. 120981, Jun. 2023,
#    :doi:`10.1016/j.apenergy.2023.120981`.
# .. [2] Y. Cascone, V. Corrado, and V. Serra, 'Calculation procedure of
#    the shading factor under complex boundary conditions', Solar Energy,
#    vol. 85, no. 10, pp. 2524-2539, Oct. 2011,
#    :doi:`10.1016/j.solener.2011.07.011`.
# .. [3] Kevin S. Anderson, Adam R. Jensen; Shaded fraction and
#    backtracking
#    in single-axis trackers on rolling terrain. J. Renewable Sustainable
#    Energy 1 March 2024; 16 (2): 023504. :doi:`10.1063/5.0202220`.

# %%
