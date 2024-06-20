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
import shapely.plotting
import pandas as pd
from pvlib.spatial import RectangularSurface

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
solar_azimuth = 155
solar_zenith = 70

# %%
# Fixed Tilt
# ----------

field = RectangularSurface(  # crops top surface
    center=[10, 5, 0],
    azimuth=180,  # chosen instead of 0 (north) for intuitive visualization
    tilt=0,
    axis_tilt=0,
    width=10,
    length=20,
)
pv_row1 = RectangularSurface(  # south-most row (lowest Y-coordinate)
    center=[10, -10 / 2 + 5, 2 / 2],
    azimuth=180,
    tilt=90,
    axis_tilt=0,
    width=2,
    length=20,
)
pv_row2 = RectangularSurface(  # north-most row (highest Y-coordinate)
    center=[10, 10 / 2 + 5, 2 / 2],
    azimuth=180,
    tilt=90,
    axis_tilt=0,
    width=2,
    length=20,
)

# %%
shades_3d = field.get_3D_shades_from(
    solar_zenith, solar_azimuth, pv_row1, pv_row2
)
print(shades_3d)
# %%
shades_2d = field.get_2D_shades_from(
    solar_zenith, solar_azimuth, pv_row1, pv_row2
)
print(shades_2d)

# %%
# Plot both the 3D and 2D shades
# ------------------------------

field_style = {"color": "forestgreen", "alpha": 0.5}
row_style = {"color": "darkblue", "alpha": 0.5}
shade_style = {"color": "dimgrey", "alpha": 0.8}

fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(10, 1)

ax1 = fig.add_subplot(gs[0:6, 0], projection="3d")
ax2 = fig.add_subplot(gs[8:, 0])

ax1.view_init(elev=45, azim=-45)
field.plot(ax=ax1, **field_style)
pv_row1.plot(ax=ax1, **row_style)
pv_row2.plot(ax=ax1, **row_style)
for shade in shades_3d.geoms:
    if shade.is_empty:
        continue  # skip empty shades; else an exception will be raised
    # use Matplotlib's Poly3DCollection natively since experimental
    # shapely.plotting.plot_polygon does not support 3D
    vertexes = shade.exterior.coords[:-1]
    ax1.add_collection3d(Poly3DCollection([vertexes], **shade_style))

ax1.axis("equal")
ax1.set_zlim(0)
ax1.set_xlabel("West(-) <X> East(+) [m]")
ax1.set_ylabel("South(-) <Y> North(+) [m]")

field2d = field.representation_in_2D_space()
field_style_2d = {**field_style, "add_points": False}
shapely.plotting.plot_polygon(field2d, ax=ax2, **field_style_2d)
shade_style_2d = {**shade_style, "add_points": False}
for shade in shades_2d.geoms:
    shapely.plotting.plot_polygon(shade, ax=ax2, **shade_style_2d)

# %%
beam_shaded_fraction = (
    sum(shade.area for shade in shades_2d.geoms) / field2d.area
)
print(beam_shaded_fraction)

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
