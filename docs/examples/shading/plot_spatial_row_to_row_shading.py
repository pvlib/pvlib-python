"""
Spatial shading of a row-to-row shading system
==============================================
"""

# %%
# This example demonstrates how to calculate the shading between two rows of
# panels in a row-to-row shading system. The example will show how to calculate
# the shaded fraction in 3D and 2D space with the help of the
# :py:mod:`~pvlib.spatial` module and its classes.
#
# This is a basic example on how to calculate and plot the shaded fraction
# for an instantaneous time. A more complex task is to calculate the shadows
# for a time range. This can be done by iterating over the time range and
# calculating the shadows for each time step. This is done since the
# :py:class:`~pvlib.spatial.FlatSurface` does not support the calculation of
# the shaded fraction for a time range.
# The example :ref:`sphx_glr_gallery_plot_par_direct_shading0_fixed_tilt.py`
# shows how to calculate the shading for a time range for a fixed tilt system.

from pvlib.spatial import RectangularSurface
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import shapely

# %%
# Description of the system
# -------------------------
# Let's start by creating a row-to-row system. We will create a
# rectangular surface for each of the two rows of panels. Both rows will be
# parallel to each other and will have the same azimuth angle. The tilt angle
# of the first row will be 20 degrees, and the tilt angle of the second row
# will be 30 degrees. The distance between the two rows will be 3 meters.
# Also, we will assume scalar values for the solar azimuth and zenith angles.
# Feel free to download the example and change the values to see how the
# shades change.

solar_azimuth = 165  # degrees
solar_zenith = 75  # degrees

row1 = RectangularSurface(  # south-most row
    center=[0, 0, 3], azimuth=165, tilt=20, axis_tilt=10, width=2, length=20
)

row2 = RectangularSurface(  # north-most row
    center=[0, 3, 3], azimuth=165, tilt=20, axis_tilt=10, width=2, length=20
)

# %%
# Calculating the shadows
# -----------------------
# The 3D shapely polygons representing the shadows can be calculated with the
# :py:meth:`~pvlib.spatial.RectangularSurface.get_3D_shades_from` method.
# The 2D shapely polygons representing the shadows can be calculated with the
# :py:meth:`~pvlib.spatial.RectangularSurface.get_2D_shades_from` method. This
# method uses either the 3D shadows or calculates them internally if not
# provided. If the 3D shadows are needed outside its scope, it is recommended
# to calculate them separately and pass them as an argument for performance
# reasons.

shades_3d = row2.get_3D_shades_from(solar_zenith, solar_azimuth, row1)
shades_2d = row2.get_2D_shades_from(
    solar_zenith, solar_azimuth, shades_3d=shades_3d
)

# %%
# Scene and shades plot
# ---------------------

row_style = {"color": "darkblue", "alpha": 0.5}
shade_style = {"color": "dimgrey", "alpha": 0.8}
row_style_2d = {**row_style, "add_points": False}
shade_style_2d = {**shade_style, "add_points": False}

fig = plt.figure(figsize=(10, 10))

# Split the figure in two axes
gs = fig.add_gridspec(10, 1)
ax1 = fig.add_subplot(gs[0:7, 0], projection="3d")
ax2 = fig.add_subplot(gs[8:, 0])

# 3D plot
ax1.view_init(
    elev=60,
    azim=-30,  # matplotlib's azimuth is right-handed to Z+, measured from X+
)
row1.plot(ax=ax1, **row_style)
row2.plot(ax=ax1, **row_style)
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

# 2D plot
row2_2d = row2.representation_in_2D_space()
shapely.plotting.plot_polygon(row2_2d, ax=ax2, **row_style_2d)
for shade in shades_2d.geoms:
    shapely.plotting.plot_polygon(shade, ax=ax2, **shade_style_2d)

# %%
# Calculate the shaded fraction
# -----------------------------
# The shaded fraction can be calculated by dividing the sum of the areas of the
# shadows by the area of the surface. The shaded fraction is a scalar value.

shaded_fraction = sum(shade.area for shade in shades_2d.geoms) / row2_2d.area
print(f"The shaded fraction is {shaded_fraction:.2f}")
