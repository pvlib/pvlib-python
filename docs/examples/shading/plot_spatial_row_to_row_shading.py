"""
Spatial shading of a row-to-row shading system
==============================================
"""

# %%
# This example demonstrates how to calculate the shading between two rows of
# panels in a row-to-row shading system. The example will show how to calculate
# the shading fraction in 3D and 2D space with the help of the
# :py:mod:`~pvlib.spatial` module and its classes.
#
# First section of the example will show how to calculate the shading fraction
# for an instantaneous time. The second section will show how to calculate the
# shading fraction for a time range.

from pvlib.spatial import RectangularSurface
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import shapely.plotting

# %%
# Let's start by creating a row-to-row system. We will create a
# rectangular surface for each of the two rows of panels. Both rows will be
# parallel to each other and will have the same azimuth angle. The tilt angle
# of the first row will be 20 degrees, and the tilt angle of the second row
# will be 30 degrees. The distance between the two rows will be 3 meters.

solar_azimuth = 180  # degrees
solar_zenith = 75  # degrees

# Create the first row of panels
row1 = RectangularSurface(  # south-most row
    center=[0, 0, 3], azimuth=165, tilt=20, axis_tilt=10, width=2, length=20
)

# Create the second row of panels
row2 = RectangularSurface(  # north-most row
    center=[0, 3, 3], azimuth=165, tilt=20, axis_tilt=10, width=2, length=20
)

# Calculate the shadow
shades_3d = row2.get_3D_shades_from(solar_zenith, solar_azimuth, row1)
shades_2d = row2.get_2D_shades_from(solar_zenith, solar_azimuth, row1)

# %%
# Plot the scene and the shadow
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

row_style = {"color": "darkblue", "alpha": 0.5}
shade_style = {"color": "dimgrey", "alpha": 0.8}

fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(10, 1)

ax1 = fig.add_subplot(gs[0:6, 0], projection="3d")
ax2 = fig.add_subplot(gs[8:, 0])

ax1.view_init(elev=45, azim=-45)
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

row_style_2d = {**row_style, "add_points": False}
row2_2d = row2.representation_in_2D_space()
print(f"{row2_2d=}")
shapely.plotting.plot_polygon(row2_2d, ax=ax2, **row_style_2d)
shade_style_2d = {**shade_style, "add_points": False}
for shade in shades_2d.geoms:
    shapely.plotting.plot_polygon(shade, ax=ax2, **shade_style_2d)
