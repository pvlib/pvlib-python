"""
Modelling power loss due to module shading in non-monolithic Si arrays
======================================================================

This example demonstrates how to model power loss due to row-to-row shading in
a PV array comprised of non-monolithic silicon cells.
"""

# %%
# This example illustrates how to use the work of Martinez et al. [1]_.
# The model is implemented in :py:func:`pvlib.shading.martinez_shade_loss`.
# This model corrects the beam and circumsolar incident irradiance
# based on the number of shaded *blocks*. A *block* is defined as a
# group of cells that are protected by a bypass diode.
# More information on the *blocks* can be found in the original paper [1]_ and
# in :py:func:`pvlib.shading.martinez_shade_loss` documentation.
#
# The following key functions are used in this example:
# 1. :py:func:`pvlib.shading.martinez_shade_loss` to calculate the adjustment
#    factor for the direct irradiance component.
# 2. :py:func:`pvlib.shading.shading_factor1d` to calculate the fraction of
#    shaded surface and consequently the number of shaded *blocks* due to
#    row-to-row shading.
#
# .. sectionauthor:: Echedey Luis <echelual (at) gmail.com>
#
# References
# ----------
# .. [1] F. Martínez-Moreno, J. Muñoz, and E. Lorenzo, 'Experimental model
#    to estimate shading losses on PV arrays', Solar Energy Materials and
#    Solar Cells, vol. 94, no. 12, pp. 2298-2303, Dec. 2010,
#    :doi:`10.1016/j.solmat.2010.07.029`.
#
# Problem description
# -------------------
# Let's consider a PV system with the following characteristics:
# - Two north-south single-axis tracker with 6 modules each one.
# - The rows have the same true-tracking tilt angles.
# - Terrain slope is 7 degrees downward to the east.
# - Row's axis are horizontal.
# - The modules are comprised of silicon cells. We will compare these cases:
#    - modules with one bypass diode
#    - modules with three bypass diodes
#    - half-cut cell modules with three bypass diodes on portrait and landscape
#
# Setting up the system
# ----------------------
# Let's start by defining the location and the time range for the analysis.

import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

pitch = 4  # meters
width = 1.5  # meters
gcr = width / pitch
N_modules_per_row = 6
axis_azimuth = 180  # N-S axis
axis_tilt = 0  # flat because the axis is perpendicular to the slope
cross_axis_tilt = -7  # 7 degrees downward to the east

# Get TMY data & create location
datapath = Path(pvlib.__path__[0], "data", "tmy_45.000_8.000_2005_2016.csv")
pvgis_data, _, metadata, _ = pvlib.iotools.read_pvgis_tmy(
    datapath, map_variables=True
)
locus = pvlib.location.Location(
    metadata["latitude"], metadata["longitude"], altitude=metadata["elevation"]
)

# Coerce a year: function above returns typical months of different years
pvgis_data.index = [ts.replace(year=2024) for ts in pvgis_data.index]
# Select day to show
weather_data = pvgis_data["2024-07-11"]

# %%
# True-tracking algorithm
# -----------------------
# Since this model is about row-to-row shading, we will use the true-tracking
# algorithm to calculate the trackers rotation. Back-tracking avoids shading
# between rows, which is not what we want to analyze here.

# Calculate solar position to get single-axis tracker rotation and irradiance
solar_pos = locus.get_solarposition(weather_data.index)
apparent_zenith, apparent_azimuth = (
    solar_pos["apparent_zenith"],
    solar_pos["azimuth"],
)  # unpack references to data for better readability

tracking_result = pvlib.tracking.singleaxis(
    apparent_zenith=apparent_zenith,
    apparent_azimuth=apparent_azimuth,
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=(-90 + cross_axis_tilt, 90 + cross_axis_tilt),  # (min, max)
    backtrack=False,
    gcr=gcr,
    cross_axis_tilt=cross_axis_tilt,
)

tracker_theta, aoi, surface_tilt, surface_azimuth = (
    tracking_result["tracker_theta"],
    tracking_result["aoi"],
    tracking_result["surface_tilt"],
    tracking_result["surface_azimuth"],
)  # unpack references into explicit variables for better readability

extra_rad = pvlib.irradiance.get_extra_radiation(weather_data.index)

poa_sky_diffuse = pvlib.irradiance.haydavies(
    surface_tilt,
    surface_azimuth,
    weather_data["dhi"],
    weather_data["dni"],
    extra_rad,
    apparent_zenith,
    apparent_azimuth,
)

poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(
    surface_tilt, weather_data["ghi"], surface_type="grass"
)

# %%
# Shaded fraction calculation
# ---------------------------
# The next step is to calculate the fraction of shaded surface. This is done
# using :py:func:`pvlib.shading.shading_factor1d`. Using this function is
# straightforward with the amount of information we already have.

shaded_fraction = pvlib.shading.shading_factor1d(
    surface_tilt, surface_azimuth, apparent_zenith, apparent_azimuth
)
