"""
Shaded fraction of a horizontal single-axis tracker
====================================================

This example illustrates how to calculate the 1D shaded fraction of three rows
in a North-South horizontal single axis tracker (HSAT) configuration.
"""

# %%
# :py:func:`pvlib.shading.shaded_fraction1d` exposes a useful method for the
# calculation of the shaded fraction of the width of a solar collector. Here,
# the width is defined as the dimension perpendicular to the axis of rotation.
# This method for calculating the shaded fraction only requires minor
# modifications to be applicable for fixed-tilt systems.
#
# It is highly recommended to read :py:func:`pvlib.shading.shaded_fraction1d`
# documentation to understand the input parameters and the method's
# capabilities. For more in-depth information, please see the journal paper
# `10.1063/5.0202220 <https://doi.org/10.1063/5.0202220>`_ describing
# the methodology.
#
# Let's start by obtaining the true-tracking angles for each of the rows and
# limiting the angles to the range of -50 to 50 degrees. This decision is
# done to create an example scenario with significant shading.
#
# Key functions used in this example are:
#
# 1. :py:func:`pvlib.tracking.singleaxis` to calculate the tracking angles.
# 2. :py:func:`pvlib.shading.projected_solar_zenith_angle` to calculate the
#    projected solar zenith angle.
# 3. :py:func:`pvlib.shading.shaded_fraction1d` to calculate the shaded
#    fractions.
#
# .. sectionauthor:: Echedey Luis <echelual (at) gmail.com>

import pvlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Define the solar system parameters
latitude, longitude = 28.51, -13.89
altitude = pvlib.location.lookup_altitude(latitude, longitude)

axis_tilt = 3  # degrees, positive is upwards in the axis_azimuth direction
axis_azimuth = 180  # degrees, N-S tracking axis
collector_width = 3.2  # m
pitch = 4.15  # m
gcr = collector_width / pitch
cross_axis_slope = -5  # degrees
surface_to_axis_offset = 0.07  # m

# Generate a time range for the simulation
times = pd.date_range(
    start="2024-01-01T05",
    end="2024-01-01T21",
    freq="5min",
    tz="Atlantic/Canary",
)

# Calculate the solar position
solar_position = pvlib.solarposition.get_solarposition(
    times, latitude, longitude, altitude
)
solar_azimuth = solar_position["azimuth"]
solar_zenith = solar_position["apparent_zenith"]

# Calculate the tracking angles
rotation_angle = pvlib.tracking.singleaxis(
    solar_zenith,
    solar_azimuth,
    axis_tilt,
    axis_azimuth,
    max_angle=(-50, 50),  # (min, max) degrees
    backtrack=False,
    gcr=gcr,
    cross_axis_tilt=cross_axis_slope,
)["tracker_theta"]

# %%
# Once the tracker angles have been obtained, the next step is to calculate
# the shaded fraction. Special care must be taken
# to ensure that the shaded or shading tracker roles are correctly assigned
# depending on the solar position.
# This means we will have a result for each row, ``eastmost_shaded_fraction``,
# ``middle_shaded_fraction``, and ``westmost_shaded_fraction``.
# Switching the parameters will be based on the
# sign of :py:func:`pvlib.shading.projected_solar_zenith_angle`.
#
# The following code is verbose to make it easier to understand the process,
# but with some effort you may be able to simplify it. This verbosity also
# allows to change the premises easily per case, e.g., in case of a tracker
# failure or with a different system configuration.

psza = pvlib.shading.projected_solar_zenith_angle(
    solar_zenith, solar_azimuth, axis_tilt, axis_azimuth
)

# Calculate the shaded fraction for the eastmost row
eastmost_shaded_fraction = np.where(
    psza < 0,
    0,  # no shaded fraction in the morning
    # shaded fraction in the evening
    pvlib.shading.shaded_fraction1d(
        solar_zenith,
        solar_azimuth,
        axis_azimuth,
        shaded_row_rotation=rotation_angle,
        axis_tilt=axis_tilt,
        collector_width=collector_width,
        pitch=pitch,
        surface_to_axis_offset=surface_to_axis_offset,
        cross_axis_slope=cross_axis_slope,
        shading_row_rotation=rotation_angle,
    ),
)

# Calculate the shaded fraction for the middle row
middle_shaded_fraction = np.where(
    psza < 0,
    # shaded fraction in the morning
    pvlib.shading.shaded_fraction1d(
        solar_zenith,
        solar_azimuth,
        axis_azimuth,
        shaded_row_rotation=rotation_angle,
        axis_tilt=axis_tilt,
        collector_width=collector_width,
        pitch=pitch,
        surface_to_axis_offset=surface_to_axis_offset,
        cross_axis_slope=cross_axis_slope,
        shading_row_rotation=rotation_angle,
    ),
    # shaded fraction in the evening
    pvlib.shading.shaded_fraction1d(
        solar_zenith,
        solar_azimuth,
        axis_azimuth,
        shaded_row_rotation=rotation_angle,
        axis_tilt=axis_tilt,
        collector_width=collector_width,
        pitch=pitch,
        surface_to_axis_offset=surface_to_axis_offset,
        cross_axis_slope=cross_axis_slope,
        shading_row_rotation=rotation_angle,
    ),
)

# Calculate the shaded fraction for the westmost row
westmost_shaded_fraction = np.where(
    psza < 0,
    # shaded fraction in the morning
    pvlib.shading.shaded_fraction1d(
        solar_zenith,
        solar_azimuth,
        axis_azimuth,
        shaded_row_rotation=rotation_angle,
        axis_tilt=axis_tilt,
        collector_width=collector_width,
        pitch=pitch,
        surface_to_axis_offset=surface_to_axis_offset,
        cross_axis_slope=cross_axis_slope,
        shading_row_rotation=rotation_angle,
    ),
    0,  # no shaded fraction in the evening
)

# %%
# Plot the shaded fraction result for each row:
plt.plot(times, eastmost_shaded_fraction, label="East-most", color="blue")
plt.plot(times, middle_shaded_fraction, label="Middle", color="green",
         linewidth=3, linestyle="--")  # fmt: skip
plt.plot(times, westmost_shaded_fraction, label="West-most", color="red")
plt.title(r"$shaded\_fraction1d$ of each row vs time")
plt.xlabel("Time")
plt.gca().xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.ylabel("Shaded Fraction")
plt.legend()
plt.show()
