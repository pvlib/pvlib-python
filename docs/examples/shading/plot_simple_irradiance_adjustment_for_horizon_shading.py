"""
Simple Irradiance Adjustment For Horizon Shading
================================================

Example for adjusting dni and global poa from horizon shading. Using
hard-coded horizon profile data, this example interpolates it to time-series
solar-position data, and adjust DNI and POA-global irradiance.
"""

# %%
# This example shows how to use horizon elevation angles with
# corresponding horizon azimuth angles for simple horizon shading adjustments.

# After location information and a date range is established, solar position
# data is queried from :py:meth:`pvlib.solar_position.get_solar_position`.
# Horizon data is assigned, and interpolated to the solar azimuth time
# series data. Finally, in times when solar elevation is greater than the
# interpolated horizon elevation angle, DNI is set to 0.

import numpy as np
import pandas as pd
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance

# Golden, CO
lat, lon = 39.76, -105.22
tz = 'MST'

# Set times in the morning of the December solstice.
times = pd.date_range(
    '2020-12-20 6:30', '2020-12-20 9:00', freq='1T', tz=tz
)

# Create location object, and get solar position and clearsky irradiance data.
location = Location(lat, lon, tz)
solar_position = location.get_solarposition(times)
clearsky = location.get_clearsky(times)

# Assign variable names for easier reading.
surface_tilt = 30
surface_azimuth = 180
solar_azimuth = solar_position.azimuth
solar_zenith = solar_position.apparent_zenith
solar_elevation = solar_position.apparent_elevation
dni = clearsky.dni
ghi = clearsky.ghi
dhi = clearsky.dhi

# Use hard-coded horizon profile data from location object above.
horizon_profile = pd.Series([
    10.7, 11.8, 11.5, 10.3, 8.0, 6.5, 3.8, 2.3, 2.3, 2.3, 4.6, 8.0, 10.3, 11.1,
    10.7, 10.3, 9.2, 6.1, 5.3, 2.3, 3.1, 1.9, 1.9, 2.7, 3.8, 5.3, 6.5, 8.4,
    8.8, 8.4, 8.4, 8.4, 6.5, 6.1, 6.5, 6.1, 7.3, 9.2, 8.4, 8.0, 5.7, 5.3, 5.3,
    4.2, 4.2, 4.2, 7.3, 9.5
], index=np.arange(0, 360, 7.5))

# Ex - To get the above horizon data from PVGIS (assuming pvlib is imported):
# horizon_profile, horizon_metadata = pvlib.iotools.get_pvgis_horizon(lat, lon)

# Interpolate the horizon elevation data to the solar azimuth, and keep as a
# numpy array.
horizon_elevation_data = np.interp(
    solar_azimuth, horizon_profile.index, horizon_profile
)

# Convert to Pandas Series for easier usage.
horizon_elevation_data = pd.Series(horizon_elevation_data, times)

# Adjust DNI based on data - note this is returned as numpy array
dni_adjusted = np.where(
    solar_elevation > horizon_elevation_data, dni, 0
)

# Adjust GHI and set it to DHI for time-periods where 'dni_adjusted' is 0.
# Note this is returned as numpy array
ghi_adjusted = np.where(
    dni_adjusted == 0, dhi, ghi
)

# Transposition using the original and adjusted irradiance components.
irrad_pre_adj = get_total_irradiance(
    surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, ghi, dhi
)

irrad_post_adj = get_total_irradiance(
    surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni_adjusted,
    ghi_adjusted, dhi
)

# Create and plot result DataFrames.
poa_global_comparison = pd.DataFrame({
    'poa_global_pre-adjustment': irrad_pre_adj.poa_global,
    'poa_global_post-adjustment': irrad_post_adj.poa_global
})

dni_comparison = pd.DataFrame({
    'dni_pre-adjustment': dni,
    'dni_post-adjustment': dni_adjusted
})

# Plot results
poa_global_comparison.plot(
    title='POA-Global: Before and after Horizon Adjustment',
    ylabel='Irradiance'
)
dni_comparison.plot(
    title='DNI: Before and after Horizon Adjustment', ylabel='Irradiance'
)
