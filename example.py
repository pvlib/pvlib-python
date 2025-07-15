# Simple pvlib demonstration script
import pvlib
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Create a location object for a specific site
location = pvlib.location.Location(
    latitude=40.0,  # New York City latitude
    longitude=-74.0,  # New York City longitude
    tz='America/New_York',
    altitude=10  # meters above sea level
)

# Calculate solar position for a day
date = datetime(2024, 3, 15)
times = pd.date_range(date, date + timedelta(days=1), freq='1H', tz=location.tz)
solpos = location.get_solarposition(times)

# Plot solar position
plt.figure(figsize=(10, 6))
plt.plot(solpos.index, solpos['elevation'], label='Elevation')
plt.plot(solpos.index, solpos['azimuth'], label='Azimuth')
plt.title('Solar Position for New York City on March 15, 2024')
plt.xlabel('Time')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate clear sky irradiance
clearsky = location.get_clearsky(times)

# Plot clear sky irradiance
plt.figure(figsize=(10, 6))
plt.plot(clearsky.index, clearsky['ghi'], label='Global Horizontal Irradiance')
plt.plot(clearsky.index, clearsky['dni'], label='Direct Normal Irradiance')
plt.plot(clearsky.index, clearsky['dhi'], label='Diffuse Horizontal Irradiance')
plt.title('Clear Sky Irradiance for New York City on March 15, 2024')
plt.xlabel('Time')
plt.ylabel('Irradiance (W/m²)')
plt.legend()
plt.grid(True)
plt.show()

# Print some basic information
print("\nSolar Position at Solar Noon:")
noon_idx = solpos['elevation'].idxmax()
print(f"Time: {noon_idx}")
print(f"Elevation: {solpos.loc[noon_idx, 'elevation']:.2f}°")
print(f"Azimuth: {solpos.loc[noon_idx, 'azimuth']:.2f}°")

print("\nMaximum Clear Sky Irradiance:")
print(f"GHI: {clearsky['ghi'].max():.2f} W/m²")
print(f"DNI: {clearsky['dni'].max():.2f} W/m²")
print(f"DHI: {clearsky['dhi'].max():.2f} W/m²") 