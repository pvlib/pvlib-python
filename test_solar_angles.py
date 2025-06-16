import pvlib
import pandas as pd
from datetime import datetime
import pytz

def test_solar_angles():
    # Create a location (example: New York City)
    latitude = 40.7128
    longitude = -74.0060
    tz = 'America/New_York'
    location = pvlib.location.Location(latitude, longitude, tz=tz)
    
    # Create a time range for one day
    start = pd.Timestamp('2024-03-20', tz=tz)  # Spring equinox
    times = pd.date_range(start=start, periods=24, freq='H')
    
    # Calculate solar position
    solpos = location.get_solarposition(times)
    
    # Print results for key times
    print("\nSolar Angles for New York City on Spring Equinox:")
    print("=" * 50)
    
    # Morning (9 AM)
    morning = solpos.loc['2024-03-20 09:00:00-04:00']
    print("\nMorning (9 AM):")
    print(f"Solar Zenith: {morning['zenith']:.2f}°")
    print(f"Solar Azimuth: {morning['azimuth']:.2f}°")
    print(f"Solar Elevation: {morning['elevation']:.2f}°")
    
    # Solar Noon
    noon = solpos.loc['2024-03-20 12:00:00-04:00']
    print("\nSolar Noon:")
    print(f"Solar Zenith: {noon['zenith']:.2f}°")
    print(f"Solar Azimuth: {noon['azimuth']:.2f}°")
    print(f"Solar Elevation: {noon['elevation']:.2f}°")
    
    # Evening (3 PM)
    evening = solpos.loc['2024-03-20 15:00:00-04:00']
    print("\nEvening (3 PM):")
    print(f"Solar Zenith: {evening['zenith']:.2f}°")
    print(f"Solar Azimuth: {evening['azimuth']:.2f}°")
    print(f"Solar Elevation: {evening['elevation']:.2f}°")
    
    # Verify the angles make sense
    print("\nVerification:")
    print("- Zenith angle should be between 0° and 90°")
    print("- Azimuth should be between 0° and 360°")
    print("- Elevation should be between -90° and 90°")
    print("- At solar noon, the sun should be at its highest point")
    print("- The sun should rise in the east (azimuth ~90°) and set in the west (azimuth ~270°)")

if __name__ == "__main__":
    test_solar_angles() 