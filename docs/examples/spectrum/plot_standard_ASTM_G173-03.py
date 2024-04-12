"""
ASTM G173-03 Standard Spectrum
==============================

This example demonstrates how to read the data from the ASTM G173-03 standard
spectrum bundled with PVLIB and plot each of the components.

The ASTM G173-03 standard provides reference solar spectral irradiance data.
"""

import matplotlib.pyplot as plt
import pvlib


# %%
# Use :py:func:`pvlib.spectrum.get_ASTM_G173` to retrieve the ASTM G173-03
# standard spectrum.

am15_df = pvlib.spectrum.get_ASTM_G173()

# Plot
plt.plot(am15_df.index, am15_df['extraterrestrial'], label='Extraterrestrial')
plt.plot(am15_df.index, am15_df['global'], label='Global')
plt.plot(am15_df.index, am15_df['direct'], label='Direct')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Irradiance (W/m²/nm)')
plt.title('ASTM G173-03 Solar Spectral Irradiance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
