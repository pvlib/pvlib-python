"""
N. Martin & J. M. Ruiz Spectral Mismatch Modifier
=================================================

How to use this correction factor to adjust the POA global irradiance.
"""

# %%
# Effectiveness of a material to convert incident sunlight to current depends
# on the incident light wavelength. During the day, the spectral distribution
# of the incident irradiance varies from the standard testing spectra,
# introducing a small difference between the expected and the real output.
# In [1]_, N. Martín and J. M. Ruiz propose 3 mismatch factors, one for each
# irradiance component. These mismatch modifiers are calculated with the help
# of the airmass, the clearness index and three experimental fitting
# parameters. In the same paper, these parameters have been obtained for m-Si,
# p-Si and a-Si modules.
# With :py:func:`pvlib.spectrum.martin_ruiz` we are able to make use of these
# already computed values or provide ours.
#
# References
# ----------
#  .. [1] Martín, N. and Ruiz, J.M. (1999), A new method for the spectral
#     characterisation of PV modules. Prog. Photovolt: Res. Appl., 7: 299-310.
#     :doi:`10.1002/(SICI)1099-159X(199907/08)7:4<299::AID-PIP260>3.0.CO;2-0`
#
# Calculating the incident and modified global irradiance
# -------------------------------------------------------
#
# Mismatch modifiers are applied to the irradiance components, so first
# step is to get them. We define an hypothetical POA surface and use TMY to
# compute sky diffuse, ground reflected and direct irradiance.

import matplotlib.pyplot as plt
from pvlib import spectrum, irradiance, iotools, location

surface_tilt = 40
surface_azimuth = 180  # Pointing South
# We will need some location to start with & the TMY
site = location.Location(40.4534, -3.7270, altitude=664,
                         name='IES-UPM, Madrid')

pvgis_data, _, _, _ = iotools.get_pvgis_tmy(site.latitude, site.longitude,
                                            map_variables=True,
                                            startyear=2005, endyear=2015)
# Coerce a year: above function returns typical months of different years
pvgis_data.index = [ts.replace(year=2022) for ts in pvgis_data.index]
# Select days to show
weather_data = pvgis_data['2022-10-03':'2022-10-07']

# Then calculate all we need to get the irradiance components
solar_pos = site.get_solarposition(weather_data.index)

extra_rad = irradiance.get_extra_radiation(weather_data.index)

poa_sky_diffuse = irradiance.haydavies(surface_tilt, surface_azimuth,
                                       weather_data['dhi'],
                                       weather_data['dni'],
                                       extra_rad,
                                       solar_pos['apparent_zenith'],
                                       solar_pos['azimuth'])

poa_ground_diffuse = irradiance.get_ground_diffuse(surface_tilt,
                                                   weather_data['ghi'])

aoi = irradiance.aoi(surface_tilt, surface_azimuth,
                     solar_pos['apparent_zenith'], solar_pos['azimuth'])

# %%
# Let's consider this the irradiance components without spectral modifiers.
# We can calculate the mismatch before and then create a "poa_irrad" var for
# modified components directly, but we want to show the output difference.
# Also, note that :py:func:`pvlib.spectrum.martin_ruiz` result is designed to
# make it easy to multiply each modifier and the irradiance component with a
# single line of code, if you get this dataframe before.

poa_irrad = irradiance.poa_components(aoi, weather_data['dni'],
                                      poa_sky_diffuse, poa_ground_diffuse)

# %%
# Here come the modifiers. Let's calculate them with the airmass and clearness
# index.
# First, let's find the airmass and the clearness index.
# Little caution: default values for this model were fitted obtaining the
# airmass through the `'kasten1966'` method, which is not used by default.

airmass = site.get_airmass(solar_position=solar_pos, model='kasten1966')
clearness_index = irradiance.clearness_index(weather_data['ghi'],
                                             solar_pos['zenith'], extra_rad)

# Get the spectral mismatch modifiers
spectral_modifiers = spectrum.martin_ruiz(clearness_index,
                                          airmass['airmass_absolute'],
                                          module_type='monosi')

# %%
# And then we can find the 3 modified components of the POA irradiance
# by means of a simple multiplication.
# Note, however, that this does not modify ``poa_global`` nor
# ``poa_diffuse``, so we should update the dataframe afterwards.

poa_irrad_modified = poa_irrad * spectral_modifiers

# We want global modified irradiance
poa_irrad_modified['poa_global'] = (poa_irrad_modified['poa_direct']
                                    + poa_irrad_modified['poa_sky_diffuse']
                                    + poa_irrad_modified['poa_ground_diffuse'])
# Don't forget to update `'poa_diffuse'` if you want to use it
# poa_irrad_modified['poa_diffuse'] = \
#     (poa_irrad_modified['poa_sky_diffuse']
#      + poa_irrad_modified['poa_ground_diffuse'])

# %%
# Finally, let's plot the incident vs modified global irradiance, and their
# difference.

poa_irrad_global_diff = (poa_irrad['poa_global']
                         - poa_irrad_modified['poa_global'])
poa_irrad['poa_global'].plot()
poa_irrad_modified['poa_global'].plot()
poa_irrad_global_diff.plot()
plt.legend(['Incident', 'Modified', 'Difference'])
plt.ylabel('POA Global irradiance [W/m²]')
plt.show()
