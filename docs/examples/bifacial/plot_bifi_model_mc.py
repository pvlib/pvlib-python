"""
Bifacial Modeling - modelchain
==============================

Example of bifacial modeling using the modelchain
"""

# %%
# This example shows how to complete a bifacial modeling example using the
# :py:class:`pvlib.modelchain.ModelChain` with the
# :py:func:`pvlib.bifacial.pvfactors_timeseries` function to transpose
# GHI data to both front and rear Plane of Array (POA) irradiance.

import pandas as pd
from pvlib import pvsystem
from pvlib import location
from pvlib import modelchain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS as PARAMS
from pvlib import bifacial

# create site location and times characteristics
lat, lon = 36.084, -79.817
tz = 'Etc/GMT+5'
times = pd.date_range('2021-06-21', '2021-6-22', freq='1T', tz=tz)

# create site system characteristics
axis_tilt = 0
axis_azimuth = 180
gcr = 0.35
max_angle = 60
pvrow_height = 3
pvrow_width = 4
albedo = 0.2

# load temperature parameters and module/inverter specifications
temp_model_parameters = PARAMS['sapm']['open_rack_glass_polymer']
sandia_modules = pvsystem.retrieve_sam('SandiaMod')
cec_inverters = pvsystem.retrieve_sam('cecinverter')
sandia_module = sandia_modules['Sanyo_HIP_200DA3_Bifacial__2007__E__']
cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

# create a location for site, and get solar position and clearsky data
site_location = location.Location(lat, lon, tz=tz, name='Greensboro, NC')
solar_position = site_location.get_solarposition(times)
cs = site_location.get_clearsky(times)

# load solar position and tracker orientation for use in pvsystem object
sat_mount = pvsystem.SingleAxisTrackerMount(axis_tilt=axis_tilt,
                                            axis_azimuth=axis_azimuth,
                                            max_angle=max_angle,
                                            backtrack=True,
                                            gcr=gcr)
# dc arrays
array = pvsystem.Array(mount=sat_mount,
                       module_parameters=sandia_module,
                       temperature_model_parameters=temp_model_parameters)

# create system object
system = pvsystem.PVSystem(arrays=[array],
                           inverter_parameters=cec_inverter)

# created for use in pvfactors timeseries
orientation = sat_mount.get_orientation(solar_position['apparent_zenith'],
                                        solar_position['azimuth'])

# get rear and front side irradiance from pvfactors transposition engine
irrad = bifacial.pvfactors_timeseries(solar_position['azimuth'],
                                      solar_position['apparent_zenith'],
                                      orientation['surface_azimuth'],
                                      orientation['surface_tilt'],
                                      axis_azimuth,
                                      times,
                                      cs['dni'],
                                      cs['dhi'],
                                      gcr,
                                      pvrow_height,
                                      pvrow_width,
                                      albedo)

# turn into pandas DataFrame
irrad = pd.concat(irrad, axis=1)

# define bifaciality coefficient (specific to module type being used)
# create bifacial effective irradiance using aoi-corrected timeseries values
bifaciality = 0.75
irrad['effective_irradiance'] = (
    irrad['total_abs_front'] + (irrad['total_abs_back'] * bifaciality)
)

# create modelchain object for bifacial system and run bifacial simulation
mc_bifi = modelchain.ModelChain(system, site_location)
mc_bifi.run_model_from_effective_irradiance(irrad)

# plot results of both monofacial and bifacial
mc_bifi.results.ac.plot(title='Bifacial Simulation on June Solstice',
                        ylabel='AC Power')
