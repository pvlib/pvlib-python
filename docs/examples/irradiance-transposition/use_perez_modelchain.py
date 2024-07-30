"""
Use different Perez coefficients with the ModelChain
=================

This example demonstrates how to customize the ModelChain to use site-specifc Perez coefficients.
"""

#%%
# The :py:class:`pvlib.modelchain.ModelChain` object provides a useful interface for easily constructing a model with a simple, unified interface.
# However, a user may want to customize their models in various ways. 
# One such example is during the irradiance transposition step. 
# The Perez models perform very well on field data, but they require a set of fitted coefficients from various sites. 
# It has been noted that these coefficients can be specific to various climates, so users may see a boost in model performance when using the correct set of parameters. 
# However, the base `pvlib.modelchain.ModelChain` only supports the default coefficients. 
# This example shows how the `pvlib.modelchain.ModelChain` can be adjusted to use a different set of Perez coefficients. 

import pandas as pd
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib import iotools, location, irradiance
import pvlib
import os
import matplotlib.pyplot as plt

# Define a system in Florida
lat, lon = 30.1702, -85.65963
# weather_data, weather_meta = get_psm3(lat, lon, interval=30, names=2012,
#                              map_variables=True, leap_day=False)

PVLIB_DIR = pvlib.__path__[0]
DATA_FILE = os.path.join(PVLIB_DIR, 'data', '723170TYA.CSV')

tmy, metadata = iotools.read_tmy3(DATA_FILE, coerce_year=1990,
                                  map_variables=True)

weather_data = pd.DataFrame({'ghi': tmy['ghi'], 'dhi': tmy['dhi'], 'dni': tmy['dni'],
                   'temp_air': tmy['temp_air'],
                   'wind_speed': tmy['wind_speed'],
                   })

loc = location.Location.from_tmy(metadata)

surface_tilt = metadata['latitude']
surface_azimuth = 180

# define some parameters for the module and inverter
sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')

cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

sandia_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

# define the system and ModelChain
system = PVSystem(arrays = None, 
                  surface_tilt=surface_tilt,
                  surface_azimuth=surface_azimuth,
                  module_parameters=sandia_module,
                  inverter_parameters=cec_inverter,
                  temperature_model_parameters=temperature_model_parameters)

mc = ModelChain(system, location=loc,
                aoi_model="no_loss", spectral_model="no_loss")

# since we're in FL, we'll likely want FL coefficients 
model_perez = 'capecanaveral1988'

solar_position = loc.get_solarposition(times=weather_data.index)
dni_extra = irradiance.get_extra_radiation(weather_data.index)

POA_irradiance = irradiance.get_total_irradiance(
    surface_tilt=surface_tilt,
    surface_azimuth=surface_azimuth,
    dni=weather_data['dni'],
    ghi=weather_data['ghi'],
    dhi=weather_data['dhi'],
    solar_zenith=solar_position['apparent_zenith'],
    solar_azimuth=solar_position['azimuth'],
    model='perez',
    dni_extra=dni_extra)

POA_irradiance_new_perez = irradiance.get_total_irradiance(
    surface_tilt=surface_tilt,
    surface_azimuth=surface_azimuth,
    dni=weather_data['dni'],
    ghi=weather_data['ghi'],
    dhi=weather_data['dhi'],
    solar_zenith=solar_position['apparent_zenith'],
    solar_azimuth=solar_position['azimuth'],
    model='perez',
    model_perez=model_perez,
    dni_extra=dni_extra)

mc.run_model_from_poa(POA_irradiance)
ac_power_default = mc.results.ac

mc.run_model_from_poa(POA_irradiance_new_perez)
ac_power_new_perez = mc.results.ac

start, stop = '1990-05-05 06:00:00', '1990-05-05 19:00:00'
plt.plot(ac_power_default.loc[start:stop], 
         label="Default Composite Perez Model")
plt.plot(ac_power_new_perez.loc[start:stop],
         label="Cape Canaveral Perez Model")

plt.xticks(rotation=90)
plt.ylabel("AC Power ($W$)")
plt.legend()