"""
Use different Perez coefficients with the ModelChain
====================================================

This example demonstrates how to customize the ModelChain
to use site-specific Perez transposition coefficients.
"""

# %%
# The :py:class:`pvlib.modelchain.ModelChain` object provides a useful method
# for easily constructing a PV system model with a simple, unified interface.
# However, a user may want to customize the steps
# in the system model in various ways.
# One such example is during the irradiance transposition step.
# The Perez model perform very well on field data, but
# it requires a set of fitted coefficients from various sites.
# It has been noted that these coefficients can be specific to
# various climates, so users may see improved model accuracy
# when using a site-specific set of coefficients.
# However, the base  :py:class:`~pvlib.modelchain.ModelChain`
# only supports the default coefficients.
# This example shows how the  :py:class:`~pvlib.modelchain.ModelChain` can
# be adjusted to use a different set of Perez coefficients.

import pandas as pd
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib import iotools, location, irradiance
import pvlib
import os
import matplotlib.pyplot as plt

# load in TMY weather data from North Carolina included with pvlib
PVLIB_DIR = pvlib.__path__[0]
DATA_FILE = os.path.join(PVLIB_DIR, 'data', '723170TYA.CSV')

tmy, metadata = iotools.read_tmy3(DATA_FILE, coerce_year=1990,
                                  map_variables=True)

weather_data = tmy[['ghi', 'dhi', 'dni', 'temp_air', 'wind_speed']]

loc = location.Location.from_tmy(metadata)

#%%
# Now, let's set up a standard PV model using the ``ModelChain``

surface_tilt = metadata['latitude']
surface_azimuth = 180

# define an example module and inverter
sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
sandia_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

# define the system and ModelChain
system = PVSystem(arrays=None,
                  surface_tilt=surface_tilt,
                  surface_azimuth=surface_azimuth,
                  module_parameters=sandia_module,
                  inverter_parameters=cec_inverter,
                  temperature_model_parameters=temp_params)

mc = ModelChain(system, location=loc)

# %%
# Now, let's calculate POA irradiance values outside of the ``ModelChain``.
# We do this for both the default Perez coefficients and the desired
# alternative Perez coefficients.  This enables comparison at the end.

# Cape Canaveral seems like the most likely match for climate
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

# %%
# Now, run the ``ModelChain`` with both sets of irradiance data and compare
# (note that to use POA irradiance as input to the ModelChain the method
# `.run_model_from_poa` is used):

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
plt.tight_layout()
plt.show()
# %%
# Note that there is a small, but noticeable difference from the default
# coefficients that may add up over longer periods of time.
