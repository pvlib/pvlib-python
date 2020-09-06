"""
ASV benchmarks for irradiance.py
"""

import pandas as pd
import pvlib
from pkg_resources import parse_version
import functools


def set_weather_data(obj):
    obj.times = pd.date_range(start='20180601', freq='1min',
                              periods=14400)
    obj.poa = pd.Series(1000, index=obj.times)
    obj.tamb = pd.Series(20, index=obj.times)
    obj.windspeed = pd.Series(2, index=obj.times)


if parse_version(pvlib.__version__) >= parse_version('0.7.0'):
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']
    params = params['open_rack_glass_glass']
    sapm_cell = functools.partial(pvlib.temperature.sapm_cell, **params)
else:
    sapm_celltemp = pvlib.pvsystem.sapm_celltemp

    def sapm_cell(poa_global, temp_air, wind_speed):
        return sapm_celltemp(poa_global, wind_speed, temp_air)


class SAPM:

    def setup(self):
        set_weather_data(self)

    def time_sapm_cell(self):
        # use version-appropriate wrapper
        sapm_cell(self.poa, self.tamb, self.windspeed)


class Fuentes:

    def setup(self):
        if parse_version(pvlib.__version__) < parse_version('0.8.0'):
            raise NotImplementedError

        set_weather_data(self)

    def time_fuentes(self):
        pvlib.temperature.fuentes(self.poa, self.tamb, self.wind_speed,
                                  noct_installed=45)
