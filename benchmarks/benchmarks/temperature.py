"""
ASV benchmarks for irradiance.py
"""

import pandas as pd
import pvlib
from pkg_resources import parse_version
from functools import partial


def set_weather_data(obj):
    obj.times = pd.date_range(start='20180601', freq='1min',
                              periods=14400)
    obj.poa = pd.Series(1000, index=obj.times)
    obj.tamb = pd.Series(20, index=obj.times)
    obj.wind_speed = pd.Series(2, index=obj.times)


class SAPM:

    def setup(self):
        set_weather_data(self)
        if parse_version(pvlib.__version__) >= parse_version('0.7.0'):
            kwargs = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']
            kwargs = kwargs['open_rack_glass_glass']
            self.sapm_cell_wrapper = partial(pvlib.temperature.sapm_cell,
                                             **kwargs)
        else:
            sapm_celltemp = pvlib.pvsystem.sapm_celltemp

            def sapm_cell_wrapper(poa_global, temp_air, wind_speed):
                # just swap order; model params are provided by default
                return sapm_celltemp(poa_global, wind_speed, temp_air)
            self.sapm_cell_wrapper = sapm_cell_wrapper

    def time_sapm_cell(self):
        # use version-appropriate wrapper
        self.sapm_cell_wrapper(self.poa, self.tamb, self.wind_speed)


class Fuentes:

    def setup(self):
        if parse_version(pvlib.__version__) < parse_version('0.8.0'):
            raise NotImplementedError

        set_weather_data(self)

    def time_fuentes(self):
        pvlib.temperature.fuentes(self.poa, self.tamb, self.wind_speed,
                                  noct_installed=45)
