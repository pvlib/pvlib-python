"""
ASV benchmarks for solarposition.py
"""

import pandas as pd
import pvlib
from pvlib import solarposition

from pkg_resources import parse_version


if parse_version(pvlib.__version__) >= parse_version('0.6.1'):
    sun_rise_set_transit_spa = solarposition.sun_rise_set_transit_spa
else:
    sun_rise_set_transit_spa = solarposition.get_sun_rise_set_transit


class SolarPosition:

    def setup(self):
        self.times = pd.date_range(start='20180601', freq='1min',
                                   periods=14400)  # 100 days
        self.times_localized = self.times.tz_localize('Etc/GMT+7')
        self.lat = 35.1
        self.lon = -106.6
        self.times_daily = pd.date_range(
            start='20180601', freq='24h', periods=100, tz='Etc/GMT+7')

    # GH 512
    def time_ephemeris(self):
        solarposition.ephemeris(self.times, self.lat, self.lon)

    # GH 512
    def time_ephemeris_localized(self):
        solarposition.ephemeris(self.times_localized, self.lat, self.lon)

    def time_spa_python(self):
        solarposition.spa_python(self.times_localized, self.lat, self.lon)

    def time_sun_rise_set_transit_spa(self):
        sun_rise_set_transit_spa(self.times_daily, self.lat, self.lon)

    def time_sun_rise_set_transit_ephem(self):
        solarposition.sun_rise_set_transit_ephem(
            self.times_daily, self.lat, self.lon)

    def time_sun_rise_set_transit_ephem_horizon(self):
        solarposition.sun_rise_set_transit_ephem(
            self.times_daily, self.lat, self.lon, horizon='3:00')

    def time_sun_rise_set_transit_geometric(self):
        dayofyear = self.times_daily.dayofyear
        declination = solarposition.declination_spencer71(dayofyear)
        equation_of_time = solarposition.equation_of_time_spencer71(dayofyear)
        solarposition.sun_rise_set_transit_geometric(
            self.times_daily, self.lat, self.lon, declination,
            equation_of_time)

    def time_nrel_earthsun_distance(self):
        solarposition.nrel_earthsun_distance(self.times_localized)
