"""
ASV benchmarks for solarposition.py
"""

import datetime
import pandas as pd
import pvlib
from pvlib import solarposition

from pkg_resources import parse_version


if parse_version(pvlib.__version__) >= parse_version('0.6.1'):
    sun_rise_set_transit_spa = solarposition.sun_rise_set_transit_spa
else:
    sun_rise_set_transit_spa = solarposition.get_sun_rise_set_transit


class SolarPosition:
    params = [1, 10, 100]  # number of days
    param_names = ['ndays']

    def setup(self, ndays):
        self.times = pd.date_range(start='20180601', freq='1min',
                                   periods=1440*ndays)
        self.times_localized = self.times.tz_localize('Etc/GMT+7')
        self.lat = 35.1
        self.lon = -106.6
        self.times_daily = pd.date_range(
            start='20180601', freq='24h', periods=ndays, tz='Etc/GMT+7')

    # GH 512
    def time_ephemeris(self, ndays):
        solarposition.ephemeris(self.times, self.lat, self.lon)

    # GH 512
    def time_ephemeris_localized(self, ndays):
        solarposition.ephemeris(self.times_localized, self.lat, self.lon)

    def time_spa_python(self, ndays):
        solarposition.spa_python(self.times_localized, self.lat, self.lon)

    def time_pyephem(self, ndays):
        solarposition.pyephem(self.times_localized, self.lat, self.lon)

    def time_sun_rise_set_transit_spa(self, ndays):
        sun_rise_set_transit_spa(self.times_daily, self.lat, self.lon)

    def time_sun_rise_set_transit_ephem(self, ndays):
        solarposition.sun_rise_set_transit_ephem(
            self.times_daily, self.lat, self.lon)

    def time_sun_rise_set_transit_geometric_full_comparison(self, ndays):
        dayofyear = self.times_daily.dayofyear
        declination = solarposition.declination_spencer71(dayofyear)
        equation_of_time = solarposition.equation_of_time_spencer71(dayofyear)
        solarposition.sun_rise_set_transit_geometric(
            self.times_daily, self.lat, self.lon, declination,
            equation_of_time)

    def time_nrel_earthsun_distance(self, ndays):
        solarposition.nrel_earthsun_distance(self.times_localized)


class SolarPositionCalcTime:

    def setup(self):
        # test calc_time for finding times at which sun is 3 degrees
        # above the horizon.
        # Tucson 2020-09-14 sunrise at 6:08 AM MST, 13:08 UTC
        # according to google.
        self.start = datetime.datetime(2020, 9, 14, 12)
        self.end = datetime.datetime(2020, 9, 14, 15)
        self.value = 0.05235987755982988
        self.lat = 32.2
        self.lon = -110.9
        self.attribute = 'alt'

    def time_calc_time(self):
        # datetime.datetime(2020, 9, 14, 13, 24, 13, 861913, tzinfo=<UTC>)
        solarposition.calc_time(
            self.start, self.end, self.lat, self.lon, self.attribute,
            self.value
        )
