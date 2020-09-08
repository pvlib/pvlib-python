"""
ASV benchmarks for location.py
"""

import pandas as pd
import pvlib
from pkg_resources import parse_version


def set_solar_position(obj):
    obj.location = pvlib.location.Location(32, -110, altitude=700,
                                           tz='Etc/GMT+7')
    obj.times = pd.date_range(start='20180601', freq='3min',
                              periods=1440)
    obj.days = pd.date_range(start='20180101', freq='d', periods=365,
                             tz=obj.location.tz)
    obj.solar_position = obj.location.get_solarposition(obj.times)


class Location:

    def setup(self):
        set_solar_position(self)

    # GH 502
    def time_location_get_airmass(self):
        self.location.get_airmass(solar_position=self.solar_position)

    def time_location_get_solarposition(self):
        self.location.get_solarposition(times=self.times)

    def time_location_get_clearsky(self):
        self.location.get_clearsky(times=self.times,
                                   solar_position=self.solar_position)


class Location_0_6_1:

    def setup(self):
        if parse_version(pvlib.__version__) < parse_version('0.6.1'):
            raise NotImplementedError

        set_solar_position(self)

    def time_location_get_sun_rise_set_transit_pyephem(self):
        self.location.get_sun_rise_set_transit(times=self.days,
                                               method='pyephem')

    def time_location_get_sun_rise_set_transit_spa(self):
        self.location.get_sun_rise_set_transit(times=self.days,
                                               method='spa')
