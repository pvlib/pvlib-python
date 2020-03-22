"""
ASV benchmarks for location.py
"""

import pandas as pd
import pvlib


class TimeSuite:

    def setup(self):
        self.location = pvlib.location.Location(32, -110, altitude=700,
                                                tz='Etc/GMT+7')
        self.times = pd.date_range(start='20180601', freq='3min',
                                   periods=1440)
        self.days = pd.date_range(start='20180101', freq='d', periods=365,
                                  tz=self.location.tz)
        self.solar_position = self.location.get_solarposition(self.times)

    # GH 502
    def time_location_get_airmass(self):
        self.location.get_airmass(solar_position=self.solar_position)

    def time_location_get_solarposition(self):
        self.location.get_solarposition(times=self.times)

    def time_location_get_clearsky(self):
        self.location.get_clearsky(times=self.times,
                                   solar_position=self.solar_position)

    def time_location_get_sun_rise_set_transit_pyephem(self):
        self.location.get_sun_rise_set_transit(times=self.days,
                                               method='pyephem')

    def time_location_get_sun_rise_set_transit_spa(self):
        self.location.get_sun_rise_set_transit(times=self.days,
                                               method='spa')
