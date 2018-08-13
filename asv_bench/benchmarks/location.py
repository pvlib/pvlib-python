# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import pandas as pd
import pvlib

class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self):
        self.location = pvlib.location.Location(32, -110, altitude=700)
        self.times = pd.DatetimeIndex(start='20180601', freq='3min',
                                      periods=1440)
        self.solar_position = self.location.get_solarposition(self.times)

    # GH 502
    def time_location_get_airmass(self):
        self.location.get_airmass(solar_position=self.solar_position)
