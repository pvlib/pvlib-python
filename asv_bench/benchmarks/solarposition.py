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
        self.times = pd.DatetimeIndex(start='20180601', freq='1min',
                                      periods=14400)
        self.times_localized = self.times.tz_localize('Etc/GMT+7')

    # GH 512
    def time_ephemeris(self):
        pvlib.solarposition.ephemeris(self.times, 35.1, -106.6)

    # GH 512
    def time_ephemeris_localized(self):
        pvlib.solarposition.ephemeris(self.times_localized, 35.1, -106.6)
