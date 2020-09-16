"""ASV benchmarks for solarposition.py using numba.

We use a separate module so that we can control the pvlib import process
using an environment variable. This will force pvlib to compile the numba
code during setup.

Try to keep relevant sections in sync with benchmarks/solarposition.py
"""

from pkg_resources import parse_version
import pandas as pd

import os
os.environ['PVLIB_USE_NUMBA'] = '1'


import pvlib  # NOQA: E402
from pvlib import solarposition  # NOQA: E402


if parse_version(pvlib.__version__) >= parse_version('0.6.1'):
    sun_rise_set_transit_spa = solarposition.sun_rise_set_transit_spa
else:
    sun_rise_set_transit_spa = solarposition.get_sun_rise_set_transit


class SolarPositionNumba:
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

    def time_spa_python(self, ndays):
        solarposition.spa_python(
            self.times_localized, self.lat, self.lon, how='numba')

    def time_sun_rise_set_transit_spa(self, ndays):
        sun_rise_set_transit_spa(
            self.times_daily, self.lat, self.lon, how='numba')
