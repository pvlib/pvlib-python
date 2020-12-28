"""
ASV benchmarks for detect clear sky function.
"""

import pandas as pd
from pvlib import clearsky, solarposition
import numpy as np


class DetectClear:
    params = [1, 10, 100]  # number of days
    param_names = ['ndays']

    def setup(self, ndays):
        self.times = pd.date_range(start='20180601', freq='1min',
                                   periods=1440*ndays)
        self.lat = 35.1
        self.lon = -106.6
        self.solar_position = solarposition.get_solarposition(
            self.times, self.lat, self.lon)
        clearsky_df = clearsky.simplified_solis(
            self.solar_position['apparent_elevation'])
        self.clearsky = clearsky_df['ghi']
        measured_dni = clearsky_df['dni'].where(
            (self.times.hour % 2).astype(bool), 0)
        cos_zen = np.cos(np.deg2rad(self.solar_position['apparent_zenith']))
        self.measured = measured_dni * cos_zen + clearsky_df['dhi']
        self.measured *= 0.98
        self.window_length = 10

    def time_detect_clearsky(self, ndays):
        clearsky.detect_clearsky(
            self.measured, self.clearsky, self.times, self.window_length
        )
