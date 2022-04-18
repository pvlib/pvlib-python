"""
ASV benchmarks for scaling.py
"""

import pandas as pd
from pvlib import scaling
import numpy as np


class Scaling:

    def setup(self):
        self.n = 1000
        lat = np.array((9.99, 10, 10.01))
        lon = np.array((4.99, 5, 5.01))
        self.coordinates = np.array([(lati, loni) for
                                     (lati, loni) in zip(lat, lon)])
        self.times = pd.date_range('2019-01-01', freq='1T', periods=self.n)
        self.positions = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        self.clearsky_index = pd.Series(np.random.rand(self.n),
                                        index=self.times)
        self.cloud_speed = 5
        self.tmscales = np.array((1, 2, 4, 8, 16, 32, 64,
                                 128, 256, 512, 1024, 2048, 4096))

    def time_latlon_to_xy(self):
        scaling.latlon_to_xy(self.coordinates)

    def time__compute_wavelet(self):
        scaling._compute_wavelet(self.clearsky_index, dt=1)

    def time__compute_vr(self):
        scaling._compute_vr(self.positions, self.cloud_speed, self.tmscales)

    def time_wvm(self):
        scaling.wvm(self.clearsky_index, self.positions,
                    self.cloud_speed, dt=1)
