"""
ASV benchmarks for tracking.py
"""

import pandas as pd
from pvlib import tracking, solarposition
import numpy as np


class SingleAxis:

    def setup(self):
        self.times = pd.date_range(start='20180601', freq='1min',
                                   periods=14400)
        self.lat = 35.1
        self.lon = -106.6
        self.solar_position = solarposition.get_solarposition(self.times,
                                                              self.lat,
                                                              self.lon)
        self.tracker = tracking.SingleAxisTracker()

    def time_singleaxis(self):
        with np.errstate(invalid='ignore'):
            tracking.singleaxis(self.solar_position.apparent_zenith,
                                self.solar_position.azimuth,
                                axis_tilt=0,
                                axis_azimuth=0,
                                max_angle=60,
                                backtrack=True,
                                gcr=0.45)

    def time_tracker_singleaxis(self):
        with np.errstate(invalid='ignore'):
            self.tracker.singleaxis(self.solar_position.apparent_zenith,
                                    self.solar_position.azimuth)
