"""
ASV benchmarks for irradiance.py
"""

import pandas as pd
from pvlib import irradiance, location


class Irradiance:

    def setup(self):
        self.times = pd.date_range(start='20180601', freq='1min',
                                   periods=14400)
        self.days = pd.date_range(start='20180601', freq='d', periods=30)
        self.location = location.Location(40, -80)
        self.solar_position = self.location.get_solarposition(self.times)
        self.clearsky_irradiance = self.location.get_clearsky(self.times)
        self.tilt = 20
        self.azimuth = 180
        self.aoi = irradiance.aoi(self.tilt, self.azimuth,
                                  self.solar_position.apparent_zenith,
                                  self.solar_position.azimuth)

    def time_get_extra_radiation(self):
        irradiance.get_extra_radiation(self.days)

    def time_aoi(self):
        irradiance.aoi(self.tilt, self.azimuth,
                       self.solar_position.apparent_zenith,
                       self.solar_position.azimuth)

    def time_aoi_projection(self):
        irradiance.aoi_projection(self.tilt, self.azimuth,
                                  self.solar_position.apparent_zenith,
                                  self.solar_position.azimuth)

    def time_get_ground_diffuse(self):
        irradiance.get_ground_diffuse(self.tilt, self.clearsky_irradiance.ghi)

    def time_get_total_irradiance(self):
        irradiance.get_total_irradiance(self.tilt, self.azimuth,
                                        self.solar_position.apparent_zenith,
                                        self.solar_position.azimuth,
                                        self.clearsky_irradiance.dni,
                                        self.clearsky_irradiance.ghi,
                                        self.clearsky_irradiance.dhi)

    def time_disc(self):
        irradiance.disc(self.clearsky_irradiance.ghi,
                        self.solar_position.apparent_zenith,
                        self.times)

    def time_dirint(self):
        irradiance.dirint(self.clearsky_irradiance.ghi,
                          self.solar_position.apparent_zenith,
                          self.times)

    def time_dirindex(self):
        irradiance.dirindex(self.clearsky_irradiance.ghi,
                            self.clearsky_irradiance.ghi,
                            self.clearsky_irradiance.dni,
                            self.solar_position.apparent_zenith,
                            self.times)

    def time_erbs(self):
        irradiance.erbs(self.clearsky_irradiance.ghi,
                        self.solar_position.apparent_zenith,
                        self.times)
