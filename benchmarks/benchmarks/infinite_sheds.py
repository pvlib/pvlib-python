"""
ASV benchmarks for infinite_sheds.py
"""

import numpy as np
import pandas as pd
from pvlib.bifacial import infinite_sheds
from pvlib import location, tracking


class InfiniteSheds:

    # benchmark variant parameters (run both vectorize=True and False)
    params = [True, False]
    param_names = ['vectorize']

    def setup(self, vectorize):
        self.times = pd.date_range(start='20180601', freq='1min',
                                   periods=1440)
        self.location = location.Location(40, -80)
        self.solar_position = self.location.get_solarposition(self.times)
        self.clearsky_irradiance = self.location.get_clearsky(
            self.times,
            solar_position=self.solar_position,
        )
        self.surface_tilt = 20
        self.surface_azimuth = 180
        self.gcr = 0.35
        self.height = 2.5
        self.pitch = 5.
        self.albedo = 0.2
        self.npoints = 100

        with np.errstate(invalid='ignore'):
            self.tracking = tracking.singleaxis(
                self.solar_position['apparent_zenith'],
                self.solar_position['azimuth'],
                axis_tilt=0,
                axis_azimuth=0,
                max_angle=60,
                backtrack=True,
                gcr=self.gcr
            )

    def time_get_irradiance_poa_fixed(self, vectorize):
        infinite_sheds.get_irradiance_poa(
            surface_tilt=self.surface_tilt,
            surface_azimuth=self.surface_azimuth,
            solar_zenith=self.solar_position['apparent_zenith'],
            solar_azimuth=self.solar_position['azimuth'],
            gcr=self.gcr,
            height=self.height,
            pitch=self.pitch,
            ghi=self.clearsky_irradiance['ghi'],
            dhi=self.clearsky_irradiance['dhi'],
            dni=self.clearsky_irradiance['dni'],
            albedo=self.albedo,
            npoints=self.npoints,
            vectorize=vectorize,
        )

    def time_get_irradiance_poa_tracking(self, vectorize):
        infinite_sheds.get_irradiance_poa(
            surface_tilt=self.tracking['surface_tilt'],
            surface_azimuth=self.tracking['surface_azimuth'],
            solar_zenith=self.solar_position['apparent_zenith'],
            solar_azimuth=self.solar_position['azimuth'],
            gcr=self.gcr,
            height=self.height,
            pitch=self.pitch,
            ghi=self.clearsky_irradiance['ghi'],
            dhi=self.clearsky_irradiance['dhi'],
            dni=self.clearsky_irradiance['dni'],
            albedo=self.albedo,
            npoints=self.npoints,
            vectorize=vectorize,
        )

    def time_get_irradiance_fixed(self, vectorize):
        infinite_sheds.get_irradiance(
            surface_tilt=self.surface_tilt,
            surface_azimuth=self.surface_azimuth,
            solar_zenith=self.solar_position['apparent_zenith'],
            solar_azimuth=self.solar_position['azimuth'],
            gcr=self.gcr,
            height=self.height,
            pitch=self.pitch,
            ghi=self.clearsky_irradiance['ghi'],
            dhi=self.clearsky_irradiance['dhi'],
            dni=self.clearsky_irradiance['dni'],
            albedo=self.albedo,
            npoints=self.npoints,
            vectorize=vectorize,
        )

    def time_get_irradiance_tracking(self, vectorize):
        infinite_sheds.get_irradiance(
            surface_tilt=self.tracking['surface_tilt'],
            surface_azimuth=self.tracking['surface_azimuth'],
            solar_zenith=self.solar_position['apparent_zenith'],
            solar_azimuth=self.solar_position['azimuth'],
            gcr=self.gcr,
            height=self.height,
            pitch=self.pitch,
            ghi=self.clearsky_irradiance['ghi'],
            dhi=self.clearsky_irradiance['dhi'],
            dni=self.clearsky_irradiance['dni'],
            albedo=self.albedo,
            npoints=self.npoints,
            vectorize=vectorize,
        )
