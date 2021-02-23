from pvlib import tracking


class FixedMount:
    def __init__(self, surface_tilt=0, surface_azimuth=180):
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth

    def __repr__(self):
        return (
            'FixedMount:'
            f'\n    surface_tilt: {self.surface_tilt}'
            f'\n    surface_azimuth: {self.surface_azimuth}'
        )

    def get_orientation(self, solar_zenith, solar_azimuth):
        return {
            'surface_tilt': self.surface_tilt,
            'surface_azimuth': self.surface_azimuth,
        }


class SingleAxisTrackerMount:
    def __init__(self, axis_tilt, axis_azimuth, max_angle, backtrack, gcr,
                 cross_axis_tilt):
        self.axis_tilt = axis_tilt
        self.axis_azimuth = axis_azimuth
        self.max_angle = max_angle
        self.backtrack = backtrack
        self.gcr = gcr
        self.cross_axis_tilt = cross_axis_tilt

    def __repr__(self):
        attrs = ['axis_tilt', 'axis_azimuth', 'max_angle',
                 'backtrack', 'gcr', 'cross_axis_tilt']

        return 'SingleAxisTrackerMount:\n    ' + '\n    '.join(
            f'{attr}: {getattr(self, attr)}' for attr in attrs
        )

    def get_orientation(self, solar_zenith, solar_azimuth):
        tracking_data = tracking.singleaxis(
            solar_zenith, solar_azimuth,
            self.axis_tilt, self.axis_azimuth,
            self.max_angle, self.backtrack,
            self.gcr, self.cross_axis_tilt
        )
        return tracking_data
