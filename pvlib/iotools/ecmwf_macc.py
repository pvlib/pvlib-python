"""
Read data from ECMWF MACC Reanalysis.
"""

from __future__ import division, print_function, absolute_import
import netCDF4
import pandas as pd


class ECMWF_MACC(object):
    """container for ECMWF MACC reanalysis data"""

    def __init__(self, filename):
        self.data = netCDF4.Dataset(filename)
        # data variables and dimensions
        variables = set(self.data.variables.keys())
        dimensions = set(self.data.dimensions.keys())
        self.variables = tuple(variables)
        self.dimensions = tuple(dimensions)
        # data names
        self.keys = tuple(variables - dimensions)
        # size of lat/lon dimensions
        self.lat_size = self.data.dimensions['latitude'].size
        self.lon_size = self.data.dimensions['longitude'].size
        # spatial resolution in degrees
        self.delta_lat = 180.0 / (self.lat_size - 1)
        self.delta_lon = 360.0 / self.lon_size
        self.time_size = self.data.dimensions['time'].size
        # time resolution in hours
        self.start_time = self.data['time'][0]
        self.stop_time = self.data['time'][-1]
        self.time_range = self.stop_time - self.start_time
        self.delta_time = self.time_range / (self.time_size - 1)

    def get_nearest_indices(self, latitude, longitude):
        # index of nearest latitude
        idx_lat = int(round((latitude - 90.0) / self.delta_lat))
        # avoid out of bounds latitudes
        if idx_lat < 0:
            idx_lat = 0  # if latitude == 90, north pole
        elif idx_lat > self.lat_size:
            idx_lat = self.lat_size  # if latitude == -90, south pole
        # adjust longitude from -180/180 to 0/360
        longitude = longitude % 360.0
        # index of nearest longitude
        idx_lon = int(round(longitude / self.delta_lon)) % self.lon_size
        return idx_lat, idx_lon


def read_ecmwf_macc(filename, latitude, longitude, utc_time_range=None):
    """
    Read data from ECMWF MACC reanalysis netCDF4 file.

    Parameters
    ----------
    filename : string
        full path to netCDF4 data file.
    latitude : float
        latitude in degrees
    longitude : float
        longitude in degrees
    utc_time_range : sequence of datetime.datetime
        pair of start and stop naive or UTC date-times

    Returns
    -------
    data : pandas.DataFrame
        dataframe for specified range of UTC date-times
    """
    ecmwf_macc = ECMWF_MACC(filename)
    ilat, ilon = ecmwf_macc.get_nearest_indices(latitude, longitude)
    nctime = ecmwf_macc.data['time']
    if utc_time_range:
        start_idx = netCDF4.date2index(
            utc_time_range[0], nctime, select='before')
        stop_idx = netCDF4.date2index(
            utc_time_range[-1], nctime, select='after')
        idx_slice = slice(start_idx, stop_idx + 1)
    else:
        idx_slice = slice(0, ecmwf_macc.time_size)
    times = netCDF4.num2date(
        nctime[idx_slice], nctime.units)
    df = {k: ecmwf_macc.data[k][idx_slice, ilat, ilon] for k in ecmwf_macc.keys}
    return pd.DataFrame(df, index=times.astype('datetime64[s]'))
