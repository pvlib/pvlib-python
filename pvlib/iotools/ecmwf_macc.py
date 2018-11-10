"""
Read data from ECMWF MACC Reanalysis.
"""

from __future__ import division, print_function, absolute_import
import os
import netCDF4
import numpy as np


class ECMWF_MACC(object):
    """container for data"""

    DIMENSIONS = ['longitude', 'latitude', 'time']

    def __init__(self, filename, delta_time):
        self.delta_time = delta_time  # time resolution
        self.data = netCDF4.Dataset(filename)
        self.variables = tuple(self.data.variables.keys())
        # data names
        names = list(self.variables)
        for v in ECMWF_MACC.DIMENSIONS:
            n = self.variables.index(v)  # raises ValueError if missing
            names.pop(n)  # not a data name
        self.names = tuple(names)
        # size of lat/lon dimensions
        self.lat_size = self.data.dimensions['latitude'].size
        self.lon_size = self.data.dimensions['longitude'].size
        # spatial resolution in degrees
        self.delta_lat = 180.0 / (self.lat_size - 1)
        self.delta_lon = 360.0 / self.lon_size

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

    def interp_data(self, lat, lon, utc_time, data, key):
        """
        Interpolate data using nearest neighbor.
        """
        nctime = data['time']  # time
        ilat, ilon = self.get_nearest_indices(lat, lon)
        # time index before
        before = netCDF4.date2index(utc_time, nctime, select='before')
        fbefore = data[key][before, ilat, ilon]
        fafter = data[key][before + 1, ilat, ilon]
        dt_num = netCDF4.date2num(utc_time, nctime.units)
        time_ratio = (dt_num - nctime[before]) / self.delta_time
        return fbefore + (fafter - fbefore) * time_ratio


def read_ecmwf_macc(key, latitude, longitude, times, filename):
    pass
