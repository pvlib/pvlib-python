"""
Read data from ECMWF MACC Reanalysis.
"""

from __future__ import division
import threading
import netCDF4
import pandas as pd

try:
    import netCDF4
except ImportError as exc:
    class netCDF4:
        def Dataset(*a, **kw):
            raise ImportError(
                'Reading ECMWF data requires netCDF4 to be installed.')

try:
    from ecmwfapi import ECMWFDataServer
except ImportError:
    def ECMWFDataServer(*a, **kw):
        raise ImportError(
            'To download data from ECMWF requires the API client.\nSee https:/'
            '/confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets'
        )
    SERVER = None
else:
    SERVER = ECMWFDataServer()

PARAMS = {
    "tcwv": "137.128",
    "aod550": "207.210",
    'aod469': '213.210',
    'aod670': '214.210',
    'aod865': '215.210',
    "aod1240": "216.210",
}


def _ecmwf(server, startdate, stopdate, params, targetname):
    # see http://apps.ecmwf.int/datasets/data/macc-reanalysis/levtype=sfc/
    server.retrieve({
        "class": "mc",
        "dataset": "macc",
        "date": "%s/to/%s" % (startdate, stopdate),
        "expver": "rean",
        "grid": "0.75/0.75",
        "levtype": "sfc",
        "param": params,
        "step": "3/6/9/12/15/18/21/24",
        "stream": "oper",
        "format": "netcdf",
        "time": "00:00:00",
        "type": "fc",
        "target": targetname,
    })


def get_ecmwf_macc(filename, params, startdate, stopdate, lookup_params=True,
                   server=SERVER, target=_ecmwf):
    """
    Download data from ECMWF MACC Reanalysis API.

    Parameters
    ----------
    filename : str
        full path of file where to save data, ``.nc`` appended if not given
    params : str or sequence of str
        keynames of parameter[s] to download
    startdate : datetime.datetime or datetime.date
        UTC date
    stopdate : datetime.datetime or datetime.date
        UTC date
    lookup_params : bool
        optional flag, if ``False``, then codes are already formatted, default
        is ``True``
    server : ecmwfapi.api.ECMWFDataServer
        optionally provide a server object, default is given
    target : callable
        optional function that calls ``server.retreive`` to pass to thread

    Returns
    -------
    t : thread
        a thread object, use it to check status by calling `t.is_alive()`

    This is a daemon thread that runs in the background. Exiting Python will
    kill this thread, however this thread will not block the main thread or
    other threads. This thread will terminate when the file is downloaded or if
    the thread raises an unhandled exception. You may submit multiple requests
    simultaneously to break up large downloads. You can also check the status
    and retrieve downloads online at `http://apps.ecmwf.int/webmars/joblist/`_.
    This is useful if you kill the thread. Downloads expire after 24 hours.

    .. warning:: Your request may be queued online for an hour or more before
        it begins to download

    """
    if not filename.endswith('nc'):
        filename += '.nc'
    if lookup_params:
        try:
            params = '/'.join(PARAMS.get(p) for p in params)
        except TypeError:
            params = PARAMS.get(params)
    startdate = startdate.strftime('%Y-%m-%d')
    stopdate = stopdate.strftime('%Y-%m-%d')
    if not server:
        server = ECMWFDataServer()
    t = threading.Thread(target=target, daemon=True,
                         args=(server, startdate, stopdate, params, filename))
    t.start()
    return t


class ECMWF_MACC(object):
    """container for ECMWF MACC reanalysis data"""

    TCWV = 'tcwv'  # total column water vapor in kg/m^2 at (1-atm,25-degC)

    def __init__(self, filename):
        self.data = netCDF4.Dataset(filename)
        # data variables and dimensions
        variables = set(self.data.variables.keys())
        dimensions = set(self.data.dimensions.keys())
        self.keys = tuple(variables - dimensions)
        # size of lat/lon dimensions
        self.lat_size = self.data.dimensions['latitude'].size
        self.lon_size = self.data.dimensions['longitude'].size
        # spatial resolution in degrees
        self.delta_lat = -180.0 / (self.lat_size - 1)  # from north to south
        self.delta_lon = 360.0 / self.lon_size  # from west to east
        # time resolution in hours
        self.time_size = self.data.dimensions['time'].size
        self.start_time = self.data['time'][0]
        self.stop_time = self.data['time'][-1]
        self.time_range = self.stop_time - self.start_time
        self.delta_time = self.time_range / (self.time_size - 1)

    def get_nearest_indices(self, latitude, longitude):
        """
        Get nearest indices to (lat, lon).

        Parmaeters
        ----------
        latitude : float
            Latitude in degrees
        longitude : float
            Longitude in degrees

        Returns
        -------
        idx_lat : int
            index of nearest latitude
        idx_lon : int
            index of nearest longitude
        """
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
        Interpolate data for UTC time using nearest indices to (lat, lon).
        """
        nctime = self.data['time']  # time
        ilat, ilon = self.get_nearest_indices(lat, lon)
        # time index before
        before = netCDF4.date2index(utc_time, nctime, select='before')
        fbefore = data[key][before, ilat, ilon]
        fafter = data[key][before + 1, ilat, ilon]
        dt_num = netCDF4.date2num(utc_time, nctime.units)
        time_ratio = (dt_num - nctime[before]) / self.delta_time
        return fbefore + (fafter - fbefore) * time_ratio


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
    try:
        ilat, ilon = ecmwf_macc.get_nearest_indices(latitude, longitude)
        nctime = ecmwf_macc.data['time']
        if utc_time_range:
            start_idx = netCDF4.date2index(
                utc_time_range[0], nctime, select='before')
            stop_idx = netCDF4.date2index(
                utc_time_range[-1], nctime, select='after')
            time_slice = slice(start_idx, stop_idx + 1)
        else:
            time_slice = slice(0, ecmwf_macc.time_size)
        times = netCDF4.num2date(nctime[time_slice], nctime.units)
        df = {k: ecmwf_macc.data[k][time_slice, ilat, ilon]
              for k in ecmwf_macc.keys}
        if ECMWF_MACC.TCWV in df:
            # convert total column water vapor in kg/m^2 at (1-atm, 25-degC) to
            # precipitable water in cm
            df['precipitable_water'] = ecmwf_macc.data[ECMWF_MACC.TCWV] / 10.0
    finally:
        ecmwf_macc.data.close()
    return pd.DataFrame(df, index=times.astype('datetime64[s]'))
