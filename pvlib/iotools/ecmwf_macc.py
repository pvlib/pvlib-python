"""
Read data from ECMWF MACC Reanalysis.
"""

import threading
import pandas as pd

try:
    import netCDF4
except ImportError:
    class netCDF4:
        @staticmethod
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

#: map of ECMWF MACC parameter keynames and codes used in API
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
                   server=None, target=_ecmwf):
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
    lookup_params : bool, default True
        optional flag, if ``False``, then codes are already formatted
    server : ecmwfapi.api.ECMWFDataServer
        optionally provide a server object, default is ``None``
    target : callable
        optional function that calls ``server.retrieve`` to pass to thread

    Returns
    -------
    t : thread
        a thread object, use it to check status by calling `t.is_alive()`

    Notes
    -----
    To download data from ECMWF requires the API client and a registration
    key. Please read the documentation in `Access ECMWF Public Datasets
    <https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets>`_.
    Follow the instructions in step 4 and save the ECMWF registration key
    as `$HOME/.ecmwfapirc` or set `ECMWF_API_KEY` as the path to the key.

    This function returns a daemon thread that runs in the background. Exiting
    Python will kill this thread, however this thread will not block the main
    thread or other threads. This thread will terminate when the file is
    downloaded or if the thread raises an unhandled exception. You may submit
    multiple requests simultaneously to break up large downloads. You can also
    check the status and retrieve downloads online at
    http://apps.ecmwf.int/webmars/joblist/. This is useful if you kill the
    thread. Downloads expire after 24 hours.

    .. warning:: Your request may be queued online for an hour or more before
        it begins to download

    Precipitable water :math:`P_{wat}` is equivalent to the total column of
    water vapor (TCWV), but the units given by ECMWF MACC Reanalysis are kg/m^2
    at STP (1-atm, 25-C). Divide by ten to convert to centimeters of
    precipitable water:

    .. math::
        P_{wat} \\left( \\text{cm} \\right) \
        = TCWV \\left( \\frac{\\text{kg}}{\\text{m}^2} \\right) \
        \\frac{100 \\frac{\\text{cm}}{\\text{m}}} \
        {1000 \\frac{\\text{kg}}{\\text{m}^3}}

    The keynames available for the ``params`` argument are given by
    :const:`pvlib.iotools.ecmwf_macc.PARAMS` which maps the keys to codes used
    in the API. The following keynames are available:

    =======  =========================================
    keyname  description
    =======  =========================================
    tcwv     total column water vapor in kg/m^2 at STP
    aod550   aerosol optical depth measured at 550-nm
    aod469   aerosol optical depth measured at 469-nm
    aod670   aerosol optical depth measured at 670-nm
    aod865   aerosol optical depth measured at 865-nm
    aod1240  aerosol optical depth measured at 1240-nm
    =======  =========================================

    If ``lookup_params`` is ``False`` then ``params`` must contain the codes
    preformatted according to the ECMWF MACC Reanalysis API. This is useful if
    you want to retrieve codes that are not mapped in
    :const:`pvlib.iotools.ecmwf_macc.PARAMS`.

    Specify a custom ``target`` function to modify how the ECMWF API function
    ``server.retrieve`` is called. The ``target`` function must have the
    following signature in which the parameter definitions are similar to
    :func:`pvlib.iotools.get_ecmwf_macc`. ::


        target(server, startdate, stopdate, params, filename) -> None

    Examples
    --------
    Retrieve the AOD measured at 550-nm and the total column of water vapor for
    November 1, 2012.

    >>> from datetime import date
    >>> from pvlib.iotools import get_ecmwf_macc
    >>> filename = 'aod_tcwv_20121101.nc'  # .nc extension added if missing
    >>> params = ('aod550', 'tcwv')
    >>> start = end = date(2012, 11, 1)
    >>> t = get_ecmwf_macc(filename, params, start, end)
    >>> t.is_alive()
    True

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
        Get nearest indices to (latitude, longitude).

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

    def interp_data(self, latitude, longitude, utc_time, param):
        """
        Interpolate ``param`` values to ``utc_time`` using indices nearest to
        (``latitude, longitude``).

        Parmaeters
        ----------
        latitude : float
            Latitude in degrees
        longitude : float
            Longitude in degrees
        utc_time : datetime.datetime or datetime.date
            Naive or UTC date or datetime to interpolate
        param : str
            Name of the parameter to interpolate from the data

        Returns
        -------
        Interpolated ``param`` value at (``utc_time, latitude, longitude``)

        Examples
        --------
        Use this to get a single value of a parameter in the data at a specific
        time and set of (latitude, longitude) coordinates.

        >>> from datetime import datetime
        >>> from pvlib.iotools import ecmwf_macc
        >>> data = ecmwf_macc.ECMWF_MACC('aod_tcwv_20121101.nc')
        >>> dt = datetime(2012, 11, 1, 11, 33, 1)
        >>> data.interp_data(38.2, -122.1, dt, 'aod550')
        """
        nctime = self.data['time']  # time
        ilat, ilon = self.get_nearest_indices(latitude, longitude)
        # time index before
        before = netCDF4.date2index(utc_time, nctime, select='before')
        fbefore = self.data[param][before, ilat, ilon]
        fafter = self.data[param][before + 1, ilat, ilon]
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
            df['precipitable_water'] = df[ECMWF_MACC.TCWV] / 10.0
    finally:
        ecmwf_macc.data.close()
    return pd.DataFrame(df, index=times.astype('datetime64[s]'))
