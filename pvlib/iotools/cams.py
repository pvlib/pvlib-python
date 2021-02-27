"""Functions to access data from Copernicus Atmosphere Monitoring Service
    (CAMS) radiation service.
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import pandas as pd
import requests
import io
import warnings


CAMS_RADIATION_INTEGRATED_COLUMNS = [
    'TOA', 'Clear sky GHI', 'Clear sky BHI', 'Clear sky DHI', 'Clear sky BNI',
    'GHI', 'BHI', 'DHI', 'BNI',
    'GHI no corr', 'BHI no corr', 'DHI no corr', 'BNI no corr']

# Dictionary mapping CAMS McClear and Radiation variables to pvlib names
CAMS_RADIATION_VARIABLE_MAP = {
    'TOA': 'ghi_extra',
    'Clear sky GHI': 'ghi_clear',
    'Clear sky BHI': 'bhi_clear',
    'Clear sky DHI': 'dhi_clear',
    'Clear sky BNI': 'dni_clear',
    'GHI': 'ghi',
    'BHI': 'bhi',
    'DHI': 'dhi',
    'BNI': 'dni',
    'sza': 'solar_zenith',
}


# Dictionary mapping Python time steps to CAMS time step format
TIME_STEPS_MAP = {'1min': 'PT01M', '15min': 'PT15M', '1h': 'PT01H',
                  '1d': 'P01D', '1M': 'P01M'}

TIME_STEPS_IN_HOURS = {'1min': 1/60, '15min': 15/60, '1h': 1, '1d': 24}

SUMMATION_PERIOD_TO_TIME_STEP = {'0 year 0 month 0 day 0 h 1 min 0 s': '1min',
                                 '0 year 0 month 0 day 0 h 15 min 0 s': '15min',  # noqa
                                 '0 year 0 month 0 day 1 h 0 min 0 s': '1h',
                                 '0 year 0 month 1 day 0 h 0 min 0 s': '1d',
                                 '0 year 1 month 0 day 0 h 0 min 0 s': '1M'}


def get_cams_radiation(start_date, end_date, latitude, longitude, email,
                       service='mcclear', altitude=None, time_step='1h',
                       time_ref='UT', verbose=False, integrated=False,
                       label=None, map_variables=True,
                       server='www.soda-is.com'):
    """
    Retrieve time-series of radiation and/or clear-sky global, beam, and
    diffuse radiation from CAMS [2]_ using the WGET service [3]_.

    Time coverage: 2004-01-01 to two days ago
    Access: free, but requires registration, see [1]_
    Requests: max. 100 per day
    Geographical coverage: Wordwide for CAMS McClear and -66° to 66° in both
                           latitude and longitude for CAMS Radiation


    Parameters
    ----------
    start_date: datetime like
        First day of the requested period
    end_date: datetime like
        Last day of the requested period
    latitude: float
        in decimal degrees, between -90 and 90, north is positive (ISO 19115)
    longitude : float
        in decimal degrees, between -180 and 180, east is positive (ISO 19115)
    altitude: float, default: None
        Altitude in meters. If None, then the altitude is determined from the
        NASA SRTM database
    email: str
        Email address linked to a SoDa account
    service: {'mcclear', 'cams_radiation'}
        Specify which whether to retrieve CAMS Radiation or McClear parameters
    time_step: str, {'1min', '15min', '1h', '1d', '1M'}, default: '1h'
        Time step of the time series, either 1 minute, 15 minute, hourly,
        daily, or monthly.
    time_reference: str, {'UT', 'TST'}, default: 'UT'
        'UT' (universal time) or 'TST' (True Solar Time)
    verbose: boolean, default: False
        Verbose mode outputs additional parameters (aerosols). Only avaiable
        for 1 minute and universal time. See [1] for parameter description.
    integrated: boolean, default False
        Whether to return integrated irradiation values (Wh/m^2) from CAMS or
        average irradiance values (W/m^2) as is more commonly used
    label: {‘right’, ‘left’}, default: None
        Which bin edge label to label bucket with. The default is ‘left’ for
        all frequency offsets except for ‘M’ which has a default of ‘right’.
    map_variables: bool, default: True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable CAMS_VARIABLE_MAP.
    server: str, default: 'www.soda-is.com'
        Main server (www.soda-is.com) or backup mirror server (pro.soda-is.com)

    Returns
    -------
    data: pandas.DataFrame
        Timeseries data, see Notes for columns
    meta: dict
        Metadata for the requested time-series

    Notes
    -----
    In order to use the CAMS services, users must registre for a free SoDa
    account using an email addres [1]_.

    The returned data DataFrame includes the following fields:

    =======================  ======  ==========================================
    Key, mapped key          Format  Description
    =======================  ======  ==========================================
    **Mapped field names are returned when the map_variables argument is True**
    --------------------------------------------------------------------------
    Observation period       str     Beginning/end of time period
    TOA, ghi_extra           float   Horizontal radiation at top of atmosphere
    Clear sky GHI, ghi_clear float   Clear sky global radiation on horizontal
    Clear sky BHI, bhi       float   Clear sky beam radiation on horizontal
    Clear sky DHI, dhi_clear float   Clear sky diffuse radiation on horizontal
    Clear sky BNI, dni_clear float   Clear sky beam radiation normal to sun
    GHI, ghi*                float   Global horizontal radiation
    BHI, bhi*                float   Beam (direct) radiation on horizontal
    DHI, dhi*                float   Diffuse horizontal radiation
    BNI, dni*                float   Beam (direct) radiation normal to the sun
    Reliability*             float   Fraction of reliable data in summarization
    =======================  ======  ==========================================

    *Parameters only returned if service='cams_radiation'. For description of
    additional output parameters in verbose mode, see [1]_ and [2]_.

    The returned units for the radiation parameters depends on the integrated
    argument, i.e. integrated=False returns units of W/m2, whereas
    integrated=True returns units of Wh/m2.

    Note that it is recommended to specify the latitude and longitude to at
    least the fourth decimal place.

    Variables corresponding to standard pvlib variables are renamed,
    e.g. `sza` becomes `solar_zenith`. See the
    `pvlib.iotools.cams.CAMS_VARIABLE_MAP` dict for the complete mapping.

    See Also
    --------
    pvlib.iotools.read_cams_radiation, pvlib.iotools.parse_cams_radiation

    Raises
    ------
    requests.HTTPError
        If the request is invalid, then an XML file is returned by the CAMS
        service and the error message will be raised as an expcetion.

    References
    ----------
    .. [1] `CAMS Radiation Service Info
       <http://www.soda-pro.com/web-services/radiation/cams-radiation-service/info>`_
    .. [2] `CAMS McClear Service Info
       <http://www.soda-pro.com/web-services/radiation/cams-mcclear/info>`_
    .. [3] `CAMS McClear Automatic Access
       <http://www.soda-pro.com/help/cams-services/cams-mcclear-service/automatic-access>`_
    """
    if time_step in TIME_STEPS_MAP.keys():
        time_step_str = TIME_STEPS_MAP[time_step]
    else:
        warnings.warn('Time step not recognized, 1 hour time step used!')
        time_step, time_step_str = '1h', 'PT01H'

    if (verbose is True) & ((time_step != '1min') | (time_ref != 'UT')):
        verbose = False
        warnings.warn("Verbose mode only supports 1 min. UT time series!")

    # Format verbose variable to the required format: {'true', 'false'}
    verbose = str(verbose).lower()

    if altitude is None:  # Let SoDa get elevation from the NASA SRTM database
        altitude = -999

    # Start and end date should be in the format: yyyy-mm-dd
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    email = email.replace('@', '%2540')  # Format email address
    service = 'get_{}'.format(service)  # Format CAMS service string

    # Manual format the request url, due to uncommon usage of & and ; in url
    url = ("http://{}/service/wps?Service=WPS&Request=Execute&"
           "Identifier={}&version=1.0.0&RawDataOutput=irradiation&"
           "DataInputs=latitude={};longitude={};altitude={};"
           "date_begin={};date_end={};time_ref={};summarization={};"
           "username={};verbose={}"
           ).format(server, service, latitude, longitude, altitude, start_date,
                    end_date, time_ref, time_step_str, email, verbose)

    res = requests.get(url)

    # Invalid requests returns helpful XML error message
    if res.headers['Content-Type'] == 'application/xml':
        errors = res.text.split('ows:ExceptionText')[1][1:-2]
        raise requests.HTTPError(errors, response=res)
    # Check if returned file is a csv data file
    elif res.headers['Content-Type'] == 'application/csv':
        fbuf = io.StringIO(res.content.decode('utf-8'))
        data, meta = parse_cams_radiation(fbuf, integrated=integrated,
                                          label=label,
                                          map_variables=map_variables)
        return data, meta
    else:
        warnings.warn('File content type not recognized.')


def parse_cams_radiation(fbuf, integrated=False, label=None,
                         map_variables=True):
    """
    Parse a file-like buffer with data in the format of a CAMS Radiation or
    McClear file. The CAMS servicess are described in [1]_ and [2]_.

    Parameters
    ----------
    fbuf: file-like object
        File-like object containing data to read.
    integrated: boolean, default False
        Whether to return integrated irradiation values (Wh/m^2) from CAMS or
        average irradiance values (W/m^2) as is more commonly used
    label: {‘right’, ‘left’}, default: None
        Which bin edge label to label bucket with. The default is ‘left’ for
        all frequency offsets except for ‘M’ which has a default of ‘right’.
    map_variables: bool, default: True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable CAMS_VARIABLE_MAP.

    Returns
    -------
    data: pandas.DataFrame
        Timeseries data from CAMS Radiation or McClear
    meta: dict
        Metadata avaiable in the file.

    See Also
    --------
    pvlib.iotools.read_cams_radiation, pvlib.iotools.get_cams_radiation

    References
    ----------
    .. [1] `CAMS Radiation Service Info
       <http://www.soda-pro.com/web-services/radiation/cams-radiation-service/info>`_
    .. [2] `CAMS McClear Service Info
       <http://www.soda-pro.com/web-services/radiation/cams-mcclear/info>`_
    """
    meta = {}
    # Initial lines starting with # contain meta-data
    while True:
        line = fbuf.readline().rstrip('\n')
        if line.startswith('# Observation period'):
            # The last line of the meta-data section contains the column names
            names = line.lstrip('# ').split(';')
            break  # End of meta-data section has been reached
        elif ': ' in line:
            meta[line.split(': ')[0].lstrip('# ')] = line.split(': ')[1]

    # Convert the latitude, longitude, and altitude from strings to floats
    meta['Latitude (positive North, ISO 19115)'] = \
        float(meta['Latitude (positive North, ISO 19115)'])
    meta['Longitude (positive East, ISO 19115)'] = \
        float(meta['Longitude (positive East, ISO 19115)'])
    meta['Altitude (m)'] = float(meta['Altitude (m)'])
    meta['Unit for radiation'] = {True: 'Wh/m2', False: 'W/m2'}[integrated]

    # Determine the time_step from the meta-data dictionary
    time_step = SUMMATION_PERIOD_TO_TIME_STEP[
        meta['Summarization (integration) period']]
    meta['time_step'] = time_step

    data = pd.read_csv(fbuf, sep=';', comment='#', header=None, names=names)

    obs_period = data['Observation period'].str.split('/')

    # Set index as the start observation time (left) and localize to UTC
    if (label == 'left') | ((label is None) & (time_step != '1M')):
        data.index = pd.to_datetime(obs_period.str[0], utc=True)
    # Set index as the stop observation time (right) and localize to UTC
    # default label for monthly data is 'right' following Pandas' convention
    elif (label == 'right') | ((label is None) & (time_step == '1M')):
        data.index = pd.to_datetime(obs_period.str[1], utc=True)

    data.index.name = 'time'  # Set index name to None

    # Change index for time_step '1d' and '1M' to be date and not datetime
    if (time_step == '1d') | (time_step == '1M'):
        data.index = pd.DatetimeIndex(data.index.date)
    # For monthly data with 'right' label, the index should be the last
    # date of the month and not the first date of the following month
    if (time_step == '1M') & (label != 'left'):
        data.index = data.index - pd.Timedelta(days=1)

    if not integrated:  # Convert radiation values from Wh/m2 to W/m2
        integrated_cols = [c for c in CAMS_RADIATION_INTEGRATED_COLUMNS
                           if c in data.columns]

        if time_step == '1M':
            time_delta = (pd.to_datetime(obs_period.str[1])
                          - pd.to_datetime(obs_period.str[0]))
            hours = time_delta.dt.total_seconds()/60/60
            data[integrated_cols] = data[integrated_cols].\
                divide(hours.tolist(), axis='rows')
        else:
            data[integrated_cols] = (data[integrated_cols] /
                                     TIME_STEPS_IN_HOURS[time_step])

    if map_variables:
        data = data.rename(columns=CAMS_RADIATION_VARIABLE_MAP)

    return data, meta


def read_cams_radiation(filename, integrated=False, label=None,
                        map_variables=True):
    """
    Read a CAMS Radiation or McClear file into a pandas DataFrame. CAMS
    radiation and McClear is described in [1]_ and [2]_, respectively.

    Parameters
    ----------
    filename: str
        Filename of a file containing data to read.
    integrated: boolean, default False
        Whether to return integrated irradiation values (Wh/m^2) from CAMS or
        average irradiance values (W/m^2) as is more commonly used
    label: {‘right’, ‘left’}, default: None
        Which bin edge label to label bucket with. The default is ‘left’ for
        all frequency offsets except for ‘M’ which has a default of ‘right’.
    map_variables: bool, default: True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable CAMS_VARIABLE_MAP.

    Returns
    -------
    data: pandas.DataFrame
        Timeseries data from CAMS Radiation or McClear
        :func:`pvlib.iotools.get_cams` for fields
    meta: dict
        Metadata avaiable in the file.

    See Also
    --------
    pvlib.iotools.parse_cams_radiation, pvlib.iotools.get_cams_radiation

    References
    ----------
    .. [1] `CAMS Radiation Service Info
       <http://www.soda-pro.com/web-services/radiation/cams-radiation-service/info>`_
    .. [2] `CAMS McClear Service Info
       <http://www.soda-pro.com/web-services/radiation/cams-mcclear/info>`_
    """
    with open(str(filename), 'r') as fbuf:
        content = parse_cams_radiation(fbuf)
    return content
