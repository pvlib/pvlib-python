"""Functions to access data from Copernicus Atmosphere Monitoring Service
    (CAMS) radiation service.
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import pandas as pd
import requests
import io


MCCLEAR_COLUMNS = ['Observation period', 'TOA', 'Clear sky GHI',
                   'Clear sky BHI', 'Clear sky DHI', 'Clear sky BNI']

MCCLEAR_VERBOSE_COLUMNS = ['sza', 'summer/winter split', 'tco3', 'tcwv',
                           'AOD BC', 'AOD DU', 'AOD SS', 'AOD OR', 'AOD SU',
                           'AOD NI', 'AOD AM', 'alpha', 'Aerosol type',
                           'fiso', 'fvol', 'fgeo', 'albedo']

# Dictionary mapping CAMS MCCLEAR variables to pvlib names
MCCLEAR_VARIABLE_MAP = {
    'TOA': 'ghi_extra',
    'Clear sky GHI': 'ghi_clear',
    'Clear sky BHI': 'bhi_clear',
    'Clear sky DHI': 'dhi_clear',
    'Clear sky BNI': 'dni_clear',
    'sza': 'solar_zenith',
}


# Dictionary mapping Python time steps to CAMS time step format
TIME_STEPS_MAP = {'1min': 'PT01M', '15min': 'PT15M', '1h': 'PT01H', '1d': 'P01D',
                  '1M': 'P01M'}

TIME_STEPS_IN_HOURS = {'1min': 1/60, '15min': 15/60, '1h': 1, '1d': 24}

SUMMATION_PERIOD_TO_TIME_STEP = {'0 year 0 month 0 day 0 h 1 min 0 s': '1min',
                                 '0 year 0 month 0 day 0 h 15 min 0 s': '15min',
                                 '0 year 0 month 0 day 1 h 0 min 0 s': '1h',
                                 '0 year 0 month 1 day 0 h 0 min 0 s': '1d',
                                 '0 year 1 month 0 day 0 h 0 min 0 s': '1M'}


def get_cams_mcclear(start_date, end_date, latitude, longitude, email,
                     altitude=None, time_step='1h', time_ref='UT',
                     integrated=False, label=None, verbose=False,
                     map_variables=True, server='www.soda-is.com'):
    """
    Retrieve time-series of clear-sky global, beam, and diffuse radiation
    anywhere in the world from CAMS McClear [1]_ using the WGET service [2]_.


    Geographical coverage: wordwide
    Time coverage: 2004-01-01 to two days ago
    Access: free, but requires registration, see [1]_
    Requests: max. 100 per day


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
    time_step: str, {'1min', '15min', '1h', '1d', '1M'}, default: '1h'
        Time step of the time series, either 1 minute, 15 minute, hourly,
        daily, or monthly.
    time_reference: str, {'UT', 'TST'}, default: 'UT'
        'UT' (universal time) or 'TST' (True Solar Time)
    integrated: boolean, default False
        Whether to return integrated irradiation values (Wh/m^2) from CAMS or
        average irradiance values (W/m^2) as is more commonly used
    label: {‘right’, ‘left’}, default: None
        Which bin edge label to label bucket with. The default is ‘left’ for
        all frequency offsets except for ‘M’ which has a default of ‘right’.
    verbose: boolean, default: False
        Verbose mode outputs additional parameters (aerosols). Only avaiable
        for 1 minute and universal time. See [1] for parameter description.
    map_variables: bool, default: True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable MCCLEAR_VARIABLE_MAP.
    server: str, default: 'www.soda-is.com'
        Main server (www.soda-is.com) or backup mirror server (pro.soda-is.com)


    Notes
    ----------
    The returned data Dataframe includes the following fields:

    =======================  ======  ==========================================
    Key, mapped key          Format  Description
    =======================  ======  ==========================================
    **Mapped field names are returned when the map_variables argument is True**
    --------------------------------------------------------------------------
    Observation period       str     Beginning/end of time period
    TOA, ghi_extra           float   Horizontal radiation at top of atmosphere
    Clear sky GHI, ghi_clear float   Clear sky global radiation on horizontal
    Clear sky BHI, bhi_clear float   Clear sky beam radiation on horizontal
    Clear sky DHI, dhi_clear float   Clear sky diffuse radiation on horizontal
    Clear sky BNI, dni_clear float   Clear sky beam radiation normal to sun
    =======================  ======  ==========================================

    For the returned units see the integrated argument. For description of
    additional output parameters in verbose mode, see [1].

    Note that it is recommended to specify the latitude and longitude to at
    least the fourth decimal place.

    Variables corresponding to standard pvlib variables are renamed,
    e.g. `sza` becomes `solar_zenith`. See the
    `pvlib.iotools.cams.MCCLEAR_VARIABLE_MAP` dict for the complete mapping.

    See Also
    --------
    pvlib.iotools.read_cams_mcclear, pvlib.iotools.parse_cams_mcclear

    References
    ----------
    .. [1] `CAMS McClear Service Info
       <http://www.soda-pro.com/web-services/radiation/cams-mcclear/info>`_
    .. [2] `CAMS McClear Automatic Access
       <http://www.soda-pro.com/help/cams-services/cams-mcclear-service/automatic-access>`_
    """

    if time_step in TIME_STEPS_MAP.keys():
        time_step_str = TIME_STEPS_MAP[time_step]
    else:
        print('WARNING: time step not recognized, 1 hour time step used!')
        time_step_str = 'PT01H'

    names = MCCLEAR_COLUMNS
    if verbose:
        if (time_step == '1min') & (time_ref == 'UT'):
            names += MCCLEAR_VERBOSE_COLUMNS
        else:
            verbose = False
            print("Verbose mode only supports 1 min. UT time series!")

    if altitude is None:  # Let SoDa get elevation from the NASA SRTM database
        altitude = -999

    # Start and end date should be in the format: yyyy-mm-dd
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    email = email.replace('@', '%2540')  # Format email address

    # Format verbose variable to the required format: {'true', 'false'}
    verbose = str(verbose).lower()

    # Manual format the request url, due to uncommon usage of & and ; in url
    url = ("http://{}/service/wps?Service=WPS&Request=Execute&"
           "Identifier=get_mcclear&version=1.0.0&RawDataOutput=irradiation&"
           "DataInputs=latitude={};longitude={};altitude={};"
           "date_begin={};date_end={};time_ref={};summarization={};"
           "username={};verbose={}"
           ).format(server, latitude, longitude, altitude, start_date,
                    end_date, time_ref, time_step_str, email, verbose)

    res = requests.get(url)

    # Invalid requests returns helpful XML error message
    if res.headers['Content-Type'] == 'application/xml':
        print('REQUEST ERROR MESSAGE:')
        print(res.text.split('ows:ExceptionText')[1][1:-2])

    # Check if returned file is a csv data file
    elif res.headers['Content-Type'] == 'application/csv':
        fbuf = io.StringIO(res.content.decode('utf-8'))
        data, meta = parse_cams_mcclear(fbuf, integrated=integrated,
                                        label=label,
                                        map_variables=map_variables)
        return data, meta
    else:
        print('Error in recognizing the file content occurred!')


def parse_cams_mcclear(fbuf, integrated=False, label=None, map_variables=True):
    """
    Parse a CAMS McClear file. CAMS McClear is described in [1]_.

    See Also
    --------
    pvlib.iotools.read_cams_mcclear, pvlib.iotools.get_cams_mcclear

    References
    ----------
    .. [1] `CAMS McClear Service Info
       <http://www.soda-pro.com/web-services/radiation/cams-mcclear/info>`_
    """
    meta = {}
    # Initial lines of the file contain meta-data, which all start with #
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
    meta['Clear sky radiation unit'] = {True:'Wh/m2', False:'W/m2'}[integrated]

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

    if not integrated:  # Convert from Wh/m2 to W/m2
        integrated_cols = MCCLEAR_COLUMNS[1:6]

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
        data = data.rename(columns=MCCLEAR_VARIABLE_MAP)

    return data, meta


def read_cams_mcclear(filename):
    """
    Read a CAMS McClear file. CAMS McClear is described in [1]_.

    Parameters
    ----------
    filename: str
        Filename of a file containing data to read.

    Returns
    -------
    data : pandas.DataFrame
        timeseries data from CAMS McClear
    meta : dict
        metadata from CAMS McClear, see
        :func:`pvlib.iotools.parse_cams_mcclear` for fields

    See Also
    --------
    pvlib.iotools.parse_cams_mcclear, pvlib.iotools.get_cams_mcclear

    References
    ----------
    .. [1] `CAMS McClear Service Info
       <http://www.soda-pro.com/web-services/radiation/cams-mcclear/info>`_
    """
    with open(str(filename), 'r') as fbuf:
        content = parse_cams_mcclear(fbuf)
    return content
