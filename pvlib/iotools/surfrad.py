"""
Import functions for NOAA SURFRAD Data.
"""
import io

try:
    # python 2 compatibility
    from urllib2 import urlopen, Request
except ImportError:
    from urllib.request import urlopen, Request

import pandas as pd
import numpy as np

SURFRAD_COLUMNS = [
    'year', 'jday', 'month', 'day', 'hour', 'minute', 'dt', 'zen',
    'dw_solar', 'dw_solar_flag', 'uw_solar', 'uw_solar_flag', 'direct_n',
    'direct_n_flag', 'diffuse', 'diffuse_flag', 'dw_ir', 'dw_ir_flag',
    'dw_casetemp', 'dw_casetemp_flag', 'dw_dometemp', 'dw_dometemp_flag',
    'uw_ir', 'uw_ir_flag', 'uw_casetemp', 'uw_casetemp_flag', 'uw_dometemp',
    'uw_dometemp_flag', 'uvb', 'uvb_flag', 'par', 'par_flag', 'netsolar',
    'netsolar_flag', 'netir', 'netir_flag', 'totalnet', 'totalnet_flag',
    'temp', 'temp_flag', 'rh', 'rh_flag', 'windspd', 'windspd_flag',
    'winddir', 'winddir_flag', 'pressure', 'pressure_flag']


def read_surfrad(filename):
    """Read in a daily NOAA SURFRAD[1] file.

    Parameters
    ----------
    filename: str
        Filepath or url.

    Returns
    -------
    Tuple of the form (data, metadata).

    data: Dataframe
        Dataframe with the fields found in SURFRAD_COLUMNS.
    metadata: dict
        Site metadata included in the file.
    
    Notes
    -----
    Metadata includes the following fields
    =============== ====== ===============
    key             format description
    =============== ====== ===============
    station         String site name
    latitude        Float  site latitude
    longitude       Float  site longitude
    elevation       Int    site elevation
    surfrad_version Int    surfrad version


    References
    ----------
    [1] NOAA Earth System Research Laboratory Surface Radiation Budget Network
        `https://www.esrl.noaa.gov/gmd/grad/surfrad/index.html`
    """
    if filename.startswith('ftp'):
        req = Request(filename)
        response = urlopen(req)
        file_buffer = io.StringIO(response.read().decode(errors='ignore'))
    else:
        file_buffer = open(filename, 'r')

    # Read and parse the first two lines to build the metadata dict.
    station = file_buffer.readline()
    file_metadata = file_buffer.readline()

    metadata_list = file_metadata.split()
    metadata = {}
    metadata['name'] = station.strip()
    metadata['latitude'] = metadata_list[0]
    metadata['longitude'] = metadata_list[1]
    metadata['elevation'] = metadata_list[2]
    metadata['surfrad_version'] = metadata_list[-1]

    data = pd.read_csv(file_buffer, delim_whitespace=True,
                       header=None, names=SURFRAD_COLUMNS)
    file_buffer.close()

    data = format_index(data)
    missing = data == -9999.9
    data = data.where(~missing, np.NaN)

    return data, metadata


def format_index(data):
    """Create UTC localized DatetimeIndex for the dataframe.

    Parameters
    ----------
    data: Dataframe
        Must contain columns 'year', 'month', 'day', 'hour' and
        'minute'.

    Return
    ------
    data: Dataframe
        Dataframe with a DatetimeIndex localized to UTC.
    """
    index = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute']])
    data.index = index
    data.tz_localize('UTC')
    return data
