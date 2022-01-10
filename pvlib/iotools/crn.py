"""Functions to read data from the US Climate Reference Network (CRN).
"""

import pandas as pd
import numpy as np


HEADERS = (
    'WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE '
    'AIR_TEMPERATURE PRECIPITATION SOLAR_RADIATION SR_FLAG '
    'SURFACE_TEMPERATURE ST_TYPE ST_FLAG RELATIVE_HUMIDITY RH_FLAG '
    'SOIL_MOISTURE_5 SOIL_TEMPERATURE_5 WETNESS WET_FLAG WIND_1_5 WIND_FLAG'
)

VARIABLE_MAP = {
    'LONGITUDE': 'longitude',
    'LATITUDE': 'latitude',
    'AIR_TEMPERATURE': 'temp_air',
    'SOLAR_RADIATION': 'ghi',
    'SR_FLAG': 'ghi_flag',
    'RELATIVE_HUMIDITY': 'relative_humidity',
    'RH_FLAG': 'relative_humidity_flag',
    'WIND_1_5': 'wind_speed',
    'WIND_FLAG': 'wind_speed_flag'
}

# as specified in CRN README.txt file. excludes 1 space between columns
WIDTHS = [5, 8, 4, 8, 4, 6, 7, 7, 7, 7, 6, 1, 7, 1, 1, 5, 1, 7, 7, 5, 1, 6, 1]
# add 1 to make fields contiguous (required by pandas.read_fwf)
WIDTHS = [w + 1 for w in WIDTHS]
# no space after last column
WIDTHS[-1] -= 1

# specify dtypes for potentially problematic values
DTYPES = [
    'int64', 'int64', 'int64', 'int64', 'int64', 'str', 'float64', 'float64',
    'float64', 'float64', 'float64', 'int64', 'float64', 'O', 'int64',
    'float64', 'int64', 'float64', 'float64', 'int64', 'int64', 'float64',
    'int64'
]


def read_crn(filename, map_variables=True):
    """Read a NOAA USCRN fixed-width file into a pandas dataframe.

    The CRN network consists of over 100 meteorological stations covering the
    U.S. and is described in [1]_ and [2]_. The primary goal of CRN is to
    provide long-term measurements of temperature, precipitation, and soil
    moisture and temperature. Additionally, global horizontal irradiance (GHI)
    is measured at each site using a photodiode pyranometer.

    Parameters
    ----------
    filename: str, path object, or file-like
        filepath or url to read for the fixed-width file.
    map_variables: bool, default: True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable VARIABLE_MAP.

    Returns
    -------
    data: Dataframe
        A dataframe with DatetimeIndex and all of the variables in the
        file.

    Notes
    -----
    CRN files contain 5 minute averages labeled by the interval ending
    time. Here, missing data is flagged as NaN, rather than the lowest
    possible integer for a field (e.g. -999 or -99). Air temperature is in
    deg C and wind speed is in m/s at a height of 1.5 m above ground level.

    Variables corresponding to standard pvlib variables are by default renamed,
    e.g. `SOLAR_RADIATION` becomes `ghi`. See the
    `pvlib.iotools.crn.VARIABLE_MAP` dict for the complete mapping.

    CRN files occasionally have a set of null characters on a line
    instead of valid data. This function drops those lines. Sometimes
    these null characters appear on a line of their own and sometimes
    they occur on the same line as valid data. In the latter case, the
    valid data will not be returned. Users may manually remove the null
    characters and reparse the file if they need that line.

    References
    ----------
    .. [1] U.S. Climate Reference Network
       `https://www.ncdc.noaa.gov/crn/qcdatasets.html
       <https://www.ncdc.noaa.gov/crn/qcdatasets.html>`_

    .. [2] Diamond, H. J. et. al., 2013: U.S. Climate Reference Network
       after one decade of operations: status and assessment. Bull.
       Amer. Meteor. Soc., 94, 489-498. :doi:`10.1175/BAMS-D-12-00170.1`
    """

    # read in data. set fields with NUL characters to NaN
    data = pd.read_fwf(filename, header=None, names=HEADERS.split(' '),
                       widths=WIDTHS, na_values=['\x00\x00\x00\x00\x00\x00'])
    # at this point we only have NaNs from NUL characters, not -999 etc.
    # these bad rows need to be removed so that dtypes can be set.
    # NaNs require float dtype so we run into errors if we don't do this.
    data = data.dropna(axis=0)
    # loop here because dtype kwarg not supported in read_fwf until 0.20
    for (col, _dtype) in zip(data.columns, DTYPES):
        data[col] = data[col].astype(_dtype)

    # set index
    # UTC_TIME does not have leading 0s, so must zfill(4) to comply
    # with %H%M format
    dts = data[['UTC_DATE', 'UTC_TIME']].astype(str)
    dtindex = pd.to_datetime(dts['UTC_DATE'] + dts['UTC_TIME'].str.zfill(4),
                             format='%Y%m%d%H%M', utc=True)
    data = data.set_index(dtindex)

    # Now we can set nans. This could be done a per column basis to be
    # safer, since in principle a real -99 value could occur in a -9999
    # column. Very unlikely to see that in the real world.
    data = data.replace([-99, -999, -9999, -99999, -999999], np.nan)

    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)

    return data
