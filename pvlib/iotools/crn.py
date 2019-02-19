"""Functions to read data from the US Climate Reference Network (CRN).
"""

import pandas as pd
import numpy as np
from numpy import dtype


HEADERS = 'WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE AIR_TEMPERATURE PRECIPITATION SOLAR_RADIATION SR_FLAG SURFACE_TEMPERATURE ST_TYPE ST_FLAG RELATIVE_HUMIDITY RH_FLAG SOIL_MOISTURE_5 SOIL_TEMPERATURE_5 WETNESS WET_FLAG WIND_1_5 WIND_FLAG'  # noqa: E501

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

# specify dtypes for potentially problematic values
DTYPES = [
    dtype('int64'), dtype('int64'), dtype('int64'), dtype('int64'),
    dtype('int64'), dtype('int64'), dtype('float64'), dtype('float64'),
    dtype('float64'), dtype('float64'), dtype('float64'),
    dtype('int64'), dtype('float64'), dtype('O'), dtype('int64'),
    dtype('float64'), dtype('int64'), dtype('float64'),
    dtype('float64'), dtype('int64'), dtype('int64'), dtype('float64'),
    dtype('int64')
]


def read_crn(filename):
    """
    Read NOAA USCRN [1] fixed-width file into pandas dataframe.

    Parameters
    ----------
    filename: str
        filepath or url to read for the tsv file.

    Returns
    -------
    data: Dataframe
        A dataframe with datetime index and all of the variables listed
        in the `VARIABLE_MAP` dict inside of the map_columns function,
        along with their associated quality control flags.

    Notes
    -----
    CRN files contain 5 minute averages labeled by the interval ending
    time. Here, missing data is flagged as NaN, rather than the lowest
    possible integer for a field (e.g. -999 or -99).
    Air temperature in deg C.
    Wind speed in m/s at a height of 1.5 m above ground level.

    References
    ----------
    [1] U.S. Climate Reference Network
        `https://www.ncdc.noaa.gov/crn/qcdatasets.html <https://www.ncdc.noaa.gov/crn/qcdatasets.html>`_
    [2] Diamond, H. J. et. al., 2013: U.S. Climate Reference Network after
        one decade of operations: status and assessment. Bull. Amer.
        Meteor. Soc., 94, 489-498. :doi:`10.1175/BAMS-D-12-00170.1`
    """

    # read in data
    data = pd.read_fwf(filename, header=None, names=HEADERS.split(' '))
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
    try:
        # to_datetime(utc=True) does not work in older versions of pandas
        data = data.tz_localize('UTC')
    except TypeError:
        pass

    # set nans
    for val in [-99, -999, -9999]:
        data = data.where(data != val, np.nan)

    data = data.rename(columns=VARIABLE_MAP)

    return data
