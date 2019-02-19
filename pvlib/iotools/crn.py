"""Functions to read data from the US Climate Reference Network (CRN).
"""

import pandas as pd


HEADERS = 'WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE AIR_TEMPERATURE PRECIPITATION SOLAR_RADIATION SR_FLAG SURFACE_TEMPERATURE ST_TYPE ST_FLAG RELATIVE_HUMIDITY RH_FLAG SOIL_MOISTURE_5 SOIL_TEMPERATURE_5 WETNESS WET_FLAG WIND_1_5 WIND_FLAG'  # noqa: E501


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
    possible integer for a field (e.g. -999 or -99)

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

    # set index
    # UTC_TIME does not have leading 0s, so must zfill(4) to comply
    # with %H%M format
    dts = data[['UTC_DATE', 'UTC_TIME']].astype(str)
    dtindex = pd.to_datetime(dts['UTC_DATE'] + dts['UTC_TIME'].str.zfill(4),
                             format='%Y%m%d%H%M', utc=True)
    data = data.set_index(dtindex)

    # set nans

    return data
