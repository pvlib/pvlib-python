"""Collection of functions to operate on data from University of Oregon Solar
Radiation Monitoring Laboratory (SRML) data.
"""
import numpy as np
import pandas as pd


# VARIABLE_MAP is a dictionary mapping SRML data element numbers to their
# pvlib names. For most variables, only the first three digits are used,
# the fourth indicating the instrument. Spectral data (7xxx) uses all
# four digits to indicate the variable. See a full list of data element
# numbers `here. <http://solardat.uoregon.edu/DataElementNumbers.html>`_

VARIABLE_MAP = {
    '100': 'ghi',
    '201': 'dni',
    '300': 'dhi',
    '920': 'wind_dir',
    '921': 'wind_speed',
    '930': 'temp_air',
    '931': 'temp_dew',
    '933': 'relative_humidity',
    '937': 'temp_cell',
}


def read_srml(filename):
    """
    Read University of Oregon SRML 1min .tsv file into pandas dataframe.  The
    SRML is described in [1]_.

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
    The time index is shifted back by one interval to account for the
    daily endtime of 2400, and to avoid time parsing errors on leap
    years. The returned data values are labeled by the left endpoint of
    interval, and should be understood to occur during the interval from
    the time of the row until the time of the next row. This is consistent
    with pandas' default labeling behavior.

    See SRML's `Archival Files`_ page for more information.

    .. _Archival Files: http://solardat.uoregon.edu/ArchivalFiles.html

    References
    ----------
    .. [1] University of Oregon Solar Radiation Monitoring Laboratory
       `http://solardat.uoregon.edu/ <http://solardat.uoregon.edu/>`_
    """
    tsv_data = pd.read_csv(filename, delimiter='\t')
    data = format_index(tsv_data)
    # Drop day of year and time columns
    data = data[data.columns[2:]]

    data = data.rename(columns=map_columns)

    # Quality flag columns are all labeled 0 in the original data. They
    # appear immediately after their associated variable and are suffixed
    # with an integer value when read from the file. So we map flags to
    # the preceding variable with a '_flag' suffix.
    #
    # Example:
    #   Columns ['ghi_0', '0.1', 'temp_air_2', '0.2']
    #
    #   Yields a flag_label_map of:
    #       { '0.1': 'ghi_0_flag',
    #         '0.2': 'temp_air_2'}
    #
    columns = data.columns
    flag_label_map = {flag: columns[columns.get_loc(flag) - 1] + '_flag'
                      for flag in columns[1::2]}
    data = data.rename(columns=flag_label_map)

    # Mask data marked with quality flag 99 (bad or missing data)
    for col in columns[::2]:
        missing = data[col + '_flag'] == 99
        data[col] = data[col].where(~(missing), np.NaN)
    return data


def map_columns(col):
    """Map data element numbers to pvlib names.

    Parameters
    ----------
    col: str
        Column label to be mapped.

    Returns
    -------
    str
        The pvlib label if it was found in the mapping,
        else the original label.
    """
    if col.startswith('7'):
        # spectral data
        try:
            return VARIABLE_MAP[col]
        except KeyError:
            return col
    try:
        variable_name = VARIABLE_MAP[col[:3]]
        variable_number = col[3:]
        return variable_name + '_' + variable_number
    except KeyError:
        return col


def format_index(df):
    """Create a datetime index from day of year, and time columns.

    Parameters
    ----------
    df: pd.Dataframe
        The srml data to reindex.

    Returns
    -------
    df: pd.Dataframe
        The Dataframe with a DatetimeIndex localized to 'Etc/GMT+8'.
    """
    # Name of the second column indicates the year of the file, but
    # the column contains times.
    year = int(df.columns[1])
    df_doy = df[df.columns[0]]
    # Times are expressed as integers from 1-2400, we convert to 0-2359 by
    # subracting the length of one interval and then correcting the times
    # at each former hour. interval_length is determined by taking the
    # difference of the first two rows of the time column.
    # e.g. The first two rows of hourly data are 100 and 200
    #      so interval_length is 100.
    interval_length = df[df.columns[1]][1] - df[df.columns[1]][0]
    df_time = df[df.columns[1]] - interval_length
    if interval_length == 100:
        # Hourly files do not require fixing the former hour timestamps.
        times = df_time
    else:
        # Because hours are represented by some multiple of 100, shifting
        # results in invalid values.
        #
        # e.g. 200 (for 02:00) shifted by 15 minutes becomes 185, the
        #      desired result is 145 (for 01:45)
        #
        # So we find all times with minutes greater than 60 and remove 40
        # to correct to valid times.
        old_hours = df_time % 100 > 60
        times = df_time.where(~old_hours, df_time - 40)
    times = times.apply(lambda x: '{:04.0f}'.format(x))
    doy = df_doy.apply(lambda x: '{:03.0f}'.format(x))
    dts = pd.to_datetime(str(year) + '-' + doy + '-' + times,
                         format='%Y-%j-%H%M')
    df.index = dts
    df = df.tz_localize('Etc/GMT+8')
    return df


def read_srml_month_from_solardat(station, year, month, filetype='PO'):
    """Request a month of SRML data from solardat and read it into
    a Dataframe.  The SRML is described in [1]_.

    Parameters
    ----------
    station: str
        The name of the SRML station to request.
    year: int
        Year to request data for
    month: int
        Month to request data for.
    filetype: string
        SRML file type to gather. See notes for explanation.

    Returns
    -------
    data: pd.DataFrame
        One month of data from SRML.

    Notes
    -----
    File types designate the time interval of a file and if it contains
    raw or processed data. For instance, `RO` designates raw, one minute
    data and `PO` designates processed one minute data. The availability
    of file types varies between sites. Below is a table of file types
    and their time intervals. See [1] for site information.

    ============= ============ ==================
    time interval raw filetype processed filetype
    ============= ============ ==================
    1 minute      RO           PO
    5 minute      RF           PF
    15 minute     RQ           PQ
    hourly        RH           PH
    ============= ============ ==================

    References
    ----------
    .. [1] University of Oregon Solar Radiation Measurement Laboratory
       `http://solardat.uoregon.edu/ <http://solardat.uoregon.edu/>`_
    """
    file_name = "{station}{filetype}{year:02d}{month:02d}.txt".format(
        station=station,
        filetype=filetype,
        year=year % 100,
        month=month)
    url = "http://solardat.uoregon.edu/download/Archive/"
    data = read_srml(url + file_name)
    return data
