"""Collection of functions to get data from UO SRML.
"""
import numpy as np
import pandas as pd


def read_srml(filename):
    """
    Read SRML file into pandas dataframe.

    Parameters
    ----------
    filename: str
        filepath or url to read for the tsv file.

    Returns
    -------
    data: Dataframe
        A dataframe with datetime index and all of the variables listed
        in the `var_map` dict inside of the map_columns function, along
        with their associated quality control flags.

    Notes
    -----
    Note that the time index is shifted back one minute to account for
    2400 hours, and to avoid time parsing errors on leap years. Data values
    on a given line should now be understood to occur during the interval
    extending from the time of the line in which they are listed to
    the ending time on the next line, rather than the previous line.
    """
    tsv_data = pd.read_csv(filename, delimiter='\t')
    year = tsv_data.columns[1]
    data = format_index(tsv_data, year)
    # Rename and drop datetime columns
    data = data[data.columns[2:]].rename(columns=map_columns)

    # Quality flags are all labeled 0, but occur immediately after their
    # associated var so we create a dict mapping them to var_flag for renaming
    flag_label_map = {flag: data.columns[data.columns.get_loc(flag)-1]+'_flag'
                      for flag in data.columns[1::2]}
    data = data.rename(columns=flag_label_map)
    # For data flagged bad or missing, replace the value with np.NaN
    for col in data.columns[::2]:
        data[col] = data[col].where(~(data[col+'_flag'] == 99), np.NaN)
    return data


def map_columns(col):
    """Map column labels to pvlib names.

    Parameters
    ----------
    col: str
        Column label to be mapped.

    Returns
    -------
    str
        The pvlib label if it was found in the mapping,
        else the original label.

    Notes
    -----
    var_map is a dictionary mapping SRML data element numbers
    to their pvlib names. For most variables, only the first
    three numbers are used, the fourth indicating the instrument.
    Spectral data (7xxx) uses all four numbers to indicate the
    variable.
    """
    var_map = {
        '100': 'ghi',
        '201': 'dni',
        '300': 'dhi',
        '930': 'temp_air',
        '931': 'temp_dew',
        '933': 'relative_humidity',
        '921': 'wind_speed',
        '920': 'wind_dir',
    }
    if col.startswith('7'):
        # spectral data
        try:
            return var_map[col]
        except KeyError:
            return col
    try:
        return var_map[col[:3]]+'_'+col[3:]
    except KeyError:
        return col


def format_index(df, year):
    """ Create a datetime index from day of year, and time columns.

    Parameters
    ----------
    df: pd.Dataframe
        The srml data to reindex.
    year: int
        The year of the file

    Returns
    -------
    df: pd.Dataframe
        The Dataframe with a datetime index applied.
    """
    df_time = df[df.columns[1]] - 1
    df_doy = df[df.columns[0]]
    hours = df_time % 100 == 99
    times = df_time.where(~hours, df_time - 40)
    times = times.apply(lambda x: '{:04.0f}'.format(x))
    doy = df_doy.apply(lambda x: '{:03.0f}'.format(x))
    dts = pd.to_datetime(str(year) + '-' + doy + '-' + times,
                         format='%Y-%j-%H%M')
    df.index = dts
    df = df.tz_localize('Etc/GMT+8')
    return df


def request_uo_data(station, year, month, filetype='PO'):
    """Read a month of SRML data from solardat into a Dataframe.

    Parameters
    ----------
    station: str
        The name of the SRML station to request.
    year: int
        Year to request data for
    month: int
        Month to request data for.
    filetype: string
        SRML file type to gather. 'RO' and 'PO' are the
        only minute resolution files.
    Returns
    -------
    data: pd.DataFrame
        One month of data from SRML.
    """
    file_name = "{station}{filetype}{year:2d}{month:2d}.txt".format(
        station=station,
        filetype=filetype,
        year=year % 100,
        month=month)
    url = "http://solardat.uoregon.edu/download/Archive/"
    data = read_srml(url+file_name)
    return data
