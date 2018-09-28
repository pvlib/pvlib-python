"""Collection of functions to operate on data from University of Oregon Solar
Radiation Monitoring Laboratory (SRML) data.
"""
import numpy as np
import pandas as pd

"""
variable_map is a dictionary mapping SRML data element numbers to their
pvlib names. For most variables, only the first three digits are used,
the fourth indicating the instrument. Spectral data (7xxx) uses all
four digits to indicate the variable. See a full list of data element
numbers `here. <http://solardat.uoregon.edu/DataElementNumbers.html>`_
"""
variable_map = {
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
    Read University of Oregon SRML[1] 1min .tsv file into pandas dataframe.

    Parameters
    ----------
    filename: str
        filepath or url to read for the tsv file.

    Returns
    -------
    data: Dataframe
        A dataframe with datetime index and all of the variables listed
        in the `variable_map` dict inside of the map_columns function,
        along with their associated quality control flags.

    Notes
    -----
    Note that the time index is shifted back one minute to account for
    2400 hours, and to avoid time parsing errors on leap years. Data values
    on a given line should now be understood to occur during the interval
    extending from the time of the line in which they are listed to
    the ending time on the next line, rather than the previous line.

    See SRML's `Archival Files`_ page for more information.

    .. _Archival Files: http://solardat.uoregon.edu/ArchivalFiles.html

    References
    ----------

    [1] University of Oregon Solar Radiation Monitoring Laboratory
        `http://solardat.uoregon.edu/ <http://solardat.uoregon.edu/>`_
    """
    tsv_data = pd.read_csv(filename, delimiter='\t')
    year = tsv_data.columns[1]
    data = format_index(tsv_data, year)
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
            return variable_map[col]
        except KeyError:
            return col
    try:
        variable_name = variable_map[col[:3]]
        variable_number = col[3:]
        return variable_name + '_' + variable_number
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
    fifty_nines = df_time % 100 == 99
    times = df_time.where(~fifty_nines, df_time - 40)
    times = times.apply(lambda x: '{:04.0f}'.format(x))
    doy = df_doy.apply(lambda x: '{:03.0f}'.format(x))
    dts = pd.to_datetime(str(year) + '-' + doy + '-' + times,
                         format='%Y-%j-%H%M')
    df.index = dts
    df = df.tz_localize('Etc/GMT+8')
    return df


def read_srml_month_from_solardat(station, year, month, filetype='PO'):
    """Request a month of SRML[1] data from solardat and read it into
    a Dataframe.

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

    References
    ----------
    [1] University of Oregon Solar Radiation Measurement Laboratory
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
