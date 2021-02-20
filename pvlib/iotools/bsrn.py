"""Functions to read data from the Baseline Surface Radiation Network (BSRN).
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import pandas as pd
import gzip

COL_SPECS = [(0, 3), (4, 9), (10, 16), (16, 22), (22, 27), (27, 32), (32, 39),
             (39, 45), (45, 50), (50, 55), (55, 64), (64, 70), (70, 75)]

BSRN_COLUMNS = ['day', 'minute',
                'ghi', 'ghi_std', 'ghi_min', 'ghi_max',
                'dni', 'dni_std', 'dni_min', 'dni_max',
                'empty', 'empty', 'empty', 'empty', 'empty',
                'dhi', 'dhi_std', 'dhi_min', 'dhi_max',
                'lwd', 'lwd_std', 'lwd_min', 'lwd_max',
                'temp_air', 'relative_humidity', 'pressure']


def read_bsrn(filename):
    """
    Read a BSRN station-to-archive file into a DataFrame.

    The BSRN (Baseline Surface Radiation Network) is a world wide network
    of high-quality solar radiation monitoring stations as described in [1]_.
    The function only parses the basic measurements (LR0100), which include
    global, diffuse, direct and downwelling long-wave radiation [2]_. Future
    updates may include parsing of additional data and meta-data.

    BSRN files are freely available and can be accessed via FTP [3]_. Required

    username and password are easily obtainable as described in the BSRN's
    Data Release Guidelines [4]_.



    Parameters
    ----------
    filename: str
        A relative or absolute file path.

    Returns
    -------
    data: DataFrame
        A DataFrame with the columns as described below. For more extensive
        description of the variables, consult [2]_.

    Notes
    -----
    The data DataFrame includes the following fields:

    =======================  ======  ==========================================
    Key                      Format  Description
    =======================  ======  ==========================================
    day                      int     Day of the month 1-31
    minute                   int     Minute of the day 0-1439
    ghi                      float   Mean global horizontal irradiance [W/m^2]
    ghi_std                  float   Std. global horizontal irradiance [W/m^2]
    ghi_min                  float   Min. global horizontal irradiance [W/m^2]
    ghi_max                  float   Max. global horizontal irradiance [W/m^2]
    dni                      float   Mean direct normal irradiance [W/m^2]
    dni_std                  float   Std. direct normal irradiance [W/m^2]
    dni_min                  float   Min. direct normal irradiance [W/m^2]
    dni_max                  float   Max. direct normal irradiance [W/m^2]
    dhi                      float   Mean diffuse horizontal irradiance [W/m^2]
    dhi_std                  float   Std. diffuse horizontal irradiance [W/m^2]
    dhi_min                  float   Min. diffuse horizontal irradiance [W/m^2]
    dhi_max                  float   Max. diffuse horizontal irradiance [W/m^2]
    lwd                      float   Mean. downward long-wave radiation [W/m^2]
    lwd_std                  float   Std. downward long-wave radiation [W/m^2]
    lwd_min                  float   Min. downward long-wave radiation [W/m^2]
    lwd_max                  float   Max. downward long-wave radiation [W/m^2]
    temp_air                 float   Air temperature [°C]
    relative_humidity        float   Relative humidity [%]
    pressure                 float   Atmospheric pressure [hPa]
    =======================  ======  ==========================================

    References
    ----------
    .. [1] `World Radiation Monitoring Center - Baseline Surface Radiation
        Network (BSRN)
        <https://bsrn.awi.de/>`_
    .. [2] `Update of the Technical Plan for BSRN Data Management, 2013,
       Global Climate Observing System (GCOS) GCOS-172.
       <https://bsrn.awi.de/fileadmin/user_upload/bsrn.awi.de/Publications/gcos-174.pdf>`_
    .. [3] `BSRN Data Retrieval via FTP
       <https://bsrn.awi.de/data/data-retrieval-via-ftp/>`_
    .. [4] `BSRN Data Release Guidelines
       <https://bsrn.awi.de/data/conditions-of-data-release/>`_
    """

    # Read file and store the starting line number for each logical record (LR)
    line_no_dict = {}
    if str(filename).endswith('.gz'):  # check if file is a gzipped (.gz) file
        open_func, mode = gzip.open, 'rt'
    else:
        open_func, mode = open, 'r'
    with open_func(filename, mode) as f:
        f.readline()  # first line should be *U0001, so read it and discard
        line_no_dict['0001'] = 0
        date_line = f.readline()  # second line contains the year and month
        start_date = pd.Timestamp(year=int(date_line[7:11]),
                                  month=int(date_line[3:6]), day=1,
                                  tz='UTC')  # BSRN timestamps are UTC
        for num, line in enumerate(f, start=2):
            if line.startswith('*'):  # Find start of all logical records
                line_no_dict[line[2:6]] = num  # key is 4 digit LR number

    # Determine start and end line of logical record LR0100 to be parsed
    start_row = line_no_dict['0100'] + 1  # Start line number
    # If LR0100 is the last logical record, then read rest of file
    if start_row-1 == max(line_no_dict.values()):
        end_row = num  # then parse rest of the file
    else:  # otherwise parse until the beginning of the next logical record
        end_row = min([i for i in line_no_dict.values() if i > start_row]) - 1
    nrows = end_row-start_row+1

    # Read file as a fixed width file (fwf)
    data = pd.read_fwf(filename, skiprows=start_row, nrows=nrows, header=None,
                       colspecs=COL_SPECS, na_values=[-999.0, -99.9],
                       compression='infer')

    # Create multi-index and unstack, resulting in one column for each variable
    data = data.set_index([data.index // 2, data.index % 2])
    data = data.unstack(level=1).swaplevel(i=0, j=1, axis='columns')

    # Sort columns to match original order and assign column names
    data = data.reindex(sorted(data.columns), axis='columns')
    data.columns = BSRN_COLUMNS
    # Drop empty columns
    data = data.drop('empty', axis='columns')

    # Change day and minute type to integer
    data['day'] = data['day'].astype('Int64')
    data['minute'] = data['minute'].astype('Int64')

    # Set datetime index
    data.index = (start_date
                  + pd.to_timedelta(data['day']-1, unit='d')
                  + pd.to_timedelta(data['minute'], unit='T'))

    return data
