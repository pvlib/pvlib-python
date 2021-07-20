"""Functions to read data from the Baseline Surface Radiation Network (BSRN).
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import pandas as pd
import gzip
import ftplib
import warnings
import io
import os

BSRN_FTP_URL = "ftp.bsrn.awi.de"

BSRN_LR0100_COL_SPECS = [(0, 3), (4, 9), (10, 16), (16, 22), (22, 27),
                         (27, 32), (32, 39), (39, 45), (45, 50), (50, 55),
                         (55, 64), (64, 70), (70, 75)]

BSRN_LR0300_COL_SPECS = [(0, 3), (4, 9), (10, 16), (16, 22), (22, 27)]

BSRN_LR0500_COL_SPECS = [(0, 3), (3, 8), (8, 14), (14, 20), (20, 26), (26, 32),
                         (32, 38), (38, 44), (44, 50), (50, 56), (56, 62),
                         (62, 68), (68, 74), (74, 80)]

BSRN_LR0100_COLUMNS = ['day', 'minute',
                       'ghi', 'ghi_std', 'ghi_min', 'ghi_max',
                       'dni', 'dni_std', 'dni_min', 'dni_max',
                       'empty', 'empty', 'empty', 'empty', 'empty',
                       'dhi', 'dhi_std', 'dhi_min', 'dhi_max',
                       'lwd', 'lwd_std', 'lwd_min', 'lwd_max',
                       'temp_air', 'relative_humidity', 'pressure']

BSRN_LR0300_COLUMNS = ['day', 'minute', 'upward short-wave reflected',
                       'upward long-wave', 'net radiation']

BSRN_LR0500_COLUMNS = ['day', 'minute', 'uva_global_mean', 'uva_global_std',
                       'uva_global_min', 'uva_global_max', 'uvb_direct_mean',
                       'uvb_direct_std', 'uvb_direct_min', 'uvb_direct_max',
                       'empty', 'empty', 'empty', 'empty',
                       'uvb_global_mean', 'uvb_global_std', 'uvb_global_min',
                       'uvb_global_max', 'uvb_diffuse_mean', 'uvb_diffuse_std',
                       'uvb_diffuse_mean', 'uvb_diffuse_std',
                       'uvb_diffuse_min', 'uvb_diffuse_max',
                       'uvb_reflect_mean', 'uvb_reflect_std',
                       'uvb_reflect_min', 'uvb_reflect_max']


def get_bsrn(start, end, station, username, password, logical_records=['0100'],
             local_path=None):
    """
    Retrieve ground measured irradiance data from the BSRN FTP server.

    The BSRN (Baseline Surface Radiation Network) is a world wide network
    of high-quality solar radiation monitoring stations as described in [1]_.
    Data is retrieved from the BSRN FTP server [2]_.

    Parameters
    ----------
    start: datetime-like
        First day of the requested period
    end: datetime-like
        Last day of the requested period
    station: str
        3-letter BSRN station abbreviation
    username: str
        username for accessing the BSRN FTP server
    password: str
        password for accessing the BSRN FTP server
    logical_records: list, default: ['0100']
        List of the logical records (LR) to parse. Options are: 0100, 0300,
        and 0500.
    local_path: str or path-like, default: None, optional
        If specified, path (abs. or relative) of where to save files

    Returns
    -------
    data: DataFrame
        timeseries data from the BSRN archive, see
        :func:`pvlib.iotools.read_bsrn` for fields
    metadata: dict
        metadata for the last available monthly file

    Raises
    ------
    KeyError
        If the specified station does not exist on the FTP server.

    Warning
    -------
    UserWarning
        If a requested file is missing a UserWarning is returned with the
        filename. Also, if no files match the specified station and timeframe.

    Notes
    -----
    Required username and password can be obtained for free as described in the
    BSRN's Data Release Guidelines [3]_.

    Currently only parsing of LR0100, LR0300, and LR0500 is supported. LR0100
    is contains the basic irradiance and auxillary measurements. See
    [4]_ for a description of the different logical records. Future updates may
    include parsing of additional data and metadata.

    Important
    ---------
    While data from the BSRN is generally of high-quality, measurement data
    should always be quality controlled before usage!

    Examples
    --------
    >>> # Retrieve two months irradiance data from the Cabauw BSRN station
    >>> data, metadata = pvlib.iotools.get_bsrn(  # doctest: +SKIP
    >>>     start=pd.Timestamp(2020,1,1), end=pd.Timestamp(2020,12,1),   # doctest: +SKIP
    >>>     station='cab', username='yourusername', password='yourpassword')  # doctest: +SKIP

    See Also
    --------
    pvlib.iotools.read_bsrn, pvlib.iotools.parse_bsrn

    References
    ----------
    .. [1] `World Radiation Monitoring Center - Baseline Surface Radiation
        Network (BSRN)
        <https://bsrn.awi.de/>`_
    .. [2] `BSRN Data Retrieval via FTP
       <https://bsrn.awi.de/data/data-retrieval-via-ftp/>`_
    .. [4] `BSRN Data Release Guidelines
       <https://bsrn.awi.de/data/conditions-of-data-release/>`_
    .. [3] `Update of the Technical Plan for BSRN Data Management, 2013,
       Global Climate Observing System (GCOS) GCOS-174.
       <https://bsrn.awi.de/fileadmin/user_upload/bsrn.awi.de/Publications/gcos-174.pdf>`_
    """  # noqa: E501
    # The FTP server uses lowercase station abbreviations
    station = station.lower()

    # Generate list files to download based on start/end (SSSMMYY.dat.gz)
    filenames = pd.date_range(start, end + pd.DateOffset(months=1), freq='1M')\
        .strftime(f"{station}%m%y.dat.gz").tolist()

    # Create FTP connection
    with ftplib.FTP(BSRN_FTP_URL, username, password) as ftp:
        # Change to station sub-directory (checks that the station exists)
        try:
            ftp.cwd(f'/{station}')
        except ftplib.error_perm as e:
            raise KeyError('Station sub-directory does not exist. Specified '
                           'station is probably not a proper three letter '
                           'station abbreviation.') from e
        dfs = []  # Initialize list for monthly dataframes
        for filename in filenames:
            try:
                bio = io.BytesIO()  # Initialize BytesIO object
                # Retrieve binary file from server and write to BytesIO object
                response = ftp.retrbinary(f'RETR {filename}', bio.write)
                # Check that transfer was successfull
                if not response.startswith('226 Transfer complete'):
                    raise ftplib.Error(response)
                # Decompress/unzip and decode the binary file
                text = gzip.decompress(bio.getvalue()).decode('utf-8')
                # Convert string to StringIO and parse data
                dfi, metadata = parse_bsrn(io.StringIO(text), logical_records)
                dfs.append(dfi)
                # Save file locally if local_path is specified
                if local_path is not None:
                    # Create local file
                    with open(os.path.join(local_path, filename), 'wb') as f:
                        f.write(bio.getbuffer())  # Write local file
            # FTP client raises an error if the file does not exist on server
            except ftplib.error_perm as e:
                if str(e) == '550 Failed to open file.':
                    warnings.warn(f'File: {filename} does not exist')
                else:
                    raise ftplib.error_perm(e)
        ftp.quit()  # Close and exit FTP connection

    # Concatenate monthly dataframes to one dataframe
    if len(dfs):
        data = pd.concat(dfs, axis='rows')
    else:  # Return empty dataframe
        data = pd.DataFrame(columns=BSRN_LR0100_COLUMNS)
        metadata = {}
        warnings.warn('No files were available for the specified timeframe.')
    # Return dataframe and metadata (metadata belongs to last available file)
    return data, metadata


def parse_bsrn(fbuf, logical_records=['0100']):
    """
    Parse a file-like buffer of a BSRN station-to-archive file.

    Parameters
    ----------
    fbuf: file-like buffer
        Buffer of a BSRN station-to-archive data file
    logical_records: list, default: ['0100']
        List of the logical records (LR) to parse. Options are: 0100, 0300,
        and 0500.

    Returns
    -------
    data: DataFrame
        A DataFrame containing time-series measurement data. See
        pvlib.iotools.read_bsrn for fields.
    metadata: dict
        Dictionary containing metadata (primarily from LR0004).

    See Also
    --------
    pvlib.iotools.read_bsrn, pvlib.iotools.get_bsrn

    """
    # Parse metadata
    fbuf.readline()  # first line should be *U0001, so read it and discard
    date_line = fbuf.readline()  # second line contains important metadata
    start_date = pd.Timestamp(year=int(date_line[7:11]),
                              month=int(date_line[3:6]), day=1,
                              tz='UTC')  # BSRN timestamps are UTC

    metadata = {}  # Initilize dictionary containing metadata
    metadata['start date'] = start_date
    metadata['station identification number'] = int(date_line[:3])
    metadata['version of data'] = int(date_line.split()[-1])
    for line in fbuf:
        if line[2:6] == '0004':  # stop once LR0004 has been reached
            break
        elif line == '':
            raise ValueError('Mandatatory record LR0004 not found.')
    metadata['date when station description changed'] = fbuf.readline().strip()
    metadata['surface type'] = int(fbuf.readline(3))
    metadata['topography type'] = int(fbuf.readline())
    metadata['address'] = fbuf.readline().strip().strip()
    metadata['telephone no. of station'] = fbuf.readline(20).strip()
    metadata['FAX no. of station'] = fbuf.readline().strip()
    metadata['TCP/IP no. of station'] = fbuf.readline(15).strip()
    metadata['e-mail address of station'] = fbuf.readline().strip()
    metadata['latitude'] = float(fbuf.readline(8))
    metadata['longitude'] = float(fbuf.readline(8))
    metadata['altitude'] = int(fbuf.readline(5))
    metadata['identification of "SYNOP" station'] = fbuf.readline().strip()
    metadata['date when horizon changed'] = fbuf.readline().strip()
    # Pass last section of LR0004 containing the horizon elevation data
    horizon = []  # list for raw horizon elevation data
    while True:
        line = fbuf.readline()
        if ('*' in line) | (line == ''):
            break
        else:
            horizon += [int(i) for i in line.split()]
    metadata['horizon'] = pd.Series(horizon[1::2], horizon[::2]).sort_index().drop(-1)  # noqa: E501

    # Read file and store the starting line number and number of lines for
    # each logical record (LR)
    fbuf.seek(0)  # reset buffer to start of file
    lr_startrow = {}  # Dictionary of starting line number for each LR
    lr_nrows = {}  # Dictionary of end line number for each LR
    for num, line in enumerate(fbuf):
        if line.startswith('*'):  # Find start of all logical records
            if len(lr_startrow) >= 1:
                lr_nrows[lr] = num - max(lr_startrow.values())-1  # noqa: F821
            lr = line[2:6]  # string of 4 digit LR number
            lr_startrow[lr] = num
    lr_nrows[lr] = num - lr_startrow[lr]

    for lr in list(logical_records):
        if lr not in ['0100', '0300', '0500']:
            raise ValueError(f"Logical record {lr} not in "
                             f"{['0100', '0300','0500']}.")
    dfs = []  # Initialize empty list for dataframe

    # Parse LR0100 - basic measurements including GHI, DNI, DHI and temperature
    if ('0100' in lr_startrow.keys()) & ('0100' in logical_records):
        fbuf.seek(0)  # reset buffer to start of file
        LR_0100 = pd.read_fwf(fbuf, skiprows=lr_startrow['0100'] + 1,
                              nrows=lr_nrows['0100'], header=None,
                              colspecs=BSRN_LR0100_COL_SPECS,
                              na_values=[-999.0, -99.9])
        # Create multi-index and unstack, resulting in 1 col for each variable
        LR_0100 = LR_0100.set_index([LR_0100.index // 2, LR_0100.index % 2])
        LR_0100 = LR_0100.unstack(level=1).swaplevel(i=0, j=1, axis='columns')
        # Sort columns to match original order and assign column names
        LR_0100 = LR_0100.reindex(sorted(LR_0100.columns), axis='columns')
        LR_0100.columns = BSRN_LR0100_COLUMNS
        # Set datetime index
        LR_0100.index = (start_date + pd.to_timedelta(LR_0100['day']-1, unit='d')  # noqa: E501
                         + pd.to_timedelta(LR_0100['minute'], unit='T'))
        # Drop empty, minute, and day columns
        LR_0100 = LR_0100.drop(columns=['empty', 'day', 'minute'])
        dfs.append(LR_0100)

    # Parse LR0300 - other time series data, including upward and net radiation
    if ('0300' in lr_startrow.keys()) & ('0300' in logical_records):
        fbuf.seek(0)  # reset buffer to start of file
        LR_0300 = pd.read_fwf(fbuf, skiprows=lr_startrow['0300']+1,
                              nrows=lr_nrows['0300'], header=None,
                              na_values=[-999.0, -99.9],
                              colspecs=BSRN_LR0300_COL_SPECS,
                              names=BSRN_LR0300_COLUMNS)
        LR_0300.index = (start_date
                         + pd.to_timedelta(LR_0300['day']-1, unit='d')
                         + pd.to_timedelta(LR_0300['minute'], unit='T'))
        LR_0300 = LR_0300.drop(columns=['day', 'minute']).astype(float)
        dfs.append(LR_0300)

    # Parse LR0500 - UV measurements
    if ('0500' in lr_startrow.keys()) & ('0500' in logical_records):
        fbuf.seek(0)  # reset buffer to start of file
        LR_0500 = pd.read_fwf(fbuf, skiprows=lr_startrow['0500']+1,
                              nrows=lr_nrows['0500'], na_values=[-99.9],
                              header=None, colspecs=BSRN_LR0500_COL_SPECS)
        # Create multi-index and unstack, resulting in 1 col for each variable
        LR_0500 = LR_0500.set_index([LR_0500.index // 2, LR_0500.index % 2])
        LR_0500 = LR_0500.unstack(level=1).swaplevel(i=0, j=1, axis='columns')
        # Sort columns to match original order and assign column names
        LR_0500 = LR_0500.reindex(sorted(LR_0500.columns), axis='columns')
        LR_0500.columns = BSRN_LR0500_COLUMNS
        LR_0500.index = (start_date
                         + pd.to_timedelta(LR_0500['day']-1, unit='d')
                         + pd.to_timedelta(LR_0500['minute'], unit='T'))
        LR_0500 = LR_0500.drop(columns=['empty', 'day', 'minute'])
        dfs.append(LR_0500)

    data = pd.concat(dfs, axis='columns')
    return data, metadata


def read_bsrn(filename, logical_records=['0100']):
    """
    Read a BSRN station-to-archive file into a DataFrame.

    The BSRN (Baseline Surface Radiation Network) is a world wide network
    of high-quality solar radiation monitoring stations as described in [1]_.
    The function is able to parse LR0100, LR0300, and LR0500. LR0100 contains
    the basic measurements, which include global, diffuse, and direct
    irradiance, as well as downwelling long-wave radiation [2]_. Future updates
    may include parsing of additional data and metadata.

    BSRN files are freely available and can be accessed via FTP [3]_. Required
    username and password are easily obtainable as described in the BSRN's
    Data Release Guidelines [4]_.

    Parameters
    ----------
    filename: str or path-like
        Name or path of a BSRN station-to-archive data file
    logical_records: list, default: ['0100']
        List of the logical records (LR) to parse. Options are: 0100, 0300,
        and 0500.

    Returns
    -------
    data: DataFrame
        A DataFrame with the columns as described below. For more extensive
        description of the variables, consult [2]_.
    metadata: dict
        Dictionary containing metadata (primarily from LR0004).

    Notes
    -----
    The data DataFrame for LR0100 includes the following fields:

    =======================  ======  ==========================================
    Key                      Format  Description
    =======================  ======  ==========================================
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

    For fields for other logical records, see [2]_.

    See Also
    --------
    pvlib.iotools.parse_bsrn, pvlib.iotools.get_bsrn

    References
    ----------
    .. [1] `World Radiation Monitoring Center - Baseline Surface Radiation
        Network (BSRN)
        <https://bsrn.awi.de/>`_
    .. [2] `Update of the Technical Plan for BSRN Data Management, 2013,
       Global Climate Observing System (GCOS) GCOS-174.
       <https://bsrn.awi.de/fileadmin/user_upload/bsrn.awi.de/Publications/gcos-174.pdf>`_
    .. [3] `BSRN Data Retrieval via FTP
       <https://bsrn.awi.de/data/data-retrieval-via-ftp/>`_
    .. [4] `BSRN Data Release Guidelines
       <https://bsrn.awi.de/data/conditions-of-data-release/>`_
    """
    if str(filename).endswith('.gz'):  # check if file is a gzipped (.gz) file
        open_func, mode = gzip.open, 'rt'
    else:
        open_func, mode = open, 'r'
    with open_func(filename, mode) as f:
        return parse_bsrn(f, logical_records)
