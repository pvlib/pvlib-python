"""Functions to read data from the Baseline Surface Radiation Network (BSRN).
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import gzip
from collections import OrderedDict
import pandas as pd
import os


def read_bsrn(filename):
    """
    Read a BSRN station-to-archive file into a DataFrame. 
    
    The BSRN (Baseline Surface Radiation Network) is a world wide network 
    of high-quality solar radiation monitoring stations as described in [1]_.
    The function only parses the basic measurements (LR0100), which include
    global, diffuse, direct and downwelling long-wave radiation [2]_. Future
    updates may include parsing of additional data and meta-data.
    
    Required username and password are easily obtainable by writing an email to
    Amelie Driemel (Amelie.Driemel@awi.de) [3]_ on condition that users follow 
    BSRN's Data Release Guidelines [4]_.

    
    Parameters
    ----------
    filename: str
        A relative or absolute file path.
    
    Returns
    -------
    Tuple of the form (data, metadata).

    data: Dataframe
        A Dataframe with the columns as described below. For more extensive
        description of the variables, consult [2]_.
        
    metadata: dict
        Site metadata included in the file.
    
    Notes
    -----
    Metadata dictionary includes the following fields:

    ===============  ======  ===============
    Key              Format  Description
    ===============  ======  ===============
    station          String  site name
    latitude         Float   site latitude
    longitude        Float   site longitude
    elevation        Int     site elevation
    ===============  ======  ===============

    The data Dataframe includes the following fields:

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
    air_temperature          float   Air temperature [°C]
    relative_humidity        float   Relative humidity [%]
    pressure                 float   Atmospheric pressure [hPa]
    =======================  ======  ==========================================

    References
    ----------
    .. [1] `World Radiation Monitoring Center - Baseline Surface Radiation Network (BSRN)
       `BSRN homepage <https:/https://bsrn.awi.de/>`_
    .. [2] `Update of the Technical Plan for BSRN Data Management, October 2013,
       Global Climate Observing System (GCOS) GCOS-172.
       <https://bsrn.awi.de/fileadmin/user_upload/bsrn.awi.de/Publications/gcos-174.pdf>`_
    .. [3] `BSRN Data Retrieval via FTP 
       <https://bsrn.awi.de/data/data-retrieval-via-ftp/>`_
    .. [4] `BSRN Data Release Guidelines
       <https://bsrn.awi.de/data/conditions-of-data-release/>`_
       


    """
    
    # Read file and store the starting line number for each each section
    line_no_dict = OrderedDict()
    if str(filename).endswith('.gz'): # if file is a gzipped (.gz) file
        with gzip.open(filename,'rt') as f:
            for num, line in enumerate(f):
                if ('*U' in line) or ('*C' in line):
                    line_no_dict[line.splitlines()[0]] = num
    else:
        with open(filename, 'r') as f:
            for num, line in enumerate(f):
                if ('*U' in line) or ('*C' in line):
                    line_no_dict[line.splitlines()[0]] = num
                    
    # Get line numbers for the data set
    line_no_dict_keys = list(line_no_dict.keys())
    data_id = [k for k in line_no_dict_keys if ('*C0100' in k) or ('*U0100' in k)][0] # tag for start of data sets, either *C0100 or *U0100
    start_row = line_no_dict[data_id] + 1 # First line number of data
    if data_id == line_no_dict_keys[-1]: # check if the dataset is the last dataset
        end_row = num
    else:
        end_row = line_no_dict[line_no_dict_keys[line_no_dict_keys.index(data_id)+1]] # Last line number of data # should there be -1?
    nrows = end_row-start_row

    # Read file as a fixed width file (fwf)
    COLSPECS = [(0,3),(4,9),(10,16),(17,22),(23,27),(28,32),(33,39),(40,45),
                (46,50),(51,55),(56,64),(65,70),(71,75)]
    data = pd.read_fwf(filename, skiprows=start_row, nrows=nrows, header=None,
                       colspecs=COLSPECS, na_values=[-999.0, -99.9])
    
    # Assign multi-index and unstack DataFrame, such that each variable has a seperate column
    data = data.set_index([data.index//2, data.index%2]).unstack(level=1).swaplevel(i=0, j=1, axis='columns')
    
    # Sort columns to match original order and assign column names
    data = data.reindex(sorted(data.columns), axis='columns')
    BSRN_COLUMNS = ['day','minute',
                    'ghi','ghi_std','ghi_min','ghi_max',
                    'dni','dni_std','dni_min','dni_max','empty0','empty1','empty2','empty3','empty4',
                    'dhi','dhi_std','dhi_min','dhi_max',
                    'lwd','lwd_std','lwd_min','lwd_max',
                    'air_temperature','relative_humidity','pressure']
    data.columns = BSRN_COLUMNS
    
    # Change day and minute type to integer and drop empty columns
    data['day'] = data['day'].astype('Int64')
    data['minute'] = data['minute'].astype('Int64')
    data = data.drop(['empty0','empty1','empty2','empty3','empty4'], axis='columns')

    # Set datetime index and localize to UTC
    basename = os.path.basename(filename)
    data.index = pd.to_datetime(basename[3:7], format='%m%y') + pd.to_timedelta(data['day']-1, unit='d') + pd.to_timedelta(data['minute'], unit='min')
    
    
    try:
        data.index = data.index.tz_localize('UTC') # all BSRN timestamps are in UTC
    except TypeError:
        pass

    # Sort index and add missing timesteps
    ##data = data.sort_index().asfreq('1min') # can cause problems with duplicate time stamps
    meta = {}
    return data, meta