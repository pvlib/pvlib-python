"""Functions to read NREL PVDAQ data
"""

from time import time
from io import StringIO
import sys
from datetime import timedelta

import requests
import numpy as np
import pandas as pd


def get_pvdaq_data(sysid=2, api_key='DEMO_KEY', year=2011, delim=',',
                   standardize=True):
    """This fuction queries one or more years of raw PV system data from NREL's
     PVDAQ data service: https://maps.nrel.gov/pvdaq/

     This function uses the annual raw data file API, which is the most
     efficient way of accessing multi-year, sub-hourly time series data.

    Parameters
    ----------
    sysid: int
        The system ID corresponding to the site that data should be
        queried from
    api_key: string
        Your API key (https://developer.nrel.gov/docs/api-key/)
    year: int of list of ints
        Either the year to request or the list of years to request. Multiple
        years will be concatenated into a single data frame
    delim: string
        The deliminator used in the CSV file being requested

    Returns
    -------
    label: pandas data frame
        A data frame containing the tabular time series data from the PVDAQ
        service over the years requested

    """
    # Force year to be a list of integers
    ti = time()
    try:
        year = int(year)
    except TypeError:
        year = [int(yr) for yr in year]
    else:
        year = [year]
    # Each year must queries separately, so iterate over the years and
    # generate a list of dataframes.
    df_list = []
    it = 0
    for yr in year:
        progress(it, len(year), 'querying year {}'.format(year[it]))
        req_params = {
            'api_key': api_key,
            'system_id': sysid,
            'year': yr
        }
        base_url = 'https://developer.nrel.gov/api/pvdaq/v3/data_file?'
        param_list = [str(item[0]) + '=' + str(item[1])
                      for item in req_params.items()]
        req_url = base_url + '&'.join(param_list)
        response = requests.get(req_url)
        if int(response.status_code) != 200:
            print('\n error: ', response.status_code)
            return
        df = pd.read_csv(StringIO(response.text), delimiter=delim)
        df_list.append(df)
        it += 1
    tf = time()
    msg = 'queries complete in {:.1f} seconds       '.format(tf - ti)
    progress(it, len(year), msg)
    print('\n')
    # concatenate the list of yearly data frames
    df = pd.concat(df_list, axis=0, sort=True)
    if standardize:
        df = standardize_time_axis(df, datetimekey='Date-Time')
    return df


def standardize_time_axis(df, datetimekey='Date-Time'):
    '''
    This function takes in a pandas data frame containing tabular time series
    data, likely generated with a call to pandas.read_csv(). It is assumed that
    each row of the data frame corresponds to a unique date-time, though not
    necessarily on standard intervals. This function will attempt to convert a
    user-specified column containing time stamps to python datetime objects,
    assign this column to the index of the data frame, and then standardize the
    index over time. By standardize, we mean reconstruct the index to be at
    regular intervals, starting at midnight of the first day of the data set.
    This solves a couple common data errors when working with raw data.
        (1) Missing data points from skipped scans in the data acquisition
            system.
        (2) Time stamps that are at irregular exact times, including fractional
            seconds.

    :param df: A pandas data frame containing the tabular time series data
    :param datetimekey: An optional key corresponding to the name of the column
        that contains the time stamps
    :return: A new data frame with a standardized time axis
    '''
    # convert index to timeseries
    try:
        df[datetimekey] = pd.to_datetime(df[datetimekey])
        df.set_index('Date-Time', inplace=True)
    except KeyError:
        time_cols = [col for col in df.columns
                     if np.logical_or('Time' in col, 'time' in col)]
        key = time_cols[0]
        df[datetimekey] = pd.to_datetime(df[key])
        df.set_index(datetimekey, inplace=True)
    # standardize the timeseries axis to a regular frequency over
    # a full set of days
    diff = (df.index[1:] - df.index[:-1]).seconds
    freq = int(np.median(diff))  # the number of secs between each measurement
    start = df.index[0]
    end = df.index[-1]
    time_index = pd.date_range(
        start=start.date(),
        end=end.date() + timedelta(days=1),
        freq='{}s'.format(freq)
    )[:-1]
    df = df.reindex(index=time_index, method='nearest')
    return df.fillna(value=0)


def progress(count, total, status=''):
    """
    Python command line progress bar in less than 10 lines of code. · GitHub
    https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    :param count: the current count, int
    :param total: to total count, int
    :param status: a message to display
    :return:
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


if __name__ == "__main__":
    df = get_pvdaq_data()
    print(df.head())
