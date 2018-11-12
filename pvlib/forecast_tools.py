# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:03:04 2018

@author: cwhanse
"""

import pandas as pd
import statsmodels.api as sm
from datetime import timedelta
import numpy as np


# forecast functions for persistence method
def _get_data_for_persistence(start, history, dataWindowLength):

    """
    Returns data from history for persistence forecast

    Parameters
    -----------

    pvobj : an instance of type PVobj

    start : datetime
        the time for the first forecast value

    history : pandas DataFrame with key 'ac_power'
        historical values of AC power from which the forecast is made.

    dataWindowLenth : timedelta
        time interval in history to be considered

    Returns
    ----------
    result :
        data from history in period of duration dataWindowLength, prior to the
        minimum of either the forecast start or the last time in history

    """

    # find last time to include. Earliest of either forecast start or the last
    # time in history
    end_data_time = min([history.index[-1], start])
    # select data within dataWindowLength
    result = history.loc[(history.index>=end_data_time - dataWindowLength) &
                         (history.index<=end_data_time)]

    return result


def _calc_ratio(X, Y):

    # Compute the ratio X / Y
    # inputs X and Y can be pandas.Series, or np.ndarray
    # returns the same type as the inputs
    np.seterr(divide='ignore')
    ratio = np.true_divide(X, Y)
    np.seterr(divide='raise')
    return ratio


def forecast_persistence(start, end, deltat, history, dataWindowLength,
                         reference=None, method='mean'):

    """
    Generate forecast from start to end at time resolution deltat
    using the persistence method applied to the data in history.

    Parameters
    -----------

    start : datetime
        the time for the first forecast value

    end : datetime
        the last time to be forecast

    deltat : timedelta
        the time interval for the forecast

    history : pandas Series
        historical values from which the forecast is made.

    dataWindowLenth : timedelta
        time interval in history to be considered

    reference : pandas Series, default None
        if forecasting an index, e.g., clearness index, reference is the
        denominator of the index. Must extend from prior to 
        (start - dataWindowLength) to end.

    method : str, default 'mean'
        method applied to history or data to calculate forecast values

    Returns
    --------
    forecast : pandas Series
        forecast from start to end at interval deltat
    """
    # time index for forecast
    dr = pd.DatetimeIndex(start=start, end=end, freq=deltat)

    forecast = pd.Series(data=0.0, index=dr)

    # get data for forecast
    fitdata = _get_data_for_persistence(start, history, dataWindowLength)

    if reference: # forecast index history / reference
        # get reference data for index calculation
        refdata = _get_data_for_persistence(fitdata.index[0], reference,
            fitdata.index[-1] - fitdata.index[0])
        if len(refdata)==0:
            raise ValueError('No data for persistence forecast in history from'
                             ' {} to {}'.format(start - dataWindowLength,
                               end - dataWindowLength))
        else:
            # interpolate reference to time points in history
            denom = np.interp(refdata.index.asi8/1e9, fitdata.asi8/1e9,
                              fitdata.values)
            index = _calc_ratio(fitdata, denom)
            forecast_data = _get_data_for_persistence(start, reference,
                                                      end - start)
            # line up reference data to desired forecast times
            forecast_data = np.interp(dr.asi8/1e9,
                                      forecast_data.index.asi8/1e9,
                                      forecast_data.values)
    else: # simple persistence forecast
        index = 1.0

    if method=='mean':
        index = index.mean()
    else:
        raise ValueError('{} is not a recognized value for method'
                         .format(method))
    # forecast
    forecast = pd.Series(data=forecast_data * index, index=dr)

    return forecast


# Forecast functions for ARMA method
def get_num_intervals(end, start, deltat):
    """
    Calculate number of intervals of length deltat from end working back to
    start, to include start.

    Parameters
    ------------
    end : datetime

    start : datetime

    deltat : timedelta

    """
    return int((end - start).total_seconds() / deltat.seconds)


def _get_data_for_ARMA_forecast(start, end, deltat, history,
                                dataWindowLength):

    """
    Select data from history for fitting the forecast model.

    Parameters
    -----------

    start : datetime
        the first time in the forecast

    end : datetime
        the last time in be forecast

    deltat : timedelta
        the time interval for the forecast

    history : pandas Series
        historical values of the quantity to be forecast

    Returns
    ----------
    fitdata : pandas Series
        data from history aligned to be in phase with requested forecast

    f_intervals : integer
        number of time steps in the forecast
    """

    # find number of deltat intervals between start and last time in history
    # assumes that start > max(history.index)
    if start < history.index[-1]:
        # truncate history
        history = history.loc[history.index < start]

    K = get_num_intervals(start, history.index[-1], deltat)

    # find number of deltat intervals covering dataWindowLength in history
    N = get_num_intervals(start - K*deltat,
                          start - K*deltat - dataWindowLength,
                          deltat)

    # start time of resampled history in phase with forecast start
    resample_start = start - (K + N - 1) * deltat
    resample_end = start - K * deltat

    # calculate base time for resample, minutes out of phase with midnight
    midnight = resample_start.replace(hour=0, minute=0, second=0)
    first_after_midnight = resample_start - \
                    int((resample_start - midnight) / deltat) * deltat
    base = (first_after_midnight - midnight).total_seconds() / 60

    # resample history
    idata = history.resample(rule=pd.to_timedelta(deltat),
                             closed='left',
                             label='left',
                             base=base).mean()

    # extract window from resampled history
    return idata.loc[(idata.index>=resample_start) &
                        (idata.index<=resample_end)].copy()


def forecast_ARMA(start, end, history, deltat,
                  dataWindowLength=timedelta(hours=1),
                  order=None,
                  start_params=None):

    """
    Generate forecast from start to end with steps deltat
    using an ARMA model fit to data in history.

    Parameters
    -----------

    start : datetime
        the first time to be forecast

    end : datetime
        the last time to be forecast

    deltat : timedelta
        the time step for the forecast

    history : pandas Series
        historical values from which the forecast is made.

    dataWindowLenth : timedelta, default one hour
        time interval for data in history to be considered for model fitting

    order : tuple of three integers
        autoregressive, difference, and moving average orders for an ARMA
        forecast model

    start_params : list of float
        initial guess of model parameters, one value for each autoregressive
        and moving average coefficient, followed by the value for the variance
        term


    """

    # TODO: input validation

    # create datetime index for forecast
    fdr = pd.DatetimeIndex(start=start, end=end, freq=pd.to_timedelta(deltat))

    # get time-averaged data from history over specified data window and that
    # is in phase with forecast times
    fitdata = _get_data_for_ARMA_forecast(start, end,
                                          deltat, history,
                                          dataWindowLength)

    # TODO: model identification logic

    # fit model of order (p, d, q)
    # defaults to first order differencing to help with non-stationarity
    if not order:
        if deltat.total_seconds()>=15*60:
            # use MA process for time intervals 15 min or longer
            p = 0
            d = 1
            q = 1
            start_params = [1, 0]
        else:
            # use AR process for time intervals < 15 min
            p = 1
            d = 1
            q = 0
            start_params = [0, 1]
        order = (p, d, q)

    model = sm.tsa.statespace.SARIMAX(fitdata,
                                      trend='n',
                                      order=order)

    # if not provided, generate guess of model parameters, helps overcome 
    # non-stationarity
    if not start_params:
        # generate a list with one entry '0' for each AR or MA parameter 
        # plus an entry '1' for the variance parameter
        start_params = []
        for i in range(0, order[0]+order[2]):
            start_params.append(0)
        start_params.append(1)

    # generate the ARMA model object
    results = model.fit(start_params=start_params)

    # total intervals to forecast from end of data to end of forecast
    idr = pd.DatetimeIndex(start=fitdata.index[-1],
                           end=end,
                           freq=pd.to_timedelta(deltat))

    f_intervals = len(idr) - 1 # first time in idr is last data point

    # forecast
    f = results.forecast(f_intervals)

    # return the requested forecast times
    f = f[fdr]
    #TODO: limit forecast by sunrise, sunset and clear-sky power
    f[f < 0] = 0.0

    return f[fdr]

