# -*- coding: utf-8 -*-
"""
Soiling models
"""

import datetime
import pandas as pd


def soiling_kimber(rainfall_timeseries, threshold=6, soiling_rate=0.0015,
                   grace_period=14, max_soiling=0.3, manual_wash_dates=None,
                   initial_soiling=0):
    """
    Kimber soiling model [1]_ assumes soiling builds-up at aa daily rate unless
    the daily rainfall is greater than a threshold. The model also assumes that
    if daily rainfall has exceeded the threshold within a grace period, then
    the ground is too damp to cause soiling build-up. The model also assumes
    there is a maximum soiling build-up. Scheduled manual washes and rain
    events are assumed to rest soiling to zero.

    Parameters
    ----------
    rainfall_timeseries : pandas.Series
        a timeseries of rainfall in millimeters
    threshold : float, default 6
        the amount of rain in millimeters [mm] required to clean the panels
    soiling_rate: float, default 0.0015
        daily soiling rate, enter as fraction, not percent, default is 0.15%
    grace_period : int, default 14
        The number of days after a rainfall event when it's assumed the ground
        is damp, and so it's assumed there is no soiling. Change to smaller
        value for dry climate, default is 14-days
    max_soiling : float, default 0.3
        maximum soiling, soiling will build-up until this value, enter as
        fraction, not percent, default is 30%
    manual_wash_dates : sequence or None, default None
        A list or tuple of Python ``datetime.date`` when the panels were
        manually cleaned. Note there is no grace period after a manual
        cleaning, so soiling begins to build-up immediately after a manual
        cleaning
    initial_soiling : float, default 0
        the initial fraction of soiling on the panels at time zero in the input

    Returns
    -------
    soiling : timeseries
        soiling build-up fraction

    References
    ----------
    .. [1] "The Effect of Soiling on Large Grid-Connected Photovoltaic Systems
    in California and the Southwest Region of the United States," Addriane
    Kimber, et al., IEEE 4th World Conference on Photovoltaic Energy
    Conference, 2006, :doi:`10.1109/WCPEC.2006.279690`
    """
    # convert grace_period to timedelata
    grace_period = datetime.timedelta(days=grace_period)

    # manual wash dates
    if manual_wash_dates is None:
        manual_wash_dates = []

    # resample rainfall as days by summing intermediate times
    rainfall = rainfall_timeseries.resample("D").sum()

    # set indices to the end of the day
    rainfall.index = rainfall.index + datetime.timedelta(hours=23)

    # soiling
    soiling = pd.Series(float('NaN'), index=rainfall_timeseries.index)

    # set 1st timestep to initial soiling
    soiling.iloc[0] = initial_soiling

    # rainfall events that clean the panels
    rain_events = rainfall > threshold

    # loop over days
    for today in rainfall.index:

        # did rain exceed threshold?
        rain_exceed_thresh = rainfall[today] > threshold

        # if yes, then set soiling to zero
        if rain_exceed_thresh:
            soiling[today] = 0
            initial_soiling = 0
            continue

        # start day of grace period
        start_day = today - grace_period

        # rainfall event during grace period?
        rain_in_grace_period = any(rain_events[start_day:today])

        # if rain exceeded threshold during grace period,
        # assume ground is still damp, so no or v. low soiling
        if rain_in_grace_period:
            soiling[today] = 0
            initial_soiling = 0
            continue

        # is this a manual wash date?
        if today.date() in manual_wash_dates:
            soiling[today] = 0
            initial_soiling = 0
            continue

        # so, it didn't rain enough to clean, it hasn't rained enough recently,
        # and we didn't manually clean panels, so soil them by adding daily
        # soiling rate to soiling from previous day
        total_soil = initial_soiling + soiling_rate

        # check if soiling has reached the maximum
        soiling[today] = (
            max_soiling if (total_soil >= max_soiling) else total_soil)

        initial_soiling = soiling[today]  # reset initial soiling

    return soiling.interpolate()
