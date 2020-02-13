# -*- coding: utf-8 -*-
"""
This module contains functions for losses of various types: soiling, mismatch,
snow cover, etc.
"""

import datetime
import numpy as np
import pandas as pd
from pvlib.tools import cosd


def soiling_hsu(rainfall, cleaning_threshold, tilt, pm2_5, pm10,
                depo_veloc={'2_5': 0.004, '10': 0.0009},
                rain_accum_period=pd.Timedelta('1h')):
    """
    Calculates soiling ratio given particulate and rain data using the model
    from Humboldt State University [1]_.

    Parameters
    ----------

    rainfall : Series
        Rain accumulated in each time period. [mm]

    cleaning_threshold : float
        Amount of rain in an accumulation period needed to clean the PV
        modules. [mm]

    tilt : float
        Tilt of the PV panels from horizontal. [degree]

    pm2_5 : numeric
        Concentration of airborne particulate matter (PM) with
        aerodynamic diameter less than 2.5 microns. [g/m^3]

    pm10 : numeric
        Concentration of airborne particulate matter (PM) with
        aerodynamicdiameter less than 10 microns. [g/m^3]

    depo_veloc : dict, default {'2_5': 0.4, '10': 0.09}
        Deposition or settling velocity of particulates. [m/s]

    rain_accum_period : Timedelta, default 1 hour
        Period for accumulating rainfall to check against `cleaning_threshold`
        It is recommended that `rain_accum_period` be between 1 hour and
        24 hours.

    Returns
    -------
    soiling_ratio : Series
        Values between 0 and 1. Equal to 1 - transmission loss.

    References
    -----------
    .. [1] M. Coello and L. Boyle, "Simple Model For Predicting Time Series
       Soiling of Photovoltaic Panels," in IEEE Journal of Photovoltaics.
       doi: 10.1109/JPHOTOV.2019.2919628
    .. [2] Atmospheric Chemistry and Physics: From Air Pollution to Climate
       Change. J. Seinfeld and S. Pandis. Wiley and Sons 2001.

    """
    try:
        from scipy.special import erf
    except ImportError:
        raise ImportError("The soiling_hsu function requires scipy.")

    # accumulate rainfall into periods for comparison with threshold
    accum_rain = rainfall.rolling(rain_accum_period, closed='right').sum()
    # cleaning is True for intervals with rainfall greater than threshold
    cleaning_times = accum_rain.index[accum_rain >= cleaning_threshold]

    horiz_mass_rate = pm2_5 * depo_veloc['2_5']\
        + np.maximum(pm10 - pm2_5, 0.) * depo_veloc['10']
    tilted_mass_rate = horiz_mass_rate * cosd(tilt)  # assuming no rain

    # tms -> tilt_mass_rate
    tms_cumsum = np.cumsum(tilted_mass_rate * np.ones(rainfall.shape))

    mass_no_cleaning = pd.Series(index=rainfall.index, data=tms_cumsum)
    mass_removed = pd.Series(index=rainfall.index)
    mass_removed[0] = 0.
    mass_removed[cleaning_times] = mass_no_cleaning[cleaning_times]
    accum_mass = mass_no_cleaning - mass_removed.ffill()

    soiling_ratio = 1 - 0.3437 * erf(0.17 * accum_mass**0.8473)

    return soiling_ratio


def soiling_kimber(rainfall, cleaning_threshold=6, soiling_rate=0.0015,
                   grace_period=14, max_soiling=0.3, manual_wash_dates=None,
                   initial_soiling=0):
    """
    Kimber soiling model [1]_ assumes soiling builds up at a daily rate unless
    the daily rainfall is greater than a threshold. The model also assumes that
    if daily rainfall has exceeded the threshold within a grace period, then
    the ground is too damp to cause soiling build-up. The model also assumes
    there is a maximum soiling build-up. Scheduled manual washes and rain
    events are assumed to reset soiling to zero.

    Parameters
    ----------
    rainfall: pandas.Series
        Accumulated rainfall at the end of each time period. [mm]
    cleaning_threshold: float, default 6
        Amount of daily rainfall required to clean the panels. [mm]
    soiling_loss_rate: float, default 0.0015
        Fraction of energy lost due to one day of soiling. [unitless]
    grace_period : int, default 14
        Number of days after a rainfall event when it's assumed the ground is
        damp, and so it's assumed there is no soiling. [days]
    max_soiling : float, default 0.3
        Maximum fraction of energy lost due to soiling. Soiling will build up
        until this value. [unitless]
    manual_wash_dates : sequence or None, default None
        List or tuple of dates as Python ``datetime.date`` when the panels were
        washed manually. Note there is no grace period after a manual wash, so
        soiling begins to build up immediately.
    initial_soiling : float, default 0
        Initial fraction of energy lost due to soiling at time zero in the
        `rainfall` series input. [unitless]

    Returns
    -------
    pandas.Series
        fraction of energy lost due to soiling, has same intervals as input

    Notes
    -----
    The soiling loss rate depends on both the geographical region and the
    soiling environment type. Rates measured by Kimber [1]_ are summarized in
    the following table:

    ===================  =======  =========  ======================
    Region/Environment   Rural    Suburban   Urban/Highway/Airport
    ===================  =======  =========  ======================
    Central Valley       0.0011   0.0019     0.0020
    Northern CA          0.0011   0.0010     0.0016
    Southern CA          0        0.0016     0.0019
    Desert               0.0030   0.0030     0.0030
    ===================  =======  =========  ======================

    Rainfall thresholds and grace periods may also vary by region. Please
    consult [1]_ more information.

    References
    ----------
    .. [1] "The Effect of Soiling on Large Grid-Connected Photovoltaic Systems
       in California and the Southwest Region of the United States," Adrianne
       Kimber, et al., IEEE 4th World Conference on Photovoltaic Energy
       Conference, 2006, :doi:`10.1109/WCPEC.2006.279690`
    """
    # convert grace_period to timedelata
    grace_period = datetime.timedelta(days=grace_period)

    # manual wash dates
    if manual_wash_dates is None:
        manual_wash_dates = []

    # resample rainfall as days by summing intermediate times
    daily_rainfall = rainfall.resample("D").sum()

    # set indices to the end of the day
    daily_rainfall.index = daily_rainfall.index + datetime.timedelta(hours=23)

    # soiling
    soiling = pd.Series(float('NaN'), index=rainfall.index)

    # set 1st timestep to initial soiling
    soiling.iloc[0] = initial_soiling

    # rainfall events that clean the panels
    rain_events = daily_rainfall > cleaning_threshold

    # loop over days
    for today in daily_rainfall.index:

        # if rain exceed threshold today, set soiling to zero
        if rain_events[today]:
            soiling[today] = initial_soiling = 0
            continue

        # start day of grace period
        start_day = today - grace_period

        # rainfall event during grace period?
        rain_in_grace_period = any(rain_events[start_day:today])

        # if rain exceeded threshold during grace period,
        # assume ground is still damp, so no or very low soiling
        if rain_in_grace_period:
            soiling[today] = initial_soiling = 0
            continue

        # is this a manual wash date?
        if today.date() in manual_wash_dates:
            soiling[today] = initial_soiling = 0
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
