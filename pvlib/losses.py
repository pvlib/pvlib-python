# -*- coding: utf-8 -*-
"""
This module contains functions for losses of various types: soiling, mismatch,
snow cover, etc.
"""

import numpy as np
import pandas as pd
from pvlib.tools import cosd


def erf(x):
    """
    Calculates the ERF function

    Parameters
    ----------
    x : numeric
        Input value/array

    Returns
    -------
    erf : numeric
        The values of the error function at the given points x.

    """

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)

    return sign*y


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
