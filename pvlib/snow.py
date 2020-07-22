"""
The ``snow`` module contains functions that model module snow cover and the
associated effects on PV module output
"""

import numpy as np
import pandas as pd
from pvlib.tools import sind


def _time_delta_in_hours(times):
    delta = times.to_series().diff()
    return delta.dt.total_seconds().div(3600)


def fully_covered_nrel(snowfall, threshold_snowfall=1.):
    '''
    Calculates the timesteps when the row's slant height is fully covered
    by snow.

    Parameters
    ----------
    snowfall : Series
        Accumulated snowfall in each time period [cm]

    threshold_snowfall : float, default 1.0
        Hourly snowfall above which snow coverage is set to the row's slant
        height. [cm/hr]

    Returns
    ----------
    boolean: Series
        True where the snowfall exceeds the defined threshold to fully cover
        the panel.

    Notes
    -----
    Implements the model described in [1]_ with minor improvements in [2]_.

    References
    ----------
    .. [1] Marion, B.; Schaefer, R.; Caine, H.; Sanchez, G. (2013).
       "Measured and modeled photovoltaic system energy losses from snow for
       Colorado and Wisconsin locations." Solar Energy 97; pp.112-121.
    .. [2] Ryberg, D; Freeman, J. "Integration, Validation, and Application
       of a PV Snow Coverage Model in SAM" (2017) NREL Technical Report
       NREL/TP-6A20-68705
    '''
    timestep = _time_delta_in_hours(snowfall.index)
    hourly_snow_rate = snowfall / timestep
    # if we can infer a time frequency, use first snowfall value
    # otherwise the first snowfall value is ignored
    freq = pd.infer_freq(snowfall.index)
    if freq is not None:
        timedelta = pd.tseries.frequencies.to_offset(freq) / pd.Timedelta('1h')
        hourly_snow_rate.iloc[0] = snowfall[0] / timedelta
    else:  # can't infer frequency from index
        hourly_snow_rate[0] = 0  # replaces NaN
    return hourly_snow_rate > threshold_snowfall


def coverage_nrel(snowfall, poa_irradiance, temp_air, surface_tilt,
                  initial_coverage=0, threshold_snowfall=1.,
                  can_slide_coefficient=-80., slide_amount_coefficient=0.197):
    '''
    Calculates the fraction of the slant height of a row of modules covered by
    snow at every time step.

    Implements the model described in [1]_ with minor improvements in [2]_,
    with the change that the output is in fraction of the row's slant height
    rather than in tenths of the row slant height. As described in [1]_, model
    validation focused on fixed tilt systems.

    Parameters
    ----------
    snowfall : Series
        Accumulated snowfall within each time period. [cm]
    poa_irradiance : Series
        Total in-plane irradiance [W/m^2]
    temp_air : Series
        Ambient air temperature [C]
    surface_tilt : numeric
        Tilt of module's from horizontal, e.g. surface facing up = 0,
        surface facing horizon = 90. [degrees]
    initial_coverage : float, default 0
        Fraction of row's slant height that is covered with snow at the
        beginning of the simulation. [unitless]
    threshold_snowfall : float, default 1.0
        Hourly snowfall above which snow coverage is set to the row's slant
        height. [cm/hr]
    can_slide_coefficient : float, default -80.
        Coefficient to determine if snow can slide given irradiance and air
        temperature. [W/(m^2 C)]
    slide_amount_coefficient : float, default 0.197
        Coefficient to determine fraction of snow that slides off in one hour.
        [unitless]

    Returns
    -------
    snow_coverage : Series
        The fraction of the slant height of a row of modules that is covered
        by snow at each time step.

    Notes
    -----
    In [1]_, `can_slide_coefficient` is termed `m`, and the value of
    `slide_amount_coefficient` is given in tenths of a module's slant height.

    References
    ----------
    .. [1] Marion, B.; Schaefer, R.; Caine, H.; Sanchez, G. (2013).
       "Measured and modeled photovoltaic system energy losses from snow for
       Colorado and Wisconsin locations." Solar Energy 97; pp.112-121.
    .. [2] Ryberg, D; Freeman, J. (2017). "Integration, Validation, and
       Application of a PV Snow Coverage Model in SAM" NREL Technical Report
       NREL/TP-6A20-68705
    '''

    # find times with new snowfall
    new_snowfall = fully_covered_nrel(snowfall, threshold_snowfall)

    # set up output Series
    snow_coverage = pd.Series(np.nan, index=poa_irradiance.index)

    # determine amount that snow can slide in each timestep
    can_slide = temp_air > poa_irradiance / can_slide_coefficient
    slide_amt = slide_amount_coefficient * sind(surface_tilt) * \
        _time_delta_in_hours(poa_irradiance.index)
    slide_amt[~can_slide] = 0.
    # don't slide during snow events
    slide_amt[new_snowfall] = 0.
    # don't slide in the interval preceding the snowfall data
    slide_amt.iloc[0] = 0

    # build time series of cumulative slide amounts
    sliding_period_ID = new_snowfall.cumsum()
    cumulative_sliding = slide_amt.groupby(sliding_period_ID).cumsum()

    # set up time series of snow coverage without any sliding applied
    snow_coverage[new_snowfall] = 1.0
    if np.isnan(snow_coverage.iloc[0]):
        snow_coverage.iloc[0] = initial_coverage
    snow_coverage.ffill(inplace=True)
    snow_coverage -= cumulative_sliding

    # clean up periods where row is completely uncovered
    return snow_coverage.clip(lower=0)


def dc_loss_nrel(snow_coverage, num_strings):
    '''
    Calculates the fraction of DC capacity lost due to snow coverage.

    DC capacity loss assumes that if a string is partially covered by snow,
    the string's capacity is lost; see [1]_, Eq. 11.8.

    Module orientation is accounted for by specifying the number of cell
    strings in parallel along the slant height.
    For example, a typical 60-cell module has 3 parallel strings, each
    comprising 20 cells in series, with the cells arranged in 6 columns of 10
    cells each. For a row consisting of single modules, if the module is
    mounted in portrait orientation, i.e., the row slant height is along a
    column of 10 cells, there is 1 string in parallel along the row slant
    height, so `num_strings=1`. In contrast, if the module is mounted in
    landscape orientation with the row slant height comprising 6 cells, there
    are 3 parallel strings along the row slant height, so `num_strings=3`.

    Parameters
    ----------
    snow_coverage : numeric
        The fraction of row slant height covered by snow at each time step.

    num_strings: int
        The number of parallel-connected strings along a row slant height.

    Returns
    -------
    loss : numeric
        fraction of DC capacity loss due to snow coverage at each time step.

    References
    ----------
    .. [1] Gilman, P. et al., (2018). "SAM Photovoltaic Model Technical
       Reference Update", NREL Technical Report NREL/TP-6A20-67399.
       Available at https://www.nrel.gov/docs/fy18osti/67399.pdf
    '''
    return np.ceil(snow_coverage * num_strings) / num_strings
