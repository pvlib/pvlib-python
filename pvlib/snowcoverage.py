"""
The ``snow`` module contains functions that model the effect of snow on
solar modules.
"""

import numpy as np
import pandas as pd
from pvlib.tools import sind


def _time_delta_in_hours(times):
    delta = times.to_series().diff()
    return delta.dt.seconds.div(3600)


def fully_covered(snowfall, threshold=1.):
    '''
    Calculates the timesteps when the row's slant height is fully covered
    by snow.

    Parameters
    ----------
    snowfall : Series
        Accumulated snowfall in each time period [cm]

    threshold : float, default 1.0
        Minimum hourly snowfall to cover a row's slant height. [cm/hr]

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
       “Measured and modeled photovoltaic system energy losses from snow for
       Colorado and Wisconsin locations.” Solar Energy 97; pp.112-121.
    .. [2] Ryberg, D; Freeman, J. "Integration, Validation, and Application
       of a PV Snow Coverage Model in SAM" (2017) NREL Technical Report
    '''
    delta = snowfall.index.to_series().diff()  # [0] will be NaT
    timestep = delta.dt.seconds.div(3600)  # convert to hours
    time_adjusted = snowfall / timestep
    time_adjusted.iloc[0] = 0  # replace NaN from NaT / timestep
    return time_adjusted >= threshold


def snow_coverage_nrel(snowfall, poa_irradiance, temperature, surface_tilt,
                       threshold_snowfall=1., m=-80,
                       sliding_coefficient=0.197):
    '''
    Calculates the fraction of the slant height of a row of modules covered by
    snow at every time step.

    Parameters
    ----------
    snowfall : Series
        Accumulated snowfall at the end of each time period. [cm]
    poa_irradiance : Series
        Total in-plane irradiance [W/m^2]
    temperature : Series
        Ambient air temperature at the surface [C]
    surface_tilt : numeric
        Tilt of module's from horizontal, e.g. surface facing up = 0,
        surface facing horizon = 90. Must be between 0 and 180. [degrees]
    threshold_snowfall : float, default 1.0
        Minimum hourly snowfall to cover a row's slant height. [cm/hr]
    m : numeric
        A coefficient used in the model described in [1]_. [W/(m^2 C)]
    sliding coefficient : numeric
        Empirically determined coefficient used in [1]_ to determine how much
        snow slides off in each time period. [unitless]

    Returns
    -------
    snow_coverage : numeric
        The fraction of a the slant height of a row of modules that is covered
        by snow at each time step.

    Notes
    -----
    Initial snow coverage is assumed to be zero. Implements the model described
    in [1]_ with minor improvements in [2]_. Validated for fixed tilt systems.

    References
    ----------
    .. [1] Marion, B.; Schaefer, R.; Caine, H.; Sanchez, G. (2013).
       “Measured and modeled photovoltaic system energy losses from snow for
       Colorado and Wisconsin locations.” Solar Energy 97; pp.112-121.
    .. [2] Ryberg, D; Freeman, J. (2017). "Integration, Validation, and
       Application of a PV Snow Coverage Model in SAM" NREL Technical Report
       NREL/TP-6A20-68705 
    '''

    # set up output Series
    snow_coverage = pd.Series(index=poa_irradiance.index, data=np.nan)
    snow_events = snowfall[fully_covered(snowfall, threshold_snowfall)]

    can_slide = temperature > poa_irradiance / m
    slide_amt = sliding_coefficient * sind(surface_tilt) * \
        _time_delta_in_hours(poa_irradiance.index)

    uncovered = pd.Series(0.0, index=poa_irradiance.index)
    uncovered[can_slide] = slide_amt[can_slide]

    windows = [(ev, ne) for (ev, ne) in
        zip(snow_events.index[:-1], snow_events.index[1:])]
    # add last time window
    windows.append((snow_events.index[-1], snowfall.index[-1]))

    for (ev, ne) in windows:
        filt = (snow_coverage.index > ev) & (snow_coverage.index <= ne)
        snow_coverage[ev] = 1.0
        snow_coverage[filt] = 1.0 - uncovered[filt].cumsum()

    # clean up periods where row is completely uncovered
    snow_coverage[snow_coverage < 0] = 0
    snow_coverage = snow_coverage.fillna(value=0.)
    return snow_coverage


def snow_loss_factor(snow_coverage, num_strings):
    '''
    Calculates the DC loss due to snow coverage. Assumes that if a string is
    partially covered by snow, it produces 0W.

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
    '''
    return np.ceil(snow_coverage * num_strings) / num_strings
