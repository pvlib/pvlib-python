"""
The ``floating`` module contains functions for calculating parameters related
to floating PV systems.
"""

import numpy as np
import pandas as pd


def daily_stream_temperature_stefan(temp_air):
    r"""
    Estimation of daily stream water temperature based on ambient temperature.

    Parameters
    ----------
    temp_air : numeric
        Ambient dry bulb temperature. [degrees C]

    Returns
    -------
    daily_stream_temperature : numeric
        Daily average stream water temperature. [degrees C]

    Notes
    -----
    The equation for calculating the daily average stream water temperature
    :math:`T_w` using the ambient air temperature :math:`T_{air}` is provided
    in [1]_ and given by:

    .. math::
       :label: stream

        T_w = 5 + 0.75 \cdot T_{air}

    The predicted daily stream water temperatrues of this equation had a
    standard deviation of 2.7 $^o$C compared to measurements. Small, shallow
    streams had smaller deviations than large, deep rivers.

    It should be noted that this equation is limited to streams, i.e., water
    bodies that are well mixed in vertical and transverse direction of a cross
    section. Also, it is only tested on ice-free streams. Consequently, when
    the mean ambient air temperature is lower than -6 $^o$C, the surface stream
    water temperature is assumed to be zero.

    References
    ----------
    .. [1] Stefan H. G., Preud'homme E. B. (1993). "Stream temperature
       estimation from air temperature." IN: Journal of the American Water
       Resources Association 29-1: 27-45.
       :doi:`10.1111/j.1752-1688.1993.tb01502.x`
    """

    temp_stream = 5 + 0.75 * temp_air

    temp_stream = np.where(temp_stream < 0, 0, temp_stream)

    if isinstance(temp_air, pd.Series):
        temp_stream = pd.Series(temp_stream, index=temp_air.index)

    return temp_stream
