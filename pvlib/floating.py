"""
The ``floating`` module contains functions for calculating parameters related
to floating PV systems.
"""

import numpy as np
import pandas as pd


def stream_temperature_stefan(temp_air):
    r"""
    Estimate daily stream water temperature from daily ambient air temperature.

    Parameters
    ----------
    temp_air : numeric
        Daily average ambient dry bulb temperature. [°C]

    Returns
    -------
    water_temperature : numeric
        Daily average stream water temperature. [°C]

    Notes
    -----
    Daily average stream water temperature
    :math:`T_w` is calculated from daily ambient air temperature :math:`T_{air}` as
    provided in [1]_:

    .. math::
       :label: stream

        T_w = 5 + 0.75 \cdot T_{air}

    The predicted daily stream water temperatrues of this equation had a
    standard deviation of 2.7 °C compared to measurements. Small, shallow
    streams had smaller deviations than large, deep rivers.

    It should be noted that this equation is limited to streams, i.e., water
    bodies that are well mixed in vertical and transverse direction of a cross
    section. Also, it is only tested on ice-free streams. Consequently, when
    the mean ambient air temperature is lower than -6 °C, the surface stream
    water temperature is returned as NaN.

    Warning
    -------
    The expression has been developed for inland streams and is thus not
    suitable for estimating ocean temperature.

    References
    ----------
    .. [1] Stefan H. G., Preud'homme E. B. (1993). "Stream temperature
       estimation from air temperature." IN: Journal of the American Water
       Resources Association 29-1: 27-45.
       :doi:`10.1111/j.1752-1688.1993.tb01502.x`
    """

    temp_stream = 5 + 0.75 * temp_air

    temp_stream = np.where(temp_stream < 0, np.nan, temp_stream)

    if isinstance(temp_air, pd.Series):
        temp_stream = pd.Series(temp_stream, index=temp_air.index)

    return temp_stream
