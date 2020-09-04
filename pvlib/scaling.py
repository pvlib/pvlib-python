"""
The ``scaling`` module contains functions for manipulating irradiance
or other variables to account for temporal or spatial characteristics.
"""

import numpy as np
import pandas as pd

import scipy.optimize
from scipy.spatial.distance import pdist


def wvm(clearsky_index, positions, cloud_speed, dt=None):
    """
    Compute spatial aggregation time series smoothing on clear sky index based
    on the Wavelet Variability model of Lave et al [1-2]. Implementation is
    basically a port of the Matlab version of the code [3].

    Parameters
    ----------
    clearsky_index : numeric or pandas.Series
        Clear Sky Index time series that will be smoothed.

    positions : numeric
        Array of coordinate distances as (x,y) pairs representing the
        easting, northing of the site positions in meters [m]. Distributed
        plants could be simulated by gridded points throughout the plant
        footprint.

    cloud_speed : numeric
        Speed of cloud movement in meters per second [m/s].

    dt : float, default None
        The time series time delta. By default, is inferred from the
        clearsky_index. Must be specified for a time series that doesn't
        include an index. Units of seconds [s].

    Returns
    -------
    smoothed : numeric or pandas.Series
        The Clear Sky Index time series smoothed for the described plant.

    wavelet: numeric
        The individual wavelets for the time series before smoothing.

    tmscales: numeric
        The timescales associated with the wavelets in seconds [s].

    References
    ----------
    [1] M. Lave, J. Kleissl and J.S. Stein. A Wavelet-Based Variability
    Model (WVM) for Solar PV Power Plants. IEEE Transactions on Sustainable
    Energy, vol. 4, no. 2, pp. 501-509, 2013.

    [2] M. Lave and J. Kleissl. Cloud speed impact on solar variability
    scaling - Application to the wavelet variability model. Solar Energy,
    vol. 91, pp. 11-21, 2013.

    [3] Wavelet Variability Model - Matlab Code:
    https://pvpmc.sandia.gov/applications/wavelet-variability-model/
    """

    # Added by Joe Ranalli (@jranalli), Penn State Hazleton, 2019

    pos = np.array(positions)
    dist = pdist(pos, 'euclidean')
    wavelet, tmscales = _compute_wavelet(clearsky_index, dt)

    # Find effective length of position vector, 'dist' is full pairwise
    n_pairs = len(dist)

    def fn(x):
        return np.abs((x ** 2 - x) / 2 - n_pairs)
    n_dist = np.round(scipy.optimize.fmin(fn, np.sqrt(n_pairs), disp=False))

    # Compute VR
    A = cloud_speed / 2  # Resultant fit for A from [2]
    vr = np.zeros(tmscales.shape)
    for i, tmscale in enumerate(tmscales):
        rho = np.exp(-1 / A * dist / tmscale)  # Eq 5 from [1]

        # 2*rho is because rho_ij = rho_ji. +n_dist accounts for sum(rho_ii=1)
        denominator = 2 * np.sum(rho) + n_dist
        vr[i] = n_dist ** 2 / denominator  # Eq 6 of [1]

    # Scale each wavelet by VR (Eq 7 in [1])
    wavelet_smooth = np.zeros_like(wavelet)
    for i in np.arange(len(tmscales)):
        if i < len(tmscales) - 1:  # Treat the lowest freq differently
            wavelet_smooth[i, :] = wavelet[i, :] / np.sqrt(vr[i])
        else:
            wavelet_smooth[i, :] = wavelet[i, :]

    outsignal = np.sum(wavelet_smooth, 0)

    try:  # See if there's an index already, if so, return as a pandas Series
        smoothed = pd.Series(outsignal, index=clearsky_index.index)
    except AttributeError:
        smoothed = outsignal  # just output the numpy signal

    return smoothed, wavelet, tmscales


def latlon_to_xy(coordinates):
    """
    Convert latitude and longitude in degrees to a coordinate system measured
    in meters from zero deg latitude, zero deg longitude.

    This is a convenience method to support inputs to wvm. Note that the
    methodology used is only suitable for short distances. For conversions of
    longer distances, users should consider use of Universal Transverse
    Mercator (UTM) or other suitable cartographic projection. Consider
    packages built for cartographic projection such as pyproj (e.g.
    pyproj.transform()) [2].

    Parameters
    ----------

    coordinates : numeric
        Array or list of (latitude, longitude) coordinate pairs. Use decimal
        degrees notation.

    Returns
    -------
    xypos : numeric
        Array of coordinate distances as (x,y) pairs representing the
        easting, northing of the position in meters [m].

    References
    ----------
    [1] H. Moritz. Geodetic Reference System 1980, Journal of Geodesy, vol. 74,
    no. 1, pp 128–133, 2000.

    [2] https://pypi.org/project/pyproj/

    [3] Wavelet Variability Model - Matlab Code:
    https://pvpmc.sandia.gov/applications/wavelet-variability-model/
    """

    # Added by Joe Ranalli (@jranalli), Penn State Hazleton, 2019

    r_earth = 6371008.7714  # mean radius of Earth, in meters
    m_per_deg_lat = r_earth * np.pi / 180
    try:
        meanlat = np.mean([lat for (lat, lon) in coordinates])  # Mean latitude
    except TypeError:  # Assume it's a single value?
        meanlat = coordinates[0]
    m_per_deg_lon = r_earth * np.cos(np.pi/180 * meanlat) * np.pi/180

    # Conversion
    pos = coordinates * np.array(m_per_deg_lat, m_per_deg_lon)

    # reshape as (x,y) pairs to return
    try:
        return np.column_stack([pos[:, 1], pos[:, 0]])
    except IndexError:  # Assume it's a single value, which has a 1D shape
        return np.array((pos[1], pos[0]))


def _compute_wavelet(clearsky_index, dt=None):
    """
    Compute the wavelet transform on the input clear_sky time series.

    Parameters
    ----------
    clearsky_index : numeric or pandas.Series
        Clear Sky Index time series that will be smoothed.

    dt : float, default None
        The time series time delta. By default, is inferred from the
        clearsky_index. Must be specified for a time series that doesn't
        include an index. Units of seconds [s].

    Returns
    -------
    wavelet: numeric
        The individual wavelets for the time series

    tmscales: numeric
        The timescales associated with the wavelets in seconds [s]

    References
    ----------
    [1] M. Lave, J. Kleissl and J.S. Stein. A Wavelet-Based Variability
    Model (WVM) for Solar PV Power Plants. IEEE Transactions on Sustainable
    Energy, vol. 4, no. 2, pp. 501-509, 2013.

    [3] Wavelet Variability Model - Matlab Code:
    https://pvpmc.sandia.gov/applications/wavelet-variability-model/
    """

    # Added by Joe Ranalli (@jranalli), Penn State Hazleton, 2019

    try:  # Assume it's a pandas type
        vals = clearsky_index.values.flatten()
    except AttributeError:  # Assume it's a numpy type
        vals = clearsky_index.flatten()
        if dt is None:
            raise ValueError("dt must be specified for numpy type inputs.")
    else:  # flatten() succeeded, thus it's a pandas type, so get its dt
        try:  # Assume it's a time series type index
            dt = (clearsky_index.index[1] - clearsky_index.index[0]).seconds
        except AttributeError:  # It must just be a numeric index
            dt = (clearsky_index.index[1] - clearsky_index.index[0])

    # Pad the series on both ends in time and place in a dataframe
    cs_long = np.pad(vals, (len(vals), len(vals)), 'symmetric')
    cs_long = pd.DataFrame(cs_long)

    # Compute wavelet time scales
    min_tmscale = np.ceil(np.log(dt)/np.log(2))  # Minimum wavelet timescale
    max_tmscale = int(12 - min_tmscale)  # maximum wavelet timescale

    tmscales = np.zeros(max_tmscale)
    csi_mean = np.zeros([max_tmscale, len(cs_long)])
    # Loop for all time scales we will consider
    for i in np.arange(0, max_tmscale):
        j = i+1
        tmscales[i] = 2**j * dt  # Wavelet integration time scale
        intvlen = 2**j  # Wavelet integration time series interval
        # Rolling average, retains only lower frequencies than interval
        df = cs_long.rolling(window=intvlen, center=True, min_periods=1).mean()
        # Fill nan's in both directions
        df = df.fillna(method='bfill').fillna(method='ffill')
        # Pop values back out of the dataframe and store
        csi_mean[i, :] = df.values.flatten()

    # Calculate the wavelets by isolating the rolling mean frequency ranges
    wavelet_long = np.zeros(csi_mean.shape)
    for i in np.arange(0, max_tmscale-1):
        wavelet_long[i, :] = csi_mean[i, :] - csi_mean[i+1, :]
    wavelet_long[max_tmscale-1, :] = csi_mean[max_tmscale-1, :]  # Lowest freq

    # Clip off the padding and just return the original time window
    wavelet = np.zeros([max_tmscale, len(vals)])
    for i in np.arange(0, max_tmscale):
        wavelet[i, :] = wavelet_long[i, len(vals)+1: 2*len(vals)+1]

    return wavelet, tmscales
