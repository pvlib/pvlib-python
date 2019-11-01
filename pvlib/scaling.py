"""
The ``scaling`` module contains functions for manipulating irradiance
or other variables to account for temporal or spatial characteristics.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import scipy.optimize


def wvm(cs_series, latitude, longitude, cloud_speed,
        method, capacity=None, density=41, dt=None):
    """
    Compute spatial aggregation time series smoothing on clear sky index based
    on the Wavelet Variability model of Lave et al [1-2]. Implementation is
    basically a port of the Matlab version of the code [3].

    Parameters
    ----------
    cs_series : numeric or pandas.Series
        Clear Sky Index time series that will be smoothed.

    latitude : numeric
        Latitudes making up the plant in degrees. Type depends on method used.
        'discrete' uses a list of lat/lon points representing discrete sites.
        'square' uses a single valued lat/lon pair for the plant center.
        'polygon' uses a list of lat/lon points representing the corners of
        a polygon making up the site.

    longitude : numeric
        Longitudes making up the plant in degrees. Type depends on method used.
        'discrete' uses a list of lat/lon points representing discrete sites.
        'square' uses a single valued lat/lon pair for the plant center.
        'polygon' uses a list of lat/lon points representing the corners of
        a polygon making up the site.

    cloud_speed : numeric
        Speed of cloud movement in meters per second

    method : string
        The type of plant distribution to model.
        Options are ``'discrete', 'square', 'polygon'``.

    capacity : numeric
        The plant capacity in MW. Must be specified for 'square' method,
        ignored otherwise.

    density : numeric, default 41
        The density of installed PV in W/m^2. Must be specified for 'square'
        method, ignored otherwise. Default value of 41 W/m^2 is 1MW per 6 acres.

    dt : numeric
        The time series time delta. By default, is inferred from the cs_series.
        Must be specified for a time series that doesn't include an index.

    Returns
    -------
    smoothed : pandas.Series or numeric, depending on inlet type
        The clear sky index time series smoothed for the described plant.

    wavelet: numeric
        The individual wavelets for the time series before smoothing

    tmscales: numeric
        The timescales (in sec) associated with the wavelets

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

    dist = _compute_distances(longitude, latitude, method, capacity, density)
    wavelet, tmscales = _compute_wavelet(cs_series, dt)

    n_pairs = len(dist)
    # Find eff length of position vector, 'dist' is full pairwise
    fn = lambda x: np.abs((x ** 2 - x) / 2 - n_pairs)
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
        smoothed = pd.Series(outsignal, index=cs_series.index)
    except AttributeError:
        smoothed = outsignal  # just output the numpy signal

    return smoothed, wavelet, tmscales


def _latlon_to_dist(latitude, longitude):
    """
    Convert latitude and longitude in degrees to a coordinate system measured
    in meters from zero deg latitude, zero deg longitude.

    Parameters
    ----------
    latitude : numeric
        Latitude in degrees

    longitude : numeric
        Longitude in degrees

    Returns
    -------
    ypos : numeric
        the northward distance in meters

    xpos: numeric
        the eastward distance in meters.

    References
    ----------
    [1] H. Moritz. Geodetic Reference System 1980, Journal of Geodesy, vol. 74,
    no. 1, pp 128–133, 2000.

    [2] Wavelet Variability Model - Matlab Code:
    https://pvpmc.sandia.gov/applications/wavelet-variability-model/
    """

    # Added by Joe Ranalli (@jranalli), Penn State Hazleton, 2019

    r_earth = 6371008.7714  # mean radius of Earth, in meters
    m_per_deg_lat = r_earth * np.pi / 180
    m_per_deg_lon = r_earth * np.cos(np.pi/180*np.mean(latitude)) * np.pi/180

    try:
        xpos = m_per_deg_lon * longitude
        ypos = m_per_deg_lat * latitude
    except TypeError:  # When longitude and latitude are a list
        xpos = m_per_deg_lon * np.array(longitude)
        ypos = m_per_deg_lat * np.array(latitude)

    return ypos, xpos


def _compute_distances(longitude, latitude, method, capacity=None, density=41):
    """
    Compute points representing a plant and the pairwise distances between them

    Parameters
    ----------
    latitude : numeric
        Latitudes making up the plant in degrees. Type depends on method used.
        'discrete' uses a list of lat/lon points representing discrete sites.
        'square' uses a single valued lat/lon pair for the plant center.
        'polygon' uses a list of lat/lon points representing the corners of
        a polygon making up the site.

    longitude : numeric
        Longitudes making up the plant in degrees. Type depends on method used.
        'discrete' uses a list of lat/lon points representing discrete sites.
        'square' uses a single valued lat/lon pair for the plant center.
        'polygon' uses a list of lat/lon points representing the corners of
        a polygon making up the site.

    cloud_speed : numeric
        Speed of cloud movement in meters per second

    method : string
        The type of plant distribution to model.
        Options are ``'discrete', 'square', 'polygon'``.

    capacity : numeric
        The plant capacity in MW. Must be specified for 'square' method,
        ignored otherwise.

    density : numeric, default 41
        The density of installed PV in W/m^2. Must be specified for 'square'
        method, ignored otherwise. Default value of 41 W/m^2 is 1MW per 6 acres.

    Returns
    -------
    dist : numeric
        The complete set of pairwise distances of all points representing the
        plant being modelled.


    References
    ----------
    [1] Wavelet Variability Model - Matlab Code:
    https://pvpmc.sandia.gov/applications/wavelet-variability-model/
    """

    # Convert latitude and longitude points to distances in meters
    ypos, xpos = _latlon_to_dist(latitude, longitude)

    if method == "discrete":
        # Positions are individual plant centers.
        # Treat as existing subscale points.
        pos = np.array([xpos, ypos]).transpose()

    elif method == "square":
        raise NotImplementedError("To be implemented")

    elif method == "polygon":
        raise NotImplementedError("To be implemented")

    else:
        raise ValueError("Plant distance calculation method must be one of: "
                         "discrete, square, polygon")

    # Compute the full list of point-to-point distances
    return pdist(pos, 'euclidean')


def _compute_wavelet(cs_series, dt=None):
    """
    Compute the wavelet transform on the input clear_sky time series.

    Parameters
    ----------
    cs_series : numeric or pandas.Series
        Clear Sky Index time series that will be smoothed.

    dt : numeric
        The time series time delta. By default, is inferred from the cs_series.
        Must be specified for a time series that doesn't include an index.

    Returns
    -------
    wavelet: numeric
        The individual wavelets for the time series before smoothing

    tmscales: numeric
        The timescales (in sec) associated with the wavelets

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
        vals = cs_series.values.flatten()
        try:  # Assume it's a time series type index
            dt = (cs_series.index[1] - cs_series.index[0]).seconds
        except AttributeError:  # It must just be a numeric index
            dt = (cs_series.index[1] - cs_series.index[0])
    except AttributeError:  # Assume it's a numpy type
        vals = cs_series.flatten()
        if dt is None:
            raise ValueError("dt must be specified for numpy type inputs.")

    # Pad the series on both ends in time and place in a dataframe
    cs_long = np.pad(vals, (len(vals), len(vals)), 'symmetric')
    cs_long = pd.DataFrame(cs_long)

    # Compute wavelet time scales
    mindt = np.ceil(np.log(dt)/np.log(2))  # Minimum wavelet dt
    maxdt = int(12 - mindt)  # maximum wavelet dt

    tmscales = np.zeros(maxdt)
    csi_mean = np.zeros([maxdt, len(cs_long)])
    # Loop for all time scales we will consider
    for i in np.arange(0, maxdt):
        j = i+1
        tmscales[i] = 2**j * dt  # Wavelet integration time scale
        intvllen = 2**j  # Wavelet integration time series interval
        # Rolling average, retains only lower frequencies than interval
        df = cs_long.rolling(window=intvllen, center=True, min_periods=1).mean()
        # Fill nan's in both directions
        df = df.fillna(method='bfill').fillna(method='ffill')
        # Pop values back out of the dataframe and store
        csi_mean[i, :] = df.values.flatten()

    # Calculate the wavelets by isolating the rolling mean frequency ranges
    wavelet_long = np.zeros(csi_mean.shape)
    for i in np.arange(0, maxdt-1):
        wavelet_long[i, :] = csi_mean[i, :] - csi_mean[i+1, :]
    wavelet_long[maxdt-1, :] = csi_mean[maxdt-1, :]  # Lowest frequency

    # Clip off the padding and just return the original time window
    wavelet = np.zeros([maxdt, len(vals)])
    for i in np.arange(0, maxdt):
        wavelet[i, :] = wavelet_long[i, len(vals)+1: 2*len(vals)+1]

    return wavelet, tmscales
