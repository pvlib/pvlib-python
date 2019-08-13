"""
The ``horizon`` module contains functions for horizon profile modeling.
There are various geometric utilities that are useful in horizon calculations.
"""

import itertools

import numpy as np

from pvlib import tools


def grid_lat_lon(lat, lon, grid_radius=200, grid_step=.001):
    '''
    Uses numpy's meshgrid to create grids around a location (lat/lon pair)
    with a specified grid radius and step. The grid will be a square with
    (2xgrid_radius)+1 points along each side.

    Parameters
    ----------
    lat : numeric
        The latitude of the location that is to be the center of the grid.

    lon : numeric
        The longitude of the location that is to be the center of the grid.

    grid_radius : numeric
        The number of points to generate in each of the cardinal directions
        for the grid. The resulting grid will be a square with
        (2xgrid_radius)+1 points along each side. Unitless.

    grid_step : numeric
        The degrees of latitude/longitude between adjacent points in the grid.

    Returns
    -------
    lat_grid: 2d-array
        Latitude values at each point on the grid. Values will vary along
        axis=1 and will be constant along axis=0.

    lon_grid: 2d-array
        Longitude values at each point on the grid. Values will vary along
        axis=0 and will be constant along axis=1.
    '''

    lat_start = lat - (grid_radius * grid_step)
    lat_stop = lat + (grid_radius * grid_step)
    lats = np.linspace(lat_start, lat_stop, 2*grid_radius + 1)

    lon_start = lon - (grid_radius * grid_step)
    lon_stop = lon + (grid_radius * grid_step)
    lons = np.linspace(lon_start, lon_stop, 2*grid_radius + 1)

    lat_grid, lon_grid = np.meshgrid(lats, lons)

    return lat_grid, lon_grid


def elevation_and_azimuth(pt1, pt2):
    '''
    Calculates the elevation angle from pt1 to pt2. Elevation angle is
    defined as the angle between the line connecting pt1 to pt2 and the plane
    tangent to the Earth's surface at pt1. A point that appears above the
    horizontal has a positive elevation angle. Also computes the azimuth
    defined as degrees East of North of the bearing from pt1 to pt2.
    This uses the Haversine formula.
    The trigonometry used to calculate the elevation angle is described in [1].

    Parameters
    ----------
    pt1 : ndarray
        Nx3 array that contains latitude, longitude, and elevation values
        that correspond to the origin (observer) points from which the
        elevation angles are to be calculated. Longitude should be given in
        degrees East of the Prime Meridian and latitude in degrees North of the
        Equator. Units are [deg, deg, meters]

    pt2 : ndarray
        Nx3 array that contains latitude, longitude, and elevation values
        that correspond to the target (observee) points to which the elevation
        angles are to be calculated. Longitude should be given in
        degrees East of the Prime Meridian and latitude in degrees North of the
        Equator. Units are [deg, deg, meters]

    Returns
    -------
    bearing_deg: numeric
        The bearings from pt1 to pt2 in degrees East of North.

    elevation_angle_deg: numeric
        The elevation angles that the points in pt2 make with the horizontal
        as observed from the points in pt1. Given in degrees above the
        horizontal.

    Examples
    ________
    site_loc = np.array([[37, 34, 100]])
    target_locs = np.array([[38, 34, 63],
                            [36, 35, 231],
                            [36, 35, 21]])
    bearing, elev_angles = elevation_and_azimuth(site_loc, target_locs)


    [1] https://aty.sdsu.edu/explain/atmos_refr/dip.html
    '''
    # Equatorial Radius of the Earth (ellipsoid model) in meters
    a = 6378137.0
    # Polar Radius of the Earth (ellipsoid model) in meters
    b = 6356752.0

    lat1 = pt1.T[0]
    lon1 = pt1.T[1]
    lat2 = pt2.T[0]
    lon2 = pt2.T[1]

    # convert to radians
    phi1 = np.radians(lat1)
    theta1 = np.radians(lon1)
    phi2 = np.radians(lat2)
    theta2 = np.radians(lon2)

    v1 = tools.lle_to_xyz(pt1)
    v2 = tools.lle_to_xyz(pt2)
    x1 = v1.T[0]
    y1 = v1.T[1]
    z1 = v1.T[2]

    delta = np.subtract(v1, v2)
    a_sqrd = a**2
    b_sqrd = b**2
    normal = 2 * np.stack([x1/a_sqrd, y1/a_sqrd, z1/b_sqrd],
                          axis=1)

    # Take the dot product of corresponding vectors
    dot = np.sum(np.multiply(delta, normal), axis=1)
    beta = np.arccos(dot / np.linalg.norm(delta, axis=1)
                     / np.linalg.norm(normal, axis=1))
    elevation_angle = beta - np.pi/2

    elevation_angle_deg = np.degrees(elevation_angle)

    bearing = np.arctan2(np.sin(theta2-theta1)*np.cos(phi2),
                         (np.cos(phi1) * np.sin(phi2)
                          - np.sin(phi1) * np.cos(phi2)*np.cos(theta2-theta1)))
    bearing_deg = np.degrees(bearing)

    mask = (bearing_deg < 0)
    bearing_deg[mask] += 360

    return bearing_deg, elevation_angle_deg


def calculate_horizon_points(lat_grid, lon_grid, elev_grid,
                             sampling_method="grid", num_samples=None):
    """
    Calculates a horizon profile viewed from the center of a latitude,
    longitude, elevation grid.

    Parameters
    ----------
    lat_grid : ndarray
        A 2d array of latitude values for the grid.

    lon_grid : ndarray
        A 2d array of longitude values for the grid.

    elev_grid : ndarray
        A 2d array of elevation values for the grid.

    sampling_method : string, default "grid"
        A string that specifies the sampling method used to generate the
        horizon profile. Acceptable values are: "grid", "triangles", "polar".
        See Notes for brief descriptions of each.

    num_samples : variable, default None
        A parameter that is passed into the function specified by
        sampling_method.
        If the sampling method is "triangles" this corresponds
        to the number of samples taken from each triangle.
        See _sampling_using_triangles for more info.
        If the sampling method is "polar" this should be a tuple with 2 values
        that define the number of points along each polar axis to sample.
        See Notes for more info.

    Returns
    -------
    bearing_deg: Nx1 ndarray
        The bearings from the grid_center to horizon points.

    elevation_angle_deg: numeric
        The angles that the sampled points make with the horizontal
        as observed from the grid center. Given in degrees above the
        horizontal.

    Notes
    _____
    Sampling methods:
    "grid" - Uses every point on the grid exclusing the grid
    center as samples for hrizon calculations.

    "triangles" - Creates triangles using nearest neighbors for
    every grid point and randomly samples the surface of each of these
    triangles. num_samples sets the number of samples taken from each triangle.

    "polar" - Creates a polar "grid" and uses scipy's grid interpolator to
    estimate elevation values at each point on the polar grid from the true
    elevation data. num_samples sets the number of points along each polar
    axis (radial and angular).
    """

    grid_shape = lat_grid.shape
    grid_center_i = (grid_shape[0] - 1) // 2
    grid_center_j = (grid_shape[1] - 1) // 2
    center_lat = lat_grid[grid_center_i, grid_center_j]
    center_lon = lon_grid[grid_center_i, grid_center_j]
    center_elev = elev_grid[grid_center_i, grid_center_j]
    center = np.array([center_lat, center_lon, center_elev])

    if sampling_method == "grid":
        samples = _sample_using_grid(lat_grid, lon_grid, elev_grid)
    elif sampling_method == "triangles":
        samples = _sample_using_triangles(lat_grid, lon_grid, elev_grid,
                                          num_samples)
    elif sampling_method == "polar":
        samples = _sample_using_interpolator(lat_grid, lon_grid, elev_grid,
                                             num_samples)
    else:
        raise ValueError('Invalid sampling method: %s', sampling_method)

    bearing_deg, elevation_angle_deg = elevation_and_azimuth(center, samples)

    return bearing_deg, elevation_angle_deg


def _sample_using_grid(lat_grid, lon_grid, elev_grid):
    """
    Returns every point on the grid excluding the grid center as samples
    for horizon calculations.

    Parameters
    ----------
    lat_grid : ndarray
        A 2d array containing latitude values that correspond to the other
        two input grids.

    lon_grid : ndarray
        A 2d array containing longitude values that correspond to the other
        two input grids.

    elev_grid : ndarray
        A 2d array containing elevation values that correspond to the other
        two input grids.


    Returns
    -------
    all_samples: Nx3 ndarray
        Array of lat, lon, elev points that are grid points.
    """

    lats = lat_grid.flatten()
    lons = lon_grid.flatten()
    elevs = elev_grid.flatten()
    samples = np.stack([lats, lons, elevs], axis=1)
    # remove grid center from samples

    all_samples = np.delete(samples, samples.shape[0]//2, axis=0)
    return all_samples


def _sample_using_triangles(lat_grid, lon_grid, elev_grid,
                            samples_per_triangle=10):
    """
    Creates triangles using nearest neighbors for every grid point and randomly
    samples each of these triangles.

    Parameters
    ----------
    lat_grid : ndarray
        A 2d array containing latitude values that correspond to the other
        two input grids.

    lon_grid : ndarray
        A 2d array containing longitude values that correspond to the other
        two input grids.

    elev_grid : ndarray
        A 2d array containing elevation values that correspond to the other
        two input grids.

    samples_per_triangle : numeric
        The number of random samples to be uniformly taken from the surface
        of each triangle.

    Returns
    -------
    all_samples: Nx3 ndarray
        Array of [lat, lon, elev] points that were sampled from the grid.

    [1] Osada et al. (2002) ACM Transactions on Graphics. 21(4) 807-832
    """

    # start with empty array
    all_samples = np.array([], dtype=np.float64).reshape(0, 3)

    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            center = np.array([lat_grid[i, j],
                               lon_grid[i, j],
                               elev_grid[i, j]])
            if i != 0 and j != 0:
                left = np.array([lat_grid[i, j-1],
                                 lon_grid[i, j-1],
                                 elev_grid[i, j-1]])
                top = np.array([lat_grid[i-1, j],
                                lon_grid[i-1, j],
                                elev_grid[i-1, j]])
                samples = _uniformly_sample_triangle(center,
                                                     top,
                                                     left,
                                                     samples_per_triangle)
                all_samples = np.vstack([all_samples, samples])

            if i != 0 and j != lat_grid.shape[1] - 1:
                right = np.array([lat_grid[i, j+1],
                                  lon_grid[i, j+1],
                                  elev_grid[i, j+1]])
                top = np.array([lat_grid[i-1, j],
                                lon_grid[i-1, j],
                                elev_grid[i-1, j]])
                samples = _uniformly_sample_triangle(center,
                                                     top,
                                                     right,
                                                     samples_per_triangle)
                all_samples = np.vstack([all_samples, samples])

            if i != lat_grid.shape[0] - 1 and j != 0:
                left = np.array([lat_grid[i, j-1],
                                 lon_grid[i, j-1],
                                 elev_grid[i, j-1]])
                bottom = np.array([lat_grid[i+1, j],
                                   lon_grid[i+1, j],
                                   elev_grid[i+1, j]])
                samples = _uniformly_sample_triangle(center,
                                                     bottom,
                                                     left,
                                                     samples_per_triangle)
                all_samples = np.vstack([all_samples, samples])

            if i != lat_grid.shape[0] - 1 and j != lat_grid.shape[1] - 1:
                right = np.array([lat_grid[i, j+1],
                                  lon_grid[i, j+1],
                                  elev_grid[i, j+1]])
                bottom = np.array([lat_grid[i+1, j],
                                   lon_grid[i+1, j],
                                   elev_grid[i+1, j]])
                samples = _uniformly_sample_triangle(center,
                                                     bottom,
                                                     right,
                                                     samples_per_triangle)
                all_samples = np.vstack([all_samples, samples])
    return np.array(all_samples)


def _sample_using_interpolator(lat_grid, lon_grid, elev_grid, num_samples):
    """
    Creates a "grid" using polar coordinates and uses scipy's grid
    interpolator to estimate elevation values at each point on the polar grid
    from the input (rectangular) grid that has true elevation values.

    Parameters
    ----------
    lat_grid : ndarray
        A 2d array containing latitude values that correspond to the other
        two input grids.

    lon_grid : ndarray
        A 2d array containing longitude values that correspond to the other
        two input grids.

    elev_grid : ndarray
        A 2d array containing elevation values that correspond to the other
        two input grids.

    num_samples : tuple
        A tuple containing two integers. The first is the desired number of
        points along the radial axis of the polar grid. The second is the
        desired number of points along the angular axis of the polar grid.


    Returns
    -------
    all_samples: list
       Array of [lat, lon, elev] points that were sampled using the polar grid.

    """

    lats = lat_grid[0]
    lons = lon_grid.T[0]

    lat_range = lats[-1] - lats[0]

    grid_shape = lat_grid.shape
    grid_center_i = (grid_shape[0] - 1) // 2
    grid_center_j = (grid_shape[1] - 1) // 2
    center_lat = lat_grid[grid_center_i, grid_center_j]
    center_lon = lon_grid[grid_center_i, grid_center_j]

    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        raise ImportError('The polar sampling function requires scipy')

    interpolator = RegularGridInterpolator((lats, lons), elev_grid.T)

    r = np.linspace(0, lat_range//2, num_samples[0])
    theta = np.linspace(0, 2 * np.pi, num_samples[1])
    polar_pts = np.array(list(itertools.product(r, theta)))

    pts = np.array([tools.polar_to_cart(e[0], e[1]) for e in polar_pts])
    pts += np.array((center_lat, center_lon))
    total_num_samples = num_samples[0]*num_samples[1]

    interpolated_elevs = interpolator(pts).reshape(total_num_samples, 1)
    samples = np.hstack((pts, interpolated_elevs))
    return samples


def _uniformly_sample_triangle(p1, p2, p3, num_samples):
    """
    Randomly sample the surface of a triangle defined by three (lat, lon, elev)
    points uniformly [1].

    Parameters
    ----------
    pt1 : ndarray
        An array conaining (lat, lon, elev) values that define one vertex
        of the triangle.
    pt2 : ndarray
        An array conaining (lat, lon, elev) values that define another vertex
        of the triangle.
    pt3 : ndarray
        An array conaining (lat, lon, elev) values that define the last vertex
        of the triangle.

    num_samples : tuple
        The number of random samples to be uniformly taken from the surface
        of the triangle.

    Returns
    -------
    points: Nx3 ndarray
        Array with N (lat, lon, elev) points that lie on the surface of the
        triangle.

    [1] Osada et al. (2002) ACM Transactions on Graphics. 21(4) 807-832
    """
    c1 = tools.lle_to_xyz(p1)
    c2 = tools.lle_to_xyz(p2)
    c3 = tools.lle_to_xyz(p3)

    r1 = np.random.rand(num_samples).reshape((num_samples, 1))
    r2 = np.random.rand(num_samples).reshape((num_samples, 1))
    sqrt_r1 = np.sqrt(r1)

    random_pts = (1-sqrt_r1)*c1 + sqrt_r1*(1-r2)*c2 + sqrt_r1*r2*c3
    random_pts = tools.xyz_to_lle(random_pts)

    return random_pts


def filter_points(azimuths, elevation_angles, bin_size=1):
    """
    Bins the horizon points by azimuth values. The azimuth value of each
    point is rounded to the nearest bin and then the max elevation angle
    in each bin is returned. The bins will have azimuth values of n*bin_size
    where n is some integer.

    Parameters
    ----------
    azimuths: numeric
        Azimuth values for points that define the horizon profile. The ith
        element in this array corresponds to the ith element in
        elevation_angles.

    elevation_angles: numeric
        Elevation angle values for points that define the horizon profile. The
        elevation angle of the horizon is the angle that the horizon makes with
        the horizontal. It is given in degrees above the horizontal. The ith
        element in this array corresponds to the ith element in
        azimuths.

    bin_size : int
        The width of the bins for the azimuth values. (degrees)

    Returns
    -------
    filtered_azimuths: numeric
        Azimuth values for points that define the horizon profile. The ith
        element in this array corresponds to the ith element in
        filtered_angles.

    filtered_angles: numeric
        elevation angle values for points that define the horizon profile given
        in degrees above the horizontal. The ith element in this array
        corresponds to the ith element in filtered_azimuths.

    """
    if azimuths.shape[0] != elevation_angles.shape[0]:
        raise ValueError('azimuths and elevation_angles must be of the same'
                         'length.')

    rounded_azimuths = tools.round_to_nearest(azimuths, bin_size)
    bins = np.unique(rounded_azimuths)

    filtered = np.column_stack((bins, np.nan * bins))

    for i in range(filtered.shape[0]):
        idx = (rounded_azimuths == filtered[i, 0])
        filtered[i, 1] = np.max(elevation_angles[idx])

    return filtered[:, 0], filtered[:, 1]


def collection_plane_elev_angle(surface_tilt, surface_azimuth, direction):
    """
    Determine the elevation angle created by the surface of a tilted plane
    intersecting the plane tangent to the Earth's surface in a given direction.
    The angle is limited to be non-negative. This comes from Equation 10 in [1]

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. surface_tilt must be >=0
        and <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. surface_azimuth must
        be >=0 and <=360. The azimuth convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).

    direction : numeric
        The direction along which the elevation angle is to be calculated in
        decimal degrees. The convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).

    Returns
    --------

    elevation_angle : numeric
        The angle between the surface of the tilted plane and the horizontal
        when looking in the specified direction. Given in degrees above the
        horizontal and limited to be non-negative.


    [1] doi.org/10.1016/j.solener.2014.09.037

    """
    tilt = np.radians(surface_tilt)
    bearing = np.radians(direction - surface_azimuth - 180.0)

    declination = np.degrees(np.arctan(1.0/np.tan(tilt)/np.cos(bearing)))
    mask = (declination <= 0)
    elevation_angle = 90.0 - declination
    elevation_angle[mask] = 0.0

    return elevation_angle


def calculate_dtf(horizon_azimuths, horizon_angles,
                  surface_tilt, surface_azimuth):
    """
    Calculate the diffuse tilt factor for a tilted plane that is adjusted
    with for horizon profile. The idea for a diffuse tilt factor is explained
    in [1].

    Parameters
    ----------
    horizon_azimuths: numeric
        Azimuth values for points that define the horizon profile. The ith
        element in this array corresponds to the ith element in horizon_angles.

    horizon_angles: numeric
        Elevation angle values for points that define the horizon profile. The
        elevation angle of the horizon is the angle that the horizon makes with
        the horizontal. It is given in degrees above the horizontal. The ith
        element in this array corresponds to the ith element in
        horizon_azimuths.

    surface_tilt : numeric
        Surface tilt angles in decimal degrees. surface_tilt must be >=0
        and <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. surface_azimuth must
        be >=0 and <=360. The azimuth convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).

    Returns
    -------
    dtf: numeric
        The diffuse tilt factor that can be multiplied with the diffuse
        horizontal irradiance (DHI) to get the incident irradiance from
        the sky that is adjusted for the horizon profile and the tilt of
        the plane.

    Notes
    _____

    The dtf in this method is calculated by approximating the surface integral
    over the visible section of the sky dome. The integrand of the surface
    integral is the cosine of the angle between the incoming radiation and the
    vector normal to the surface. The method calculates a sum of integrations
    from the "peak" of the sky dome down to the elevation angle of the horizon.
    A similar method is used in section II of [1] although it is looking at
    both ground and sky diffuse irradiation.

    [2] Wright D. (2019) IEEE Journal of Photovoltaics 9(2), 391-396
    """
    if horizon_azimuths.shape[0] != horizon_angles.shape[0]:
        raise ValueError('azimuths and elevation_angles must be of the same'
                         'length.')
    tilt_rad = np.radians(surface_tilt)
    plane_az_rad = np.radians(surface_azimuth)
    a = np.sin(tilt_rad) * np.cos(plane_az_rad)
    b = np.sin(tilt_rad) * np.sin(plane_az_rad)
    c = np.cos(tilt_rad)

    # this gets either a float or an array of zeros
    dtf = np.multiply(0.0, surface_tilt)
    num_points = horizon_azimuths.shape[0]
    for i in range(horizon_azimuths.shape[0]):
        az = np.radians(horizon_azimuths[i])
        horizon_elev = np.radians(horizon_angles[i])
        temp = np.radians(collection_plane_elev_angle(surface_tilt,
                                                      surface_azimuth,
                                                      horizon_azimuths[i]))
        elev = np.maximum(horizon_elev, temp)

        first_term = .5 * (a*np.cos(az) + b*np.sin(az)) * \
                          (np.pi/2 - elev - np.sin(elev) * np.cos(elev))
        second_term = .5 * c * np.cos(elev)**2
        dtf += 2 * (first_term + second_term) / num_points
    return dtf


def dni_horizon_adjustment(horizon_angles, solar_zenith, solar_azimuth):
    '''
    Calculates an adjustment to direct normal irradiance based on a horizon
    profile. The adjustment is a vector of binary values with the same length
    as the provided solar position values. Where the sun is below the horizon,
    the adjustment vector is 0 and it is 1 elsewhere. The horizon profile must
    be given as a vector with 360 values where the ith value corresponds to the
    ith degree of azimuth (0-359).


    Parameters
    ----------
    horizon_angles: numeric
        Elevation angle values for points that define the horizon profile. The
        elevation angle of the horizon is the angle that the horizon makes with
        the horizontal. It is given in degrees above the horizontal. The ith
        element in this array corresponds to the ith degree of azimuth.

    solar_zenith : numeric
        Solar zenith angle.

    solar_azimuth : numeric
        Solar azimuth angle.

    Returns
    -------
    adjustment : numeric
        A vector of binary values with the same shape as the inputted solar
        position values. 0 when the sun is below the horizon and 1 elsewhere.
    '''
    adjustment = np.ones(solar_zenith.shape)

    if (horizon_angles.shape[0] != 360):
        raise ValueError('horizon_angles must contain exactly 360 values'
                         '(for each degree of azimuth 0-359).')

    rounded_solar_azimuth = np.round(solar_azimuth).astype(int)
    rounded_solar_azimuth[rounded_solar_azimuth == 360] = 0
    horizon_zenith = 90 - horizon_angles[rounded_solar_azimuth]
    mask = solar_zenith > horizon_zenith
    adjustment[mask] = 0
    return adjustment
