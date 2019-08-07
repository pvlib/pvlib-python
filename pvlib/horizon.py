"""
The ``horizon`` module contains functions for horizon profile modeling.
There are various geometric utilities that are useful in horizon calculations
as well as a method that uses the googlemaps elevation API to create a
horizon profile.
"""
from __future__ import division

import itertools

import numpy as np
from scipy.interpolate import RegularGridInterpolator

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


def elevation_angle_calc(pt1, pt2):
    '''
    Calculates the elevation angle from pt1 to pt2 where elevation angle is
    defined as the angle between the line connecting pt1 to pt2 and the plane
    normal to the Earth's surface at pt1. A point that appears above the
    horizontal has a positive elevation angle. Also computes the azimuth
    defined as degrees East of North the bearing of pt2 from pt1.
    This uses the Haversine formula.

    Parameters
    ----------
    pt1 : ndarray
        Nx3 array that contains lat, lon, and elev values that correspond
        to the origin points from which the elevation angles are to be
        calculated. The observer points.

    pt2 : ndarray
        Nx3 array that contains lat, lon, and elev values that correspond
        to the target points to which the elevation angles are to be
        calculated. The observee points.

    Returns
    -------
    bearing_deg: numeric
        The bearings from pt1 to pt2 in degrees East of North.

    elevation_angle_deg: numeric
        The elevation angles that the points in pt2 make with the horizontal
        as observed from the points in pt1. Given in degrees above the
        horizontal.

    '''
    a = 6378137.0
    b = 6356752.0

    lat1 = np.atleast_1d(pt1.T[0])
    lon1 = np.atleast_1d(pt1.T[1])
    elev1 = np.atleast_1d(pt1.T[2])
    lat2 = np.atleast_1d(pt2.T[0])
    lon2 = np.atleast_1d(pt2.T[1])
    elev2 = np.atleast_1d(pt2.T[2])

    # convert to radians
    phi1 = np.radians(lat1)
    theta1 = np.radians(lon1)
    phi2 = np.radians(lat2)
    theta2 = np.radians(lon2)

    v1 = tools.lle_to_xyz(np.stack([lat1, lon1, elev1], axis=1))
    v2 = tools.lle_to_xyz(np.stack([lat2, lon2, elev2], axis=1))
    x1 = np.atleast_1d(v1.T[0])
    y1 = np.atleast_1d(v1.T[1])
    z1 = np.atleast_1d(v1.T[2])

    delta = np.atleast_2d(np.subtract(v1, v2))

    normal = np.atleast_2d(np.stack([2*x1/a**2, 2*y1/a**2, 2*z1/b**2], axis=1))

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
                             sampling_method="grid", sampling_param=400):
    """
    Calculates a horizon profile from a three grids containing lat, lon,
    and elevation values. The "site" is assumed to be at the center of the
    grid.

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

    sampling_method : string
        A string that specifies the sampling method used to generate the
        horizon profile.

    sampling_param : variable
        A parameter that is passed into the function specified by
        sampling_method.

    Returns
    -------
    bearing_deg: Nx1 ndarray
        The bearings from the "site" to sampled points in degrees
        East of North.

    elevation_angle_deg: numeric
        The angles that the sampled points make with the horizontal
        as observed from the "site". Given in degrees above the
        horizontal.

    """
    assert(lat_grid.shape == lon_grid.shape == elev_grid.shape)

    grid_shape = lat_grid.shape
    grid_center_i = (grid_shape[0] - 1) // 2
    grid_center_j = (grid_shape[1] - 1) // 2
    site_lat = lat_grid[grid_center_i, grid_center_j]
    site_lon = lon_grid[grid_center_i, grid_center_j]
    site_elev = elev_grid[grid_center_i, grid_center_j]
    site = np.array([site_lat, site_lon, site_elev])

    if sampling_method == "grid":
        samples = sample_using_grid(lat_grid, lon_grid, elev_grid)
    elif sampling_method == "triangles":
        samples = sample_using_triangles(lat_grid, lon_grid, elev_grid,
                                         sampling_param)
    elif sampling_method == "interpolator":
        samples = sample_using_interpolator(lat_grid, lon_grid, elev_grid,
                                            sampling_param)

    bearing_deg, elevation_angle_deg = elevation_angle_calc(site, samples)

    return bearing_deg, elevation_angle_deg


def sample_using_grid(lat_grid, lon_grid, elev_grid):
    """
    Calculates the Elevation angle from the site (center of the grid)
    to every point on the grid.

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
    assert(lat_grid.shape == lon_grid.shape == elev_grid.shape)

    lats = lat_grid.flatten()
    lons = lon_grid.flatten()
    elevs = elev_grid.flatten()
    samples = np.stack([lats, lons, elevs], axis=1)
    # remove site from samples

    all_samples = np.delete(samples, samples.shape[0]//2, axis=0)
    return all_samples


def sample_using_triangles(lat_grid, lon_grid, elev_grid,
                           samples_per_triangle=10):
    """
    Creates triangles using nearest neighbors for every grid point and randomly
    samples each of these triangles to find elevation angles for the horizon
    profile.

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

    [1] http://graphics.stanford.edu/courses/cs468-08-fall/pdf/osada.pdf
    """
    assert(lat_grid.shape == lon_grid.shape == elev_grid.shape)

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
                samples = uniformly_sample_triangle(center,
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
                samples = uniformly_sample_triangle(center,
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
                samples = uniformly_sample_triangle(center,
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
                samples = uniformly_sample_triangle(center,
                                                    bottom,
                                                    right,
                                                    samples_per_triangle)
                all_samples = np.vstack([all_samples, samples])
    return np.array(all_samples)


def sample_using_interpolator(lat_grid, lon_grid, elev_grid, num_samples):
    """
    Creates a "grid" using polar coordinates and uses the scipy's grid
    interpolator to estimate elevation values at each point on the polar grid
    from the input (rectangular) grid that has true elevation values. Elevation
    calculations are done at each point on the polar grid and the results
    are returned.

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
    assert(lat_grid.shape == lon_grid.shape == elev_grid.shape)

    lats = lat_grid[0]
    lons = lon_grid.T[0]

    lat_range = lats[-1] - lats[0]

    grid_shape = lat_grid.shape
    grid_center_i = (grid_shape[0] - 1) // 2
    grid_center_j = (grid_shape[1] - 1) // 2
    site_lat = lat_grid[grid_center_i, grid_center_j]
    site_lon = lon_grid[grid_center_i, grid_center_j]

    interpolator = RegularGridInterpolator((lats, lons), elev_grid.T)

    r = np.linspace(0, lat_range//2, num_samples[0])
    theta = np.linspace(0, 2 * np.pi, num_samples[1])
    polar_pts = np.array(list(itertools.product(r, theta)))

    pts = np.array([tools.polar_to_cart(e[0], e[1]) for e in polar_pts])
    pts += np.array((site_lat, site_lon))
    total_num_samples = num_samples[0]*num_samples[1]

    interpolated_elevs = interpolator(pts).reshape(total_num_samples, 1)
    samples = np.hstack((pts, interpolated_elevs))
    return samples


def uniformly_sample_triangle(p1, p2, p3, num_samples):
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

    [1] http://graphics.stanford.edu/courses/cs468-08-fall/pdf/osada.pdf
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


def filter_points(horizon_azimuths, horizon_angles, bin_size=1):
    """
    Bins the horizon_points by azimuth values. The azimuth value of each
    point in horizon_points is rounded to the nearest bin and then the
    max value in each bin is returned.

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

    bin_size : int
        The width of the bins for the azimuth values.

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
    assert(horizon_azimuths.shape[0] == horizon_angles.shape[0])

    wedges = {}
    for i in range(horizon_angles.shape[0]):
        azimuth = horizon_azimuths[i]
        elevation = horizon_angles[i]
        azimuth_wedge = tools.round_to_nearest(azimuth, bin_size)

        if azimuth_wedge in wedges:
            wedges[azimuth_wedge] = max(elevation, wedges[azimuth_wedge])
        else:
            wedges[azimuth_wedge] = elevation

    filtered_azimuths = []
    filtered_angles = []
    for key in sorted(wedges.keys()):
        filtered_azimuths.append(key)
        filtered_angles.append(wedges[key])

    filtered_angles = np.array(filtered_angles)
    filtered_azimuths = np.array(filtered_azimuths)
    return filtered_azimuths, filtered_angles


def collection_plane_elev_angle(surface_tilt, surface_azimuth, direction):
    """
    Determine the elevation angle created by the surface of a tilted plane
    in a given direction. The angle is limited to be non-negative.

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
    Calculate the diffuse tilt factor that is adjusted with the horizon
    profile.

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

    """
    assert(horizon_azimuths.shape[0] == horizon_angles.shape[0])
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
