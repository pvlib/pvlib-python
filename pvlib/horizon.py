"""
The ``horizon`` module contains functions for horizon profile modeling.
There are various geometric utilities that are useful in horizon calculations
as well as a method that uses the googlemaps elevation API to create a
horizon profile.
"""

import random
import itertools

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pvlib import tools
import matplotlib.pyplot as plt


def grid_lat_lon(lat, lon, grid_size=200, grid_step=.001):
    '''
    Creates a grid around a location (lat/lon pair) with a specified
    grid size and step. The returned grid will be a 3-d matrix with
    shape: grid_size+1 x grid_size+1 x 2.
    '''
    grid = np.ndarray((grid_size + 1, grid_size + 1, 2))

    # fill out grid
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            grid[i, j, 0] = lat + (i - grid_size / 2) * grid_step
            grid[i, j, 1] = lon + (j - grid_size / 2) * grid_step

    return grid


def grid_elevations_from_gmaps(in_grid, GMAPS_API_KEY):
    """
    Takes in a grid of lat lon values (shape: grid_size+1 x grid_size+1 x 2).
    Queries the googlemaps elevation API to get elevation data at each lat/lon
    point. Outputs the original grid with the elevation data appended along
    the third axis so the shape is grid_size+1 x grid_size+1 x 3.
    """

    import googlemaps

    in_shape = in_grid.shape
    lats = in_grid.T[0].flatten()
    longs = in_grid.T[1].flatten()
    locations = zip(lats, longs)
    gmaps = googlemaps.Client(key=GMAPS_API_KEY)

    out_grid = np.ndarray((in_shape[0], in_shape[1], 3))

    # Get elevation data from gmaps
    elevations = []
    responses = []

    while len(locations) > 512:
        locations_to_request = locations[:512]
        locations = locations[512:]
        responses += gmaps.elevation(locations=locations_to_request)
    responses += gmaps.elevation(locations=locations)
    for entry in responses:
        elevations.append(entry["elevation"])

    for i in range(in_shape[0]):
        for j in range(in_shape[1]):
            lat = in_grid[i, j, 0]
            lon = in_grid[i, j, 1]
            elevation = elevations[i + j * in_shape[1]]

            out_grid[i, j, 0] = lat
            out_grid[i, j, 1] = lon
            out_grid[i, j, 2] = elevation
    return out_grid


def dip_calc(pt1, pt2):
    '''
    input: two LLE tuples
    output: distance, dip angle, azimuth

    Calculates the dip angle from pt1 to pt2 where dip angle is defined as
    the angle that the line connecting pt1 to pt2 makes with the plane normal
    to the Earth's surface at pt2. Also computes the azimuth defined as degrees
    East of North the bearing of pt2 from pt1. This uses the Haversine formula.

    Parameters
    ----------
    pt1 : tuple
        (lat, lon, elev) tuple that corresponds to the origin from which
        the dip angle is to be calculated. The observer point.

    pt1 : tuple
        (lat, lon, elev) tuple that corresponds to the origin from which
        the dip angle is to be calculated. The observee point.

    Returns
    -------
    bearing_deg: numeric
        The bearing from pt1 to pt2 in degrees East of North

    dip_angle:
        The dip angle that pt2 makes with the horizontal as observed at pt1.

    '''
    a = 6378137.0
    b = 6356752.0

    lat1 = pt1[0]
    lon1 = pt1[1]
    elev1 = pt1[2]
    lat2 = pt2[0]
    lon2 = pt2[1]
    elev2 = pt2[2]

    # convert to radians
    phi1 = lat1*np.pi/180.0
    theta1 = lon1*np.pi/180.0
    phi2 = lat2*np.pi/180.0
    theta2 = lon2*np.pi/180.0

    v1 = tools.lle_to_xyz((lat1, lon1, elev1))
    v2 = tools.lle_to_xyz((lat2, lon2, elev2))

    x1 = v1[0]
    y1 = v1[1]
    z1 = v1[2]

    delta = np.subtract(v1, v2)

    normal = np.array((2*x1/a**2, 2*y1/a**2, 2*z1/b**2))
    beta = np.arccos(np.dot(delta, normal)/np.linalg.norm(delta)
                     / np.linalg.norm(normal))
    dip_angle = beta - np.pi/2
    dip_angle_deg = dip_angle*180.0/np.pi

    # might wanna double check this formula (haversine?)
    bearing = np.arctan2(np.sin(theta2-theta1)*np.cos(phi2),
                         (np.cos(phi1) * np.sin(phi2)
                          - np.sin(phi1) * np.cos(phi2)*np.cos(theta2-theta1)))
    bearing_deg = bearing*180.0/np.pi
    if bearing_deg < 0:
        bearing_deg += 360

    return (bearing_deg, dip_angle_deg)


def calculate_horizon_points(grid, sampling_method="grid", sampling_param=400):
    """
    Calculates a horizon profile from a grid of (lat, lon, elev) tuples.
    The "site" is assumed to be at the center of the grid.

    Parameters
    ----------
    grid : ndarray
        Assumes

    sampling_method : string
        A string that specifies the sampling method used to generate the
        horizon profile.

    sampling_param : variable
        A parameter that is passed into the function specified by
        sampling_method.

    Returns
    -------
    horizon_points: list
        List of (azimuth, dip_angle) tuples that define the horizon at the
        point at the center of the grid.

    """

    grid_shape = grid.shape
    grid_center_i = (grid_shape[0] - 1) / 2
    grid_center_j = (grid_shape[1] - 1) / 2
    site_lat = grid[grid_center_i, grid_center_j, 0]
    site_lon = grid[grid_center_i, grid_center_j, 1]
    site_elev = grid[grid_center_i, grid_center_j, 2]
    site = (site_lat, site_lon, site_elev)

    horizon_points = []

    if sampling_method == "grid":
        samples = sample_using_grid(grid)
    elif sampling_method == "triangles":
        samples = sample_using_triangles(grid, sampling_param)
    elif sampling_method == "interpolator":
        samples = sample_using_interpolator(grid, sampling_param)

    horizon_points = np.array(list(map(lambda pt: dip_calc(site, pt),
                                       samples)))

    return horizon_points


def sample_using_grid(grid):
    """
    Calculates the dip angle from the site (center of the grid)
    to every point on the grid and uses the results as the
    horizon profile.

    """
    grid_shape = grid.shape
    grid_center_i = (grid_shape[0] - 1) / 2
    grid_center_j = (grid_shape[1] - 1) / 2

    all_samples = []
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # make sure the site is excluded
            if i != grid_center_i or j != grid_center_j:
                lat = grid[i, j, 0]
                lon = grid[i, j, 1]
                elev = grid[i, j, 2]
                all_samples.append((lat, lon, elev))
    return all_samples


def sample_using_triangles(grid, samples_per_triangle=10):
    """
    Creates triangles using nearest neighbors for every grid point and randomly
    samples each of these triangles to find dip angles for the horizon profile.

    Parameters
    ----------
    grid : ndarray
        Grid that contains lat lon and elev information.

    samples_per_triangle : numeric
        The number of random samples to be uniformly taken from the surface
        of each triangle.

    Returns
    -------
    all_samples: list
        List of (azimuth, dip_angle) tuples that were sampled from the grid

    [1] http://graphics.stanford.edu/courses/cs468-08-fall/pdf/osada.pdf
    """

    all_samples = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            center = (grid[i, j, 0], grid[i, j, 1], grid[i, j, 2])
            if i != 0 and j != 0:
                left = (grid[i, j-1, 0], grid[i, j-1, 1], grid[i, j-1, 2])
                top = (grid[i-1, j, 0], grid[i-1, j, 1], grid[i-1, j, 2])
                all_samples += uniformly_sample_triangle(center,
                                                         top,
                                                         left,
                                                         samples_per_triangle)

            if i != 0 and j != grid.shape[1] - 1:
                right = (grid[i, j+1, 0], grid[i, j+1, 1], grid[i, j+1, 2])
                top = (grid[i-1, j, 0], grid[i-1, j, 1], grid[i-1, j, 2])
                all_samples += uniformly_sample_triangle(center,
                                                         top,
                                                         right,
                                                         samples_per_triangle)

            if i != grid.shape[0] - 1 and j != 0:
                left = (grid[i, j-1, 0], grid[i, j-1, 1], grid[i, j-1, 2])
                bottom = (grid[i+1, j, 0], grid[i+1, j, 1], grid[i+1, j, 2])
                all_samples += uniformly_sample_triangle(center,
                                                         bottom,
                                                         left,
                                                         samples_per_triangle)

            if i != grid.shape[0] - 1 and j != grid.shape[1] - 1:
                right = (grid[i, j+1, 0], grid[i, j+1, 1], grid[i, j+1, 2])
                bottom = (grid[i+1, j, 0], grid[i+1, j, 1], grid[i+1, j, 2])
                all_samples += uniformly_sample_triangle(center,
                                                         bottom,
                                                         right,
                                                         samples_per_triangle)
    return all_samples


def sample_using_interpolator(grid, num_samples):
    """
    Creates a "grid" using polar coordinates and uses the scipy's grid
    interpolator to estimate elevation values at each point on the polar grid
    from the input (rectangular) grid that has true elevation values. Dip
    calculations are done at each point on the polar grid and the results
    are returned.

    Parameters
    ----------
    grid : ndarray
        Grid that contains lat lon and elev information.

    num_samples : tuple
        A tuple containing two integers. The first is the desired number of
        points along the radial axis of the polar grid. The second is the
        desired number of points along the angular axis of the polar grid.


    Returns
    -------
    all_samples: list
        List of (azimuth, dip_angle) tuples taken from the polar grid

    """
    x = grid.T[0][0]
    y = grid.T[1].T[0]

    x_range = x[-1] - x[0]

    grid_shape = grid.shape
    grid_center_i = (grid_shape[0] - 1) // 2
    grid_center_j = (grid_shape[1] - 1) // 2
    site_lat = grid[grid_center_i, grid_center_j, 0]
    site_lon = grid[grid_center_i, grid_center_j, 1]

    elevs = grid.T[2].T
    interpolator = RegularGridInterpolator((x, y), elevs)

    r = np.linspace(0, x_range/2, num_samples[0])
    theta = np.linspace(0, 2 * np.pi, num_samples[1])
    polar_pts = np.array(list(itertools.product(r, theta)))

    pts = np.array([tools.polar_to_cart(e[0], e[1]) for e in polar_pts])
    pts += np.array((site_lat, site_lon))
    total_num_samples = num_samples[0]*num_samples[1]

    interpolated_elevs = interpolator(pts).reshape(total_num_samples, 1)
    samples = np.concatenate((pts, interpolated_elevs), axis=1)
    return samples


def uniformly_sample_triangle(p1, p2, p3, num_samples):
    """
    Randomly sample the surface of a triangle defined by three (lat, lon, elev)
    points uniformly [1].

    Parameters
    ----------
    pt1 : tuple
        A (lat, lon, elev) tuple that defines one vertex of the triangle.
    pt2 : tuple
        A (lat, lon, elev) tuple that defines another vertex of the triangle.
    pt3 : tuple
        A (lat, lon, elev) tuple that defines the last vertex of the triangle.

    num_samples : tuple
        The number of random samples to be uniformly taken from the surface
        of the triangle.

    Returns
    -------
    points: list
        List of (lat, lon, elev) tuples that lie on the surface of the
        triangle.

    [1] http://graphics.stanford.edu/courses/cs468-08-fall/pdf/osada.pdf
    """
    c1 = tools.lle_to_xyz(p1)
    c2 = tools.lle_to_xyz(p2)
    c3 = tools.lle_to_xyz(p3)

    points = []
    for i in range(num_samples):
        r1 = np.random.rand()
        r2 = np.random.rand()
        sqrt_r1 = np.sqrt(r1)

        random_pt = (1-sqrt_r1)*c1 + sqrt_r1*(1-r2)*c2 + sqrt_r1*r2*c3
        points.append(tools.xyz_to_lle(random_pt))
    return points


def filter_points(horizon_points, bin_size=1):
    """
    Bins the horizon_points by azimuth values. The azimuth value of each
    point in horizon_points is rounded to the nearest bin and then the
    max value in each bin is returned.

    Parameters
    ----------
    horizon_points : list
        List of (azimuth, dip_angle) tuples that define the horizon.

    bin_size : int
        The bin size of azimuth values.

    Returns
    -------
    sorted_points: list
        List of (azimuth, dip_angle) values that correspond to the greatest
        dip_angle in each azimuth bin.
    """
    wedges = {}
    for pair in horizon_points:
        azimuth = pair[0]
        dip = pair[1]
        azimuth_wedge = tools.round_to_nearest(azimuth, bin_size)

        if azimuth_wedge in wedges:
            wedges[azimuth_wedge] = max(dip, wedges[azimuth_wedge])
        else:
            wedges[azimuth_wedge] = dip

    filtered_points = []
    for key in wedges.keys():
        filtered_points.append((key, wedges[key]))

    sorted_points = sorted(filtered_points, key=lambda tup: tup[0])
    return sorted_points


def visualize(horizon_profile, pvsyst_scale=False):
    """
    Plots a horizon profile with azimuth on the x-axis and dip angle on the y.
    """
    azimuths = []
    dips = []
    for pair in horizon_profile:
        azimuth = pair[0]
        azimuths.append(azimuth)
        dips.append(pair[1])
    plt.figure(figsize=(10, 6))
    if pvsyst_scale:
        plt.ylim(0, 90)
    plt.plot(azimuths, dips, "-")
    plt.show


def polar_plot(horizon_profile):
    """
    Plots a horizon profile on a polar plot with dip angle as the raidus and
    azimuth as the theta value. An offset of 5 is added to the dip_angle to
    make the plot more readable with low dip angles.
    """
    azimuths = []
    dips = []
    for pair in horizon_profile:
        azimuth = pair[0]
        azimuths.append(np.radians(azimuth))
        dips.append(pair[1] + 5)
    plt.figure(figsize=(10, 6))
    sp = plt.subplot(1, 1, 1, projection='polar')
    sp.set_theta_zero_location('N')
    sp.set_theta_direction(-1)
    plt.plot(azimuths, dips, "o")
    plt.show


def invert_for_pvsyst(horizon_points, hemisphere="north"):
    """
    Modify the azimuth values in horizon_points to match PVSyst's azimuth
    convention (which is dependent on hemisphere)
    """

    # look at that northern hemisphere bias right there
    # not even sorry.
    assert hemisphere == "north" or hemisphere == "south"

    inverted_points = []
    for pair in horizon_points:
        azimuth = pair[0]
        if hemisphere == "north":
            azimuth -= 180
            if azimuth < -180:
                azimuth += 360
        elif hemisphere == "south":
            azimuth = -azimuth
        inverted_points.append((azimuth, pair[1]))
    sorted_points = sorted(inverted_points, key=lambda tup: tup[0])
    return sorted_points


def horizon_from_gmaps(lat, lon, GMAPS_API_KEY):
    """
    Uses the functions defined in this modules to generate a complete horizon
    profile for a location (specified by lat/lon). An API key for the
    googlemaps elevation API is needeed.
    """

    grid = grid_lat_lon(lat, lon, grid_size=400, grid_step=.002)
    elev_grid = grid_elevations_from_gmaps(grid, GMAPS_API_KEY)
    horizon_points = calculate_horizon_points(elev_grid,
                                              sampling_method="interpolator",
                                              sampling_param=(1000, 1000))
    filtered_points = filter_points(horizon_points, bin_size=1)
    return filtered_points


def fake_horizon_profile(max_dip):
    """
    Creates a bogus horizon profile by randomly generating dip_angles at
    integral azimuth values. Used for testing purposes.
    """
    fake_profile = []
    for i in range(-180, 181):
        fake_profile.append((i, random.random() * max_dip))
    return fake_profile


def collection_plane_dip_angle(surface_tilt, surface_azimuth, direction):
    """
    Determine the dip angle created by the surface of a tilted plane
    in a given direction. The dip angle is limited to be non-negative.

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
        The direction along which the dip angle is to be calculated in
        decimal degrees. The convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).

    Returns
    --------

    dip_angle : numeric
        The dip angle in direction created by the tilted plane.

    """
    tilt = np.radians(surface_tilt)
    bearing = np.radians(direction - surface_azimuth - 180.0)

    declination = np.degrees(np.arctan(1.0/np.tan(tilt)/np.cos(bearing)))
    mask = (declination <= 0)
    dip = 90.0 - declination
    dip[mask] = 0.0

    return dip


def calculate_dtf(horizon_profile, surface_tilt, surface_azimuth):
    """
    Calculate the diffuse tilt factor that is adjusted with the horizon
    profile.
    """
    tilt_rad = np.radians(surface_tilt)
    plane_az_rad = np.radians(surface_azimuth)
    a = np.sin(tilt_rad) * np.cos(plane_az_rad)
    b = np.sin(tilt_rad) * np.sin(plane_az_rad)
    c = np.cos(tilt_rad)

    # this gets either an int or an array of zeros
    dtf = np.multiply(0.0, surface_tilt)
    for pair in horizon_profile:
        az = np.radians(pair[0])
        horizon_dip = np.radians(pair[1])
        temp = np.radians(collection_plane_dip_angle(surface_tilt,
                                                     surface_azimuth,
                                                     pair[0]))
        dip = np.maximum(horizon_dip, temp)

        first_term = .5 * (a*np.cos(az) + b*np.sin(az)) * \
                          (np.pi/2 - dip - np.sin(dip) * np.cos(dip))
        second_term = .5 * c * np.cos(dip)**2
        dtf += 2 * (first_term + second_term) / len(horizon_profile)
    return dtf
