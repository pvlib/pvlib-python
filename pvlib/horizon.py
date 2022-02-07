'''
The 'horizon' module contains function definitions that
retrive & calculate the surrounding horizon using DEM
elevation data
'''
import numpy as np
from scipy.signal import resample


def latlong(ds):
    r'''From a gdal dataset, retrive the geotransform
    and return an latitude and longitude coordinates
    for a DEM GeoTiff.

    Parameters
    ----------
    ds : tuple
        Geotransform parameters given by a osgeo.gdal.Dataset

    Returns
    -------
    lat,long : tuple
        Tuple of np.array of latitude,longitude
        corresponding to the pixels in the GeoTiff

    '''

    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5]
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]
    lat = np.linspace(miny, maxy, height)
    long = np.linspace(minx, maxx, width)
    return (lat, long)


def load_DEM(filepath):
    r'''Loads a DEM from a .hgt file, segments into the GeoTiff
    (elevation) and corresponding latitude and longitude

    Parameters
    ----------
    filepath : string
        Absolute or relative path to the .hgt file

    Returns
    -------
    elevation : np.array
        2D numpy array, pixel value is elevation in meters
    lat : np.array
        latitudes corresponding to the pixels along the 0th dim of elevation
    long : np.array
        longitudes corresponding to the pixels along the 1th dim of elevation
    '''
    import gdal
    from osgeo import gdal_array
    dataset = gdal.Open(filepath)
    rasterArray = gdal_array.LoadFile(filepath)
    elevation = rasterArray
    lat, long = latlong(dataset)
    return elevation, lat, long


def get_pixel_coords(lat, lon, DEM_path):
    r'''Gets pixel coordinates from the raster, given latitude and longitude

    Parameters
    ----------
    lat : float
        Latitude of the point
    long: float
        Longitude of the point
    DEM_path: string
        Path of the DEM .hgt file

    Returns
    -------
    coords : tuple
        Tuple with two elements, containing the x and y coordinate
        of the raster corresponding to latitiude and longitude
    '''
    from geoio import GeoImage
    img = GeoImage(DEM_path)
    return map(int, img.proj_to_raster(lon, lat))


def _point_symmetry(x, y):
    r"""Reflect a point 8 ways to form a circle.
    Parameters
    ----------
    x : numeric
        x point of the circle
    y: numeric
        y point of the circle

    Returns
    -------
    points : list
        List of reflected points
    """
    return [(x,  y),
            (y,  x),
            (-x,  y),
            (-y,  x),
            (x, -y),
            (y, -x),
            (-x, -y),
            (-y, -x)]


def _bresenham_circ(r, center=(0, 0)):
    r""" Use midpoint algorithum to build a rasterized circle.
    Modified from
    https://funloop.org/post/2021-03-15-bresenham-circle-drawing-algorithm.html#bresenhams-algorithm
    to add circles not centered at the origin

    Parameters
    ----------
    r : numeric
        Radius of the circle
    center: tuple
        Center of the point (Cartesian)

    Returns
    -------
    points : np.array
        Array of shape (n,2) of points that form a rasterized circle.
        n depends on the radius of the circle.
    """
    points = []
    x = 0
    y = -r
    F_M = 1 - r
    # Initial value for (0,-r) for 2x + 3 = 0x + 3 = 3.
    d_e = 3
    # Initial value for (0,-r) for 2(x + y) + 5 = 0 - 2y + 5 = -2y + 5.
    d_ne = -(r << 1) + 5
    points.extend(_point_symmetry(x, y))
    while x < -y:
        if F_M < 0:
            F_M += d_e
        else:
            F_M += d_ne
            d_ne += 2
            y += 1
        d_e += 2
        d_ne += 2
        x += 1
        points.extend(_point_symmetry(x, y))
    points = np.array(points)
    newx = points[:, 0] + center[0]
    newy = points[:, 1] + center[1]
    return np.vstack((newx, newy))


def _pol2cart(r, theta):
    r'''Converts polar to cartesian
    Parameters
    ----------
    r : np.array
        r values for the points
    theta: np.array
        theta values for the points

    Returns
    -------
    x : np.array
        x values for the points
    y : np.array
        y values for the points

    '''

    z = r * np.exp(1j * theta)
    x, y = z.real, z.imag
    return x, y


def _cart2pol(x, y):
    r'''Converts cartesian to polar
    Parameters
    ----------
    x : np.array
        x values for the points
    y : np.array
        y values for the points

    Returns
    -------
    r : np.array
        r values for the points
    theta: np.array
        theta values for the points

    '''

    z = x + y * 1j
    r, theta = np.abs(z), np.angle(z)
    return r, theta


def _sort_circ(pts, center=(0, 0), az_len=360):
    r'''Sort and resample points on a circle such that the zeroth element is
    due east and the points move around the circle counter clockwise.
    While in polar domain, resample points using FFT to
    obtain desired number of bins,typically 360, for
    degrees around the circle.

    Parameters
    ----------
    pts : np.array
        Array of shape (n, 2) of points in Cartesian system
    center : tuple
        Center (x,y) of the circle rasterized by pts
    az_len : numeric
        Desired number of points in the rasterized circle.

    Returns
    -------
    points : np.array
        Sorted and resampled points that comprise the circle
    '''
    pts[0] -= center[0]
    pts[1] -= center[1]
    r, theta = _cart2pol(pts[0], pts[1])
    stacked = np.vstack((theta, r))
    sort = np.sort(stacked, axis=1)
    sort = resample(sort, az_len, axis=1)
    theta = sort[0]
    r = sort[1]
    x, y = _pol2cart(r, theta)
    x += center[0]
    y += center[1]
    return np.vstack((x, y)).astype(int)


def horizon_map(dem_pixel, elevation, dem_res=30.0,
                view_distance=500, az_len=36):
    r"""Finds the horizon at point on a dem in pixel coordinates dem_pixel

    Parameters
    ----------
    dem_pixel : tuple (int,int)
        Point on the DEM expressed in pixel coordinates
    elevation : np.array
        nxn DEM of elevation values
    dem_res : float
        Resolution of the DEM. The default is SRTM 30m
    view_distance : int
        Radius of the area of consideration.
    az_len : int
        Number of samples on the perimeter of the circle.

    Returns
    -------
    azimuth : np.array
        Numpy array of shape (az_len). Linearly spaced from [0,360]
    elevation_angles: np.array
        The calculated elevation values at each point of azimuth
    profile:
        The largest elevation in meters on the line between the observer
        and the highest point on the horizon within view_distance

    """
    from skimage.draw import line
    azimuth = np.linspace(0, 360, az_len)

    # Use Midpoint Circle Algo/ Bresenham's circle
    pts = _bresenham_circ(view_distance, dem_pixel)

    # sort circle points and resample to desired number of points (az_len)
    pts = _sort_circ(pts, center=dem_pixel, az_len=az_len)
    x0, y0 = dem_pixel
    profile = np.zeros(azimuth.shape)
    elevation_angles = np.zeros(azimuth.shape)
    x_bound, y_bound = elevation.shape
    for az in range(az_len):
        source_lon = pts[1][az]
        source_lat = pts[0][az]
        # check for outmapping
        # for now, if out of the DEM, assign to the border
        if source_lon >= y_bound:
            source_lon = y_bound-1
        elif source_lon <= 0:
            source_lon = 0
        if source_lat >= y_bound:
            source_lat = y_bound-1
        elif source_lat <= 0:
            source_lat = 0

        target_lon = int(y0)
        target_lat = int(x0)

        # draw a line from observer to horizon and record points
        rr, cc = line(source_lon, source_lat, target_lon, target_lat)

        # get the highest elevation on the line
        elvs_on_line = elevation[rr, cc]
        idx = np.stack((rr, cc), axis=-1)
        highest_elv = np.max(elvs_on_line)
        highest_point = idx[np.argmax(elvs_on_line)]
        high_x, high_y = tuple(highest_point)

        # convert from altitude in m to elevation degrees.
        xdist = np.abs(highest_point[0]-x0)
        ydist = highest_elv - elevation[y0][x0]
        elv_ang = np.arctan(ydist/xdist)
        elevation_angles[az] = np.rad2deg(elv_ang)
        profile[az] = highest_elv
    return azimuth, elevation_angles, profile


def calculate_dtf(horizon_azimuths, horizon_angles,
                  surface_tilt, surface_azimuth):
    r"""Author: JPalakapillyKWH
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


def collection_plane_elev_angle(surface_tilt, surface_azimuth, direction):
    r"""Author: JPalakapillyKWH
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


def dni_horizon_adjustment(horizon_angles, solar_zenith, solar_azimuth):
    r'''Author: JPalakapillyKWH
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
