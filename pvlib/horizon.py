'''
The 'horizon' module contains function definitions that
retrive & calculate the surrounding horizon using DEM
elevation data.
Optional dependencies for this module include gdal, osgeo, geoio, and scikit-image. 
'''
import numpy as np
from scipy.signal import resample


def latlong(ds):
    r'''From a gdal dataset, retrieve the geotransform
    and return latitude and longitude coordinates
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

    Notes
    ------
    Latitude and longitude are in decimal degrees and negative longitude is west of the prime meridian. 
    '''
    import gdal
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


def get_pixel_coords(latitude, longitude, DEM_path):
    r'''Gets pixel coordinates from the raster, given latitude and longitude

    Parameters
    ----------
    latitude : float
        Latitude of the point
    longitude: float
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
    r : float
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
        theta values for the points. [radian]

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
        theta values for the points. [radian]

    '''

    z = x + y * 1j
    r, theta = np.abs(z), np.angle(z)
    return r, theta


def _sort_circ(pts, center=(0, 0), az_len=360):
    r'''Sort and resample points on a circle such that the zeroth element is
    due east and the points move around the circle counter clockwise.
    While in polar domain, resample points using FFT to
    obtain desired number of bins, typically 360, for
    degrees around the circle.

    Parameters
    ----------
    pts : np.array
        Array of shape (n, 2) of points in Cartesian system
    center : tuple
        Center (x,y) of the circle rasterized by pts
    az_len : int, default 360
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
    r"""Finds the horizon for the point ``dem_pixel``.

    Parameters
    ----------
    dem_pixel : tuple (int,int)
        Point on the DEM expressed in pixel coordinates
    elevation : np.array
        nxn DEM of elevation values
    dem_res : float, default 30
        Resolution of the DEM. The default is SRTM's 30m. [m]
    view_distance : float, default 500
        Radius of the area of consideration. [pixels]
    az_len : int, default 360
        Number of samples on the perimeter of the circle.

    Returns
    -------
    azimuth : np.array
        Numpy array of shape (az_len). Linearly spaced from [0,360)
    elevation_angles: np.array
        The calculated elevation values at each point of azimuth
    profile:
        The largest elevation in meters on the line between the observer
        and the highest point on the horizon within view_distance

    """
    from skimage.draw import line
    azimuth = np.linspace(0, 359, num=az_len)

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
        elv_ang = np.arctan2(ydist, xdist)
        elevation_angles[az] = np.rad2deg(elv_ang)
        profile[az] = highest_elv
    return azimuth, elevation_angles, profile
