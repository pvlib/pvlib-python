.. _horizon:

Horizon
========

.. ipython:: python
   :suppress:

    import pandas as pd
    from pvlib import pvsystem


The :py:mod:`~pvlib.horizon` module contains many functions for horizon
profile modeling.

The horizon profile at a location is a mapping from azimuth to elevation angle
(also called dip angle). 

A source of elevation data is needed for many of the 
A common source of elevation data is NASA's SRTM [1]. An easily accessible source
of elevation data (albeit not free) is the googlemaps elevation API. There
are a few examples of how to query the googlemaps elevation API further below.


def fake_horizon_profile(max_dip):
    """
    Creates a bogus horizon profile by randomly generating dip_angles at
    integral azimuth values. Used for testing purposes.
    """
    fake_profile = []
    for i in range(-180, 181):
        fake_profile.append((i, random.random() * max_dip))
    return fake_profile



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