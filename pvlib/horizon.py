import random
import pytz
import time 
import itertools

import numpy as np
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
import pandas as pd

import matplotlib.pyplot as plt

import googlemaps


def latitude_to_geocentric(phi):
    a = 6378.137
    b = 6356.752
    return np.arctan(b**2/a**2*np.tan(phi))

def latitude_to_geodetic(phi):
    a = 6378.137
    b = 6356.752
    return np.arctan(a**2/b**2*np.tan(phi))

def xyz_from_lle(point):
    lat = point[0]
    lon = point[1]
    elev = point[2]
    
    a = 6378137.0
    b = 6356752.0
    
    # convert to radians
    phi = lat*np.pi/180.0
    theta = lon*np.pi/180.0
  
    # compute radius of earth at each point
    r = (a**2 * np.cos(phi))**2 + (b**2 * np.sin(phi))**2
    r = r / (a**2 * np.cos(phi)**2 + b**2 * np.sin(phi)**2)
    r = np.sqrt(r)
    
    h = r + elev
    alpha = latitude_to_geocentric(phi)
    beta = theta
    x = h * np.cos(alpha) * np.cos(beta)
    y = h * np.cos(alpha) * np.sin(beta)
    z = h * np.sin(alpha)
    v = np.array((x, y, z))
    return v

def lle_from_xyz(point):
    a = 6378137.0
    b = 6356752.0
    
    x = point[0]
    y = point[1]
    z = point[2]
    
    # get corresponding point on earth's surface
    t = np.sqrt((a*b)**2/(b**2*(x**2+y**2)+a**2*z**2))
    point_s = t * point
    x_s = point_s[0]
    y_s = point_s[1]
    z_s = point_s[2]
    
    elev = np.linalg.norm(point-point_s)
    r = np.linalg.norm(point_s)
    
    alpha = np.arcsin(z_s / r)
    phi = latitude_to_geodetic(alpha)
    lat = phi*180.0/np.pi
    
    lon = np.arctan2(y, x)*180/np.pi
    return (lat, lon, elev)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def grid_lat_lon(lat, lon, grid_size=200, grid_step=.001):
    '''
    input lat long
    output grid of lat,long tuples
    '''
    grid = np.ndarray((grid_size + 1, grid_size + 1, 2))
    
    # fill out grid
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            grid[i,j,0] = lat + (i - grid_size / 2) * grid_step
            grid[i,j,1] = lon + (j - grid_size / 2) * grid_step
            
    return grid
    
def get_grid_elevations(in_grid, api_key):
    '''
    input grid of lat,lon tuples
    output grid of LLE tuples
    '''
    in_shape = in_grid.shape
    lats = in_grid.T[0].flatten()
    longs = in_grid.T[1].flatten()
    locations = zip(lats, longs)
    gmaps = googlemaps.Client(key=api_key)
    
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
            lat = in_grid[i,j,0]
            lon = in_grid[i,j,1]
            elevation = elevations[i + j * in_shape[1]]
            
            out_grid[i,j,0] = lat
            out_grid[i,j,1] = lon
            out_grid[i,j,2] = elevation  
    return out_grid
    
def dip_calc(pt1, pt2):
    '''
    input: two LLE tuples
    output: distance, dip angle, azimuth
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
    
    v1 = xyz_from_lle((lat1, lon1, elev1))
    v2 = xyz_from_lle((lat2, lon2, elev2))
    
    x1 = v1[0]
    y1 = v1[1]
    z1 = v1[2]
    x2 = v2[0]
    y2 = v2[1]
    z2 = v2[2]
    
    delta = np.subtract(v1,v2)
    distance = np.linalg.norm(delta)
    
    normal = np.array((2*x1/a**2, 2*y1/a**2, 2*z1/b**2))
    beta = np.arccos(np.dot(delta, normal)/np.linalg.norm(delta)/np.linalg.norm(normal))
    dip_angle = beta - np.pi/2
    dip_angle_deg = dip_angle*180.0/np.pi

    # might wanna double check this formula (haversine?)
    azimuth = np.arctan2(np.sin(theta2-theta1)*np.cos(phi2), np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(theta2-theta1))
    azimuth_deg = azimuth*180.0/np.pi
    
    
    return (azimuth_deg, dip_angle_deg)
    
def calculate_horizon_points(grid, sampling_method="grid", sampling_param=400):
    '''
    input grid of lat,lon,elevation tuples
    output list of azimuth, distance, dip angle
    use grid points in first pass and then try randomly sampling triangle method
    '''
    
    grid_shape = grid.shape
    grid_center_i = (grid_shape[0] - 1) / 2
    grid_center_j = (grid_shape[1] - 1) / 2
    site_lat = grid[grid_center_i, grid_center_j, 0]
    site_lon = grid[grid_center_i, grid_center_j, 1]
    site_elev = grid[grid_center_i, grid_center_j, 2]
    site = (site_lat, site_lon, site_elev)
    
    horizon_points = []
    start = time.time()
    if sampling_method == "grid":
        samples = sample_using_grid(grid)
    elif sampling_method == "triangles":
        samples = sample_using_triangles(grid, sampling_param)
    elif sampling_method == "interpolator":
        samples = sample_using_interpolator(grid, sampling_param)
    post_sampling = time.time()
    print("Sampling took " + str(post_sampling-start) + " sec")
    
    dip_calc_lambda = lambda pt: dip_calc(site, pt)
    horizon_points = np.array(list(map(dip_calc_lambda, samples)))

    post_calcs = time.time()
    print("Dip calcs on samples took " + str(post_calcs-post_sampling) + " sec")
    return horizon_points

def sample_using_grid(grid):
    # use every grid point as a sample
    # returns a list of LLE tuples
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
    # uniformly sample all triangles between neighboring grid points
    # returns a list of LLE tuples
    
    all_samples = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            center = (grid[i,j,0], grid[i,j,1], grid[i,j,2])
            if i != 0 and j != 0:
                left = (grid[i,j-1,0], grid[i,j-1,1], grid[i,j-1,2])
                top = (grid[i-1,j,0], grid[i-1,j,1], grid[i-1,j,2])
                all_samples += uniformly_sample_triangle(center, top, left , samples_per_triangle)
            
            if i != 0 and j != grid.shape[1] - 1:
                right = (grid[i,j+1,0], grid[i,j+1,1], grid[i,j+1,2])
                top = (grid[i-1,j,0], grid[i-1,j,1], grid[i-1,j,2])
                all_samples += uniformly_sample_triangle(center, top, right , samples_per_triangle)
            
            if i != grid.shape[0] - 1 and j != 0:
                left = (grid[i,j-1,0], grid[i,j-1,1], grid[i,j-1,2])
                bottom = (grid[i+1,j,0], grid[i+1,j,1], grid[i+1,j,2])
                all_samples += uniformly_sample_triangle(center, bottom, left , samples_per_triangle)
            
            if i != grid.shape[0] - 1 and j != grid.shape[1] - 1:
                right = (grid[i,j+1,0], grid[i,j+1,1], grid[i,j+1,2])
                bottom = (grid[i+1,j,0], grid[i+1,j,1], grid[i+1,j,2])
                all_samples += uniformly_sample_triangle(center, bottom, right , samples_per_triangle)
    return all_samples

def sample_using_interpolator(grid, num_samples):
    x = grid.T[0][0]
    y = grid.T[1].T[0]
    
    x_base = x[0]
    x_range = x[-1] - x[0]
    y_base = y[0]
    y_range = y[-1] - y[0]
    
    grid_shape = grid.shape
    grid_center_i = (grid_shape[0] - 1) / 2
    grid_center_j = (grid_shape[1] - 1) / 2
    site_lat = grid[grid_center_i, grid_center_j, 0]
    site_lon = grid[grid_center_i, grid_center_j, 1]

    elevs = grid.T[2].T
    interpolator = RegularGridInterpolator((x,y), elevs)
    

    r = np.linspace(0, x_range/2, num_samples[0])
    theta = np.linspace(0, 2 * np.pi, num_samples[1])
    polar_pts = np.array(list(itertools.product(r, theta)))
    
    pts = np.array([pol2cart(e[0], e[1]) for e in polar_pts])
    pts += np.array((site_lat, site_lon))
    
    interpolated_elevs = interpolator(pts).reshape(num_samples[0]*num_samples[1], 1)
    samples = np.concatenate((pts, interpolated_elevs), axis=1)
    return samples
    

def uniformly_sample_triangle(p1, p2, p3, num_samples):
    # returns a list of LLE tuples
    c1 = xyz_from_lle(p1)
    c2 = xyz_from_lle(p2)
    c3 = xyz_from_lle(p3)
    
    points = []
    for i in range(num_samples):
        r1 = np.random.rand()
        r2 = np.random.rand()
        sqrt_r1 = np.sqrt(r1)
        
        # use uniform sampling from http://www.sherrytowers.com/randomly_sample_points_in_triangle.pdf
        random_pt = (1-sqrt_r1)*c1 + sqrt_r1*(1-r2)*c2 + sqrt_r1*r2*c3
        points.append(lle_from_xyz(random_pt))
    return points

def round_to_nearest(x, base):
    return base * round(float(x) / base)

def filter_points(horizon_points, bucket_size=1):
    wedges = {}
    for pair in horizon_points:
        azimuth = pair[0]
        dip = pair[1]
        azimuth_wedge = round_to_nearest(azimuth, bucket_size)
        
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
        
    azimuths = []
    dips = []
    for pair in horizon_profile:
        azimuth = pair[0]
        azimuths.append(azimuth)
        dips.append(pair[1])
    plt.figure(figsize=(10,6))
    if pvsyst_scale:
        plt.ylim(0, 90)
    plt.plot(azimuths, dips, "-")
    plt.show

    
def polar_plot(horizon_profile):
        
    azimuths = []
    dips = []
    for pair in horizon_profile:
        azimuth = pair[0]
        azimuths.append(np.radians(azimuth))
        dips.append(pair[1] + 5)
    plt.figure(figsize=(10,6))
    sp = plt.subplot(1, 1, 1, projection='polar')
    sp.set_theta_zero_location('N')
    sp.set_theta_direction(-1)
    plt.plot(azimuths, dips, "o")
    plt.show

def horizon_table(horizon_points):
    for pair in horizon_points:
        azimuth = pair[0]
        print(str(azimuth) + ": " + str(pair[1]))
        
def invert_for_pvsyst(horizon_points, hemisphere="north"):
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
	grid = grid_lat_lon(lat, lon, grid_size=400, grid_step=.002)
	elev_grid = get_grid_elevations(grid, GMAPS_API_KEY)
	horizon_points = calculate_horizon_points(elev_grid, sampling_method="interpolator", sampling_param=(1000,1000))
	filtered_points = filter_points(horizon_points, bucket_size=1)
	return filtered_points
