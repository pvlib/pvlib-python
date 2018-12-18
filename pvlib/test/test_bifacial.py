import numpy as np
from datetime import datetime
from pvlib.bifacial import pvfactors_timeseries
from conftest import requires_pvfactors


@requires_pvfactors
def test_pvfactors_timeseries():
    """ Test that pvfactors is functional, using the TLDR section inputs of the
    package github repo README.md file"""

    # Create some inputs
    timestamps = np.array([datetime(2017, 8, 31, 11),
                           datetime(2017, 8, 31, 12)])
    solar_zenith = np.array([20., 10.])
    solar_azimuth = np.array([110., 140.])
    surface_tilt = np.array([10., 0.])
    surface_azimuth = np.array([90., 90.])
    dni = np.array([1000., 300.])
    dhi = np.array([50., 500.])
    gcr = 0.4
    pvrow_height = 1.75
    pvrow_width = 2.44
    albedo = 0.2
    n_pvrows = 3
    index_observed_pvrow = 1
    rho_front_pvrow = 0.03
    rho_back_pvrow = 0.05
    horizon_band_angle = 15.

    # Expected values
    expected_ipoa_front = [1034.96216923, 795.4423259]
    expected_ipoa_back = [92.11871485, 70.39404124]
    tolerance = 1e-6

    # Test serial calculations
    ipoa_front, ipoa_back, _ = pvfactors_timeseries(
        solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
        timestamps, dni, dhi, gcr, pvrow_height, pvrow_width, albedo,
        n_pvrows=n_pvrows, index_observed_pvrow=index_observed_pvrow,
        rho_front_pvrow=rho_front_pvrow, rho_back_pvrow=rho_back_pvrow,
        horizon_band_angle=horizon_band_angle,
        run_parallel_calculations=False, n_workers_for_parallel_calcs=None)

    np.testing.assert_allclose(ipoa_front, expected_ipoa_front,
                               atol=0, rtol=tolerance)
    np.testing.assert_allclose(ipoa_back, expected_ipoa_back,
                               atol=0, rtol=tolerance)

    # Run calculations in parallel
    ipoa_front, ipoa_back, _ = pvfactors_timeseries(
        solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
        timestamps, dni, dhi, gcr, pvrow_height, pvrow_width, albedo,
        n_pvrows=n_pvrows, index_observed_pvrow=index_observed_pvrow,
        rho_front_pvrow=rho_front_pvrow, rho_back_pvrow=rho_back_pvrow,
        horizon_band_angle=horizon_band_angle,
        run_parallel_calculations=True, n_workers_for_parallel_calcs=None)

    np.testing.assert_allclose(ipoa_front, expected_ipoa_front,
                               atol=0, rtol=tolerance)
    np.testing.assert_allclose(ipoa_back, expected_ipoa_back,
                               atol=0, rtol=tolerance)
