import pandas as pd
from datetime import datetime
from pvlib.bifacial import pvfactors_timeseries
from conftest import requires_pvfactors, assert_series_equal
import pytest


@requires_pvfactors
def test_pvfactors_timeseries():
    """ Test that pvfactors is functional, using the TLDR section inputs of the
    package github repo README.md file:
    https://github.com/SunPower/pvfactors/blob/master/README.md#tldr---quick-start"""

    # Create some inputs
    timestamps = pd.DatetimeIndex([datetime(2017, 8, 31, 11),
                                   datetime(2017, 8, 31, 12)]
                                  ).set_names('timestamps')
    solar_zenith = [20., 10.]
    solar_azimuth = [110., 140.]
    surface_tilt = [10., 0.]
    surface_azimuth = [90., 90.]
    axis_azimuth = 0.
    dni = [1000., 300.]
    dhi = [50., 500.]
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
    expected_ipoa_front = pd.Series([1034.95474708997, 795.4423259036623],
                                    index=timestamps,
                                    name=('total_inc_front'))
    expected_ipoa_back = pd.Series([92.12563846416197, 78.05831585685098],
                                   index=timestamps,
                                   name=('total_inc_back'))

    # Run calculation
    ipoa_inc_front, ipoa_inc_back, _, _ = pvfactors_timeseries(
        solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
        axis_azimuth,
        timestamps, dni, dhi, gcr, pvrow_height, pvrow_width, albedo,
        n_pvrows=n_pvrows, index_observed_pvrow=index_observed_pvrow,
        rho_front_pvrow=rho_front_pvrow, rho_back_pvrow=rho_back_pvrow,
        horizon_band_angle=horizon_band_angle)

    assert_series_equal(ipoa_inc_front, expected_ipoa_front)
    assert_series_equal(ipoa_inc_back, expected_ipoa_back)


@requires_pvfactors
def test_pvfactors_timeseries_pandas_inputs():
    """ Test that pvfactors is functional, using the TLDR section inputs of the
    package github repo README.md file, but converted to pandas Series:
    https://github.com/SunPower/pvfactors/blob/master/README.md#tldr---quick-start"""

    # Create some inputs
    timestamps = pd.DatetimeIndex([datetime(2017, 8, 31, 11),
                                   datetime(2017, 8, 31, 12)]
                                  ).set_names('timestamps')
    solar_zenith = pd.Series([20., 10.])
    solar_azimuth = pd.Series([110., 140.])
    surface_tilt = pd.Series([10., 0.])
    surface_azimuth = pd.Series([90., 90.])
    axis_azimuth = 0.
    dni = pd.Series([1000., 300.])
    dhi = pd.Series([50., 500.])
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
    expected_ipoa_front = pd.Series([1034.95474708997, 795.4423259036623],
                                    index=timestamps,
                                    name=('total_inc_front'))
    expected_ipoa_back = pd.Series([92.12563846416197, 78.05831585685098],
                                   index=timestamps,
                                   name=('total_inc_back'))

    # Run calculation
    ipoa_inc_front, ipoa_inc_back, _, _ = pvfactors_timeseries(
        solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
        axis_azimuth,
        timestamps, dni, dhi, gcr, pvrow_height, pvrow_width, albedo,
        n_pvrows=n_pvrows, index_observed_pvrow=index_observed_pvrow,
        rho_front_pvrow=rho_front_pvrow, rho_back_pvrow=rho_back_pvrow,
        horizon_band_angle=horizon_band_angle)

    assert_series_equal(ipoa_inc_front, expected_ipoa_front)
    assert_series_equal(ipoa_inc_back, expected_ipoa_back)
