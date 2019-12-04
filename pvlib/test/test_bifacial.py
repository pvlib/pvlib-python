import pandas as pd
import numpy as np
from datetime import datetime
from pvlib.bifacial import pvfactors_timeseries, PVFactorsReportBuilder
from conftest import requires_pvfactors
import pytest


@requires_pvfactors
@pytest.mark.parametrize('run_parallel_calculations',
                         [False, True])
def test_pvfactors_timeseries(run_parallel_calculations):
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
    expected_ipoa_back = pd.Series([91.88707460262768, 78.05831585685215],
                                   index=timestamps,
                                   name=('total_inc_back'))

    # Run calculation
    ipoa_front, ipoa_back = pvfactors_timeseries(
        solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
        axis_azimuth,
        timestamps, dni, dhi, gcr, pvrow_height, pvrow_width, albedo,
        n_pvrows=n_pvrows, index_observed_pvrow=index_observed_pvrow,
        rho_front_pvrow=rho_front_pvrow, rho_back_pvrow=rho_back_pvrow,
        horizon_band_angle=horizon_band_angle,
        run_parallel_calculations=run_parallel_calculations,
        n_workers_for_parallel_calcs=-1)

    pd.testing.assert_series_equal(ipoa_front, expected_ipoa_front)
    pd.testing.assert_series_equal(ipoa_back, expected_ipoa_back)


@requires_pvfactors
@pytest.mark.parametrize('run_parallel_calculations',
                         [False, True])
def test_pvfactors_timeseries_pandas_inputs(run_parallel_calculations):
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
    expected_ipoa_back = pd.Series([91.88707460262768, 78.05831585685215],
                                   index=timestamps,
                                   name=('total_inc_back'))

    # Run calculation
    ipoa_front, ipoa_back = pvfactors_timeseries(
        solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
        axis_azimuth,
        timestamps, dni, dhi, gcr, pvrow_height, pvrow_width, albedo,
        n_pvrows=n_pvrows, index_observed_pvrow=index_observed_pvrow,
        rho_front_pvrow=rho_front_pvrow, rho_back_pvrow=rho_back_pvrow,
        horizon_band_angle=horizon_band_angle,
        run_parallel_calculations=run_parallel_calculations,
        n_workers_for_parallel_calcs=-1)

    pd.testing.assert_series_equal(ipoa_front, expected_ipoa_front)
    pd.testing.assert_series_equal(ipoa_back, expected_ipoa_back)


def test_build_1():
    """Test that build correctly instantiates a dictionary, when passed a Nones
    for the report and pvarray arguments.
    """
    report = None
    pvarray = None
    expected = {'total_inc_back': [np.nan], 'total_inc_front': [np.nan]}
    assert expected == PVFactorsReportBuilder.build(report, pvarray)


def test_merge_1():
    """Test that merge correctly returns the first element of the reports
    argument when there is only dictionary in reports.
    """
    test_dict = {'total_inc_back': [1, 2, 3], 'total_inc_front': [4, 5, 6]}
    reports = [test_dict]
    assert test_dict == PVFactorsReportBuilder.merge(reports)


def test_merge_2():
    """Test that merge correctly combines two dictionary reports.
    """
    test_dict_1 = {'total_inc_back': [1, 2], 'total_inc_front': [4, 5]}
    test_dict_2 = {'total_inc_back': [3], 'total_inc_front': [6]}
    expected = {'total_inc_back': [1, 2, 3], 'total_inc_front': [4, 5, 6]}
    reports = [test_dict_1, test_dict_2]
    assert expected == PVFactorsReportBuilder.merge(reports)


def test_merge_3():
    """Test that merge correctly combines three dictionary reports.
    """
    test_dict_1 = {'total_inc_back': [1], 'total_inc_front': [4]}
    test_dict_2 = {'total_inc_back': [2], 'total_inc_front': [5]}
    test_dict_3 = {'total_inc_back': [3], 'total_inc_front': [6]}
    expected = {'total_inc_back': [1, 2, 3], 'total_inc_front': [4, 5, 6]}
    reports = [test_dict_1, test_dict_2, test_dict_3]
    assert expected == PVFactorsReportBuilder.merge(reports)
