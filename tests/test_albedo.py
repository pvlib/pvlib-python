import numpy as np
import pandas as pd
import pytest
from pvlib import albedo

from .conftest import assert_series_equal
from numpy.testing import assert_allclose


def test_inland_water_dvoracek_default():
    result = albedo.inland_water_dvoracek(solar_elevation=90,
                                          color_coeff=0.13,
                                          wave_roughness_coeff=0.29)
    assert_allclose(result, 0.072, 0.001)


def test_inland_water_dvoracek_negative_elevation():
    result = albedo.inland_water_dvoracek(solar_elevation=-60,
                                          color_coeff=0.13,
                                          wave_roughness_coeff=0.29)
    assert_allclose(result, 0.13, 0.01)


def test_inland_water_dvoracek_string_surface_condition():
    result = albedo.inland_water_dvoracek(solar_elevation=90,
                                          surface_condition='clear_water_no_waves')   # noqa: E501
    assert_allclose(result, 0.072, 0.001)


def test_inland_water_dvoracek_ndarray():
    solar_elevs = np.array([-50, 0, 20, 60, 90])
    color_coeffs = np.array([0.1, 0.1, 0.2, 0.3, 0.4])
    roughness_coeffs = np.array([0.3, 0.3, 0.8, 1.5, 2])
    result = albedo.inland_water_dvoracek(solar_elevation=solar_elevs,
                                          color_coeff=color_coeffs,
                                          wave_roughness_coeff=roughness_coeffs)   # noqa: E501
    expected = np.array([0.1, 0.1, 0.12875, 0.06278, 0.064])
    assert_allclose(expected, result, atol=1e-5)


def test_inland_water_dvoracek_series():
    times = pd.date_range(start="2015-01-01 00:00", end="2015-01-02 00:00",
                          freq="6h")
    solar_elevs = pd.Series([-50, 0, 20, 60, 90], index=times)
    color_coeffs = pd.Series([0.1, 0.1, 0.2, 0.3, 0.4], index=times)
    roughness_coeffs = pd.Series([0.1, 0.3, 0.8, 1.5, 2], index=times)
    result = albedo.inland_water_dvoracek(solar_elevation=solar_elevs,
                                          color_coeff=color_coeffs,
                                          wave_roughness_coeff=roughness_coeffs)   # noqa: E501
    expected = pd.Series([0.1, 0.1, 0.12875, 0.06278, 0.064], index=times)
    assert_series_equal(expected, result, atol=1e-5)


def test_inland_water_dvoracek_series_mix_with_array():
    times = pd.date_range(start="2015-01-01 00:00", end="2015-01-01 06:00",
                          freq="6h")
    solar_elevs = pd.Series([45, 60], index=times)
    color_coeffs = 0.13
    roughness_coeffs = 0.29
    result = albedo.inland_water_dvoracek(solar_elevation=solar_elevs,
                                          color_coeff=color_coeffs,
                                          wave_roughness_coeff=roughness_coeffs)   # noqa: E501
    expected = pd.Series([0.08555, 0.07787], index=times)
    assert_series_equal(expected, result, atol=1e-5)


def test_inland_water_dvoracek_invalid():
    with pytest.raises(ValueError, match='Either a `surface_condition` has to '
                       'be chosen or a combination of `color_coeff` and'
                       ' `wave_roughness_coeff`.'):  # no surface info given
        albedo.inland_water_dvoracek(solar_elevation=45)
    with pytest.raises(KeyError, match='not_a_surface_type'):  # invalid type
        albedo.inland_water_dvoracek(solar_elevation=45,
                                     surface_condition='not_a_surface_type')
    with pytest.raises(ValueError, match='Either a `surface_condition` has to '
                       'be chosen or a combination of `color_coeff` and'
                       ' `wave_roughness_coeff`.'):  # only one coeff given
        albedo.inland_water_dvoracek(solar_elevation=45,
                                     color_coeff=0.13)
    with pytest.raises(ValueError, match='Either a `surface_condition` has to '
                       'be chosen or a combination of `color_coeff` and'
                       ' `wave_roughness_coeff`.'):  # only one coeff given
        albedo.inland_water_dvoracek(solar_elevation=45,
                                     wave_roughness_coeff=0.29)
