import numpy as np
import pandas as pd
from pvlib import albedo

from .conftest import assert_series_equal
from numpy.testing import assert_allclose


def test_albedo_water_default():
    result = albedo.albedo_water(solar_elevation=90,
                                 color_coeff=0.13,
                                 wave_roughness_coeff=0.29)
    assert_allclose(result, 0.072, 0.001)


def test_albedo_water_string_surface_condition():
    result = albedo.albedo_water(solar_elevation=90,
                                 surface_condition='clear_water_no_waves')
    assert_allclose(result, 0.072, 0.001)


def test_albedo_water_ndarray():
    solar_elevs = np.array([0, 20, 60])
    color_coeffs = np.array([0.1, 0.25, 0.4])
    roughness_coeffs = np.array([0.3, 1.3, 2.3])
    result = albedo.albedo_water(solar_elevation=solar_elevs,
                                 color_coeff=color_coeffs,
                                 wave_roughness_coeff=roughness_coeffs)
    expected = np.array([0.1, 0.134973, 0.064479])
    assert_allclose(expected, result, atol=1e-5)


def test_lindholm_floating_series():
    times = pd.date_range(start="2015-01-01 00:00", end="2015-01-01 12:00",
                          freq="6h")
    solar_elevs = pd.Series([0, 20, 60], index=times)
    color_coeffs = pd.Series([0.1, 0.25, 0.4], index=times)
    roughness_coeffs = pd.Series([0.3, 1.3, 2.3], index=times)
    result = albedo.albedo_water(solar_elevation=solar_elevs,
                                 color_coeff=color_coeffs,
                                 wave_roughness_coeff=roughness_coeffs)
    expected = pd.Series([0.1, 0.134973, 0.064479], index=times)
    assert_series_equal(expected, result, atol=1e-5)
