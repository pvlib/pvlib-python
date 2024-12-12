import itertools

import numpy as np
from numpy import nan
from numpy.testing import assert_allclose
import pandas as pd
from .conftest import assert_series_equal
import pytest

from pvlib import atmosphere

from pvlib._deprecation import pvlibDeprecationWarning


def test_pres2alt():
    out = atmosphere.pres2alt(np.array([10000, 90000, 101325]))
    expected = np.array([15797.638, 988.637, 0.124])
    assert_allclose(out, expected, atol=0.001)


def test_alt2pres():
    out = atmosphere.alt2pres(np.array([-100, 0, 1000, 8000]))
    expected = np.array([102532.073, 101324.999,  89874.750,  35600.496])
    assert_allclose(out, expected, atol=0.001)


@pytest.fixture
def zeniths():
    return np.array([100, 89.9, 80, 0])


@pytest.mark.parametrize("model,expected",
                         [['simple', [nan, 572.958,   5.759,   1.000]],
                          ['kasten1966', [nan, 35.365,  5.580,  0.999]],
                          ['youngirvine1967', [
                                 nan, -2.251358367165932e+05, 5.5365, 1.0000]],
                          ['kastenyoung1989', [nan, 36.467,  5.586,  1.000]],
                          ['gueymard1993', [nan, 36.431,  5.581,  1.000]],
                          ['young1994', [nan, 30.733,  5.541,  1.000]],
                          ['pickering2002', [nan, 37.064,  5.581,  1.000]],
                          ['gueymard2003', [nan, 36.676, 5.590, 1.000]]])
def test_airmass(model, expected, zeniths):
    out = atmosphere.get_relative_airmass(zeniths, model)
    expected = np.array(expected)
    assert_allclose(out, expected, equal_nan=True, atol=0.001)
    # test series in/out. index does not matter
    # hits the isinstance() block in get_relative_airmass
    times = pd.date_range(start='20180101', periods=len(zeniths), freq='1s')
    zeniths = pd.Series(zeniths, index=times)
    expected = pd.Series(expected, index=times)
    out = atmosphere.get_relative_airmass(zeniths, model)
    assert_series_equal(out, expected, check_less_precise=True)


def test_airmass_scalar():
    assert not np.isnan(atmosphere.get_relative_airmass(10))


def test_airmass_invalid():
    with pytest.raises(ValueError):
        atmosphere.get_relative_airmass(0, 'invalid')


def test_get_absolute_airmass():
    # input am
    relative_am = np.array([nan, 40, 2, .999])
    # call without pressure kwarg
    out = atmosphere.get_absolute_airmass(relative_am)
    expected = np.array([nan, 40., 2., 0.999])
    assert_allclose(out, expected, equal_nan=True, atol=0.001)
    # call with pressure kwarg
    out = atmosphere.get_absolute_airmass(relative_am, pressure=90000)
    expected = np.array([nan, 35.529, 1.776, 0.887])
    assert_allclose(out, expected, equal_nan=True, atol=0.001)


def test_gueymard94_pw():
    temp_air = np.array([0, 20, 40])
    relative_humidity = np.array([0, 30, 100])
    temps_humids = np.array(
        list(itertools.product(temp_air, relative_humidity)))
    pws = atmosphere.gueymard94_pw(temps_humids[:, 0], temps_humids[:, 1])

    expected = np.array(
        [  0.1       ,   0.33702061,   1.12340202,   0.1       ,
         1.12040963,   3.73469877,   0.1       ,   3.44859767,  11.49532557])

    assert_allclose(pws, expected, atol=0.01)


def test_tdew_to_rh_to_tdew():

    # dewpoint temp calculated with wmo and aekr coefficients
    dewpoint_original = pd.Series([
        15.0, 20.0, 25.0, 12.0, 8.0
    ])

    temperature_ambient = pd.Series([20.0, 25.0, 30.0, 15.0, 10.0])

    # Calculate relative humidity using pandas series as input
    relative_humidity = atmosphere.rh_from_tdew(
        temp_air=temperature_ambient,
        temp_dew=dewpoint_original
    )

    dewpoint_calculated = atmosphere.tdew_from_rh(
        temp_air=temperature_ambient,
        relative_humidity=relative_humidity
    )

    # test
    pd.testing.assert_series_equal(
        dewpoint_original,
        dewpoint_calculated,
        check_names=False
    )


def test_rh_from_tdew():

    dewpoint = pd.Series([
        15.0, 20.0, 25.0, 12.0, 8.0
    ])

    # relative humidity calculated with wmo and aekr coefficients
    relative_humidity_wmo = pd.Series([
        72.95185312581116, 73.81500029087906, 74.6401272083123,
        82.27063889868842, 87.39018119185337
    ])
    relative_humidity_aekr = pd.Series([
        72.93876680928582, 73.8025121880607, 74.62820502423823,
        82.26135295757305, 87.38323744820416
    ])

    temperature_ambient = pd.Series([20.0, 25.0, 30.0, 15.0, 10.0])

    # Calculate relative humidity using pandas series as input
    rh_series = atmosphere.rh_from_tdew(
        temp_air=temperature_ambient,
        temp_dew=dewpoint
    )

    pd.testing.assert_series_equal(
        rh_series,
        relative_humidity_wmo,
        check_names=False
    )

    # Calulate relative humidity using pandas series as input
    # with AEKR coefficients
    rh_series_aekr = atmosphere.rh_from_tdew(
        temp_air=temperature_ambient,
        temp_dew=dewpoint,
        coeff=(6.1094, 17.625, 243.04)
    )

    pd.testing.assert_series_equal(
        rh_series_aekr,
        relative_humidity_aekr,
        check_names=False
    )

    # Calculate relative humidity using array as input
    rh_array = atmosphere.rh_from_tdew(
        temp_air=temperature_ambient.to_numpy(),
        temp_dew=dewpoint.to_numpy()
    )

    np.testing.assert_allclose(rh_array, relative_humidity_wmo.to_numpy())

    # Calculate relative humidity using float as input
    rh_float = atmosphere.rh_from_tdew(
        temp_air=temperature_ambient.iloc[0],
        temp_dew=dewpoint.iloc[0]
    )

    assert np.isclose(rh_float, relative_humidity_wmo.iloc[0])


# Unit tests
def test_tdew_from_rh():

    dewpoint = pd.Series([
        15.0, 20.0, 25.0, 12.0, 8.0
    ])

    # relative humidity calculated with wmo and aekr coefficients
    relative_humidity_wmo = pd.Series([
        72.95185312581116, 73.81500029087906, 74.6401272083123,
        82.27063889868842, 87.39018119185337
    ])
    relative_humidity_aekr = pd.Series([
        72.93876680928582, 73.8025121880607, 74.62820502423823,
        82.26135295757305, 87.38323744820416
    ])

    temperature_ambient = pd.Series([20.0, 25.0, 30.0, 15.0, 10.0])

    # test as series
    dewpoint_series = atmosphere.tdew_from_rh(
        temp_air=temperature_ambient,
        relative_humidity=relative_humidity_wmo
    )

    pd.testing.assert_series_equal(
        dewpoint_series, dewpoint, check_names=False
    )

    # test as series with AEKR coefficients
    dewpoint_series_aekr = atmosphere.tdew_from_rh(
        temp_air=temperature_ambient,
        relative_humidity=relative_humidity_aekr,
        coeff=(6.1094, 17.625, 243.04)
    )

    pd.testing.assert_series_equal(
        dewpoint_series_aekr, dewpoint,
        check_names=False
    )

    # test as numpy array
    dewpoint_array = atmosphere.tdew_from_rh(
        temp_air=temperature_ambient.to_numpy(),
        relative_humidity=relative_humidity_wmo.to_numpy()
    )

    np.testing.assert_allclose(dewpoint_array, dewpoint.to_numpy())

    # test as float
    dewpoint_float = atmosphere.tdew_from_rh(
        temp_air=temperature_ambient.iloc[0],
        relative_humidity=relative_humidity_wmo.iloc[0]
    )

    assert np.isclose(dewpoint_float, dewpoint.iloc[0])


def test_first_solar_spectral_correction_deprecated():
    with pytest.warns(pvlibDeprecationWarning,
                      match='Use pvlib.spectrum.spectral_factor_firstsolar'):
        atmosphere.first_solar_spectral_correction(1, 1, 'cdte')


def test_kasten96_lt():
    """Test Linke turbidity factor calculated from AOD, Pwat and AM"""
    amp = np.array([1, 3, 5])
    pwat = np.array([0, 2.5, 5])
    aod_bb = np.array([0, 0.1, 1])
    lt_expected = np.array(
        [[[1.3802, 2.4102, 11.6802],
          [1.16303976, 2.37303976, 13.26303976],
          [1.12101907, 2.51101907, 15.02101907]],

         [[2.95546945, 3.98546945, 13.25546945],
          [2.17435443, 3.38435443, 14.27435443],
          [1.99821967, 3.38821967, 15.89821967]],

         [[3.37410769, 4.40410769, 13.67410769],
          [2.44311797, 3.65311797, 14.54311797],
          [2.23134152, 3.62134152, 16.13134152]]]
    )
    lt = atmosphere.kasten96_lt(*np.meshgrid(amp, pwat, aod_bb))
    assert np.allclose(lt, lt_expected, 1e-3)


def test_angstrom_aod():
    """Test Angstrom turbidity model functions."""
    aod550 = 0.15
    aod1240 = 0.05
    alpha = atmosphere.angstrom_alpha(aod550, 550, aod1240, 1240)
    assert np.isclose(alpha, 1.3513924317859232)
    aod700 = atmosphere.angstrom_aod_at_lambda(aod550, 550, alpha)
    assert np.isclose(aod700, 0.10828110997681031)


def test_bird_hulstrom80_aod_bb():
    """Test Bird_Hulstrom broadband AOD."""
    aod380, aod500 = 0.22072480948195175, 0.1614279181106312
    bird_hulstrom = atmosphere.bird_hulstrom80_aod_bb(aod380, aod500)
    assert np.isclose(0.11738229553812768, bird_hulstrom)


@pytest.fixture
def windspeeds_data_powerlaw():
    data = pd.DataFrame(
        index=pd.date_range(start="2015-01-01 00:00", end="2015-01-01 05:00",
                            freq="1h"),
        columns=["wind_ref", "height_ref", "height_desired", "wind_calc"],
        data=[
            (10, -2, 5, np.nan),
            (-10, 2, 5, np.nan),
            (5, 4, 5, 5.067393209486324),
            (7, 6, 10, 7.2178684911195905),
            (10, 8, 20, 10.565167835216586),
            (12, 10, 30, 12.817653329393977)
        ]
    )
    return data


def test_windspeed_powerlaw_ndarray(windspeeds_data_powerlaw):
    # test wind speed estimation by passing in surface_type
    result_surface = atmosphere.windspeed_powerlaw(
        windspeeds_data_powerlaw["wind_ref"].to_numpy(),
        windspeeds_data_powerlaw["height_ref"],
        windspeeds_data_powerlaw["height_desired"],
        surface_type='unstable_air_above_open_water_surface')
    assert_allclose(windspeeds_data_powerlaw["wind_calc"].to_numpy(),
                    result_surface)
    # test wind speed estimation by passing in the exponent corresponding
    # to the surface_type above
    result_exponent = atmosphere.windspeed_powerlaw(
        windspeeds_data_powerlaw["wind_ref"].to_numpy(),
        windspeeds_data_powerlaw["height_ref"],
        windspeeds_data_powerlaw["height_desired"],
        exponent=0.06)
    assert_allclose(windspeeds_data_powerlaw["wind_calc"].to_numpy(),
                    result_exponent)


def test_windspeed_powerlaw_series(windspeeds_data_powerlaw):
    result = atmosphere.windspeed_powerlaw(
        windspeeds_data_powerlaw["wind_ref"],
        windspeeds_data_powerlaw["height_ref"],
        windspeeds_data_powerlaw["height_desired"],
        surface_type='unstable_air_above_open_water_surface')
    assert_series_equal(windspeeds_data_powerlaw["wind_calc"],
                        result, check_names=False)


def test_windspeed_powerlaw_invalid():
    with pytest.raises(ValueError, match="Either a 'surface_type' or an "
                       "'exponent' parameter must be given"):
        # no exponent or surface_type given
        atmosphere.windspeed_powerlaw(wind_speed_reference=10,
                                      height_reference=5,
                                      height_desired=10)
    with pytest.raises(ValueError, match="Either a 'surface_type' or an "
                       "'exponent' parameter must be given"):
        # no exponent or surface_type given
        atmosphere.windspeed_powerlaw(wind_speed_reference=10,
                                      height_reference=5,
                                      height_desired=10,
                                      exponent=1.2,
                                      surface_type="surf")
    with pytest.raises(KeyError, match='not_an_exponent'):
        # invalid surface_type
        atmosphere.windspeed_powerlaw(wind_speed_reference=10,
                                      height_reference=5,
                                      height_desired=10,
                                      surface_type='not_an_exponent')
