import numpy as np
import pandas as pd

from pandas.testing import assert_series_equal
from numpy.testing import assert_allclose, assert_approx_equal
import pytest
from datetime import timezone, timedelta

from pvlib import shading
from pvlib.tools import atand


@pytest.fixture
def test_system():
    syst = {
        "height": 1.0,
        "pitch": 2.0,
        "surface_tilt": 30.0,
        "surface_azimuth": 180.0,
        "rotation": -30.0,
    }  # rotation of right edge relative to horizontal
    syst["gcr"] = 1.0 / syst["pitch"]
    return syst


def test__ground_angle(test_system):
    ts = test_system
    x = np.array([0.0, 0.5, 1.0])
    angles = shading.ground_angle(ts["surface_tilt"], ts["gcr"], x)
    expected_angles = np.array([0.0, 5.866738789543952, 9.896090638982903])
    assert np.allclose(angles, expected_angles)


def test__ground_angle_zero_gcr():
    surface_tilt = 30.0
    x = np.array([0.0, 0.5, 1.0])
    angles = shading.ground_angle(surface_tilt, 0, x)
    expected_angles = np.array([0, 0, 0])
    assert np.allclose(angles, expected_angles)


@pytest.fixture
def surface_tilt():
    idx = pd.date_range("2019-01-01", freq="h", periods=3)
    return pd.Series([0, 20, 90], index=idx)


@pytest.fixture
def masking_angle(surface_tilt):
    # masking angles for the surface_tilt fixture,
    # assuming GCR=0.5 and height=0.25
    return pd.Series([0.0, 11.20223712, 20.55604522], index=surface_tilt.index)


@pytest.fixture
def average_masking_angle(surface_tilt):
    # average masking angles for the surface_tilt fixture, assuming GCR=0.5
    return pd.Series([0.0, 7.20980655, 13.779867461], index=surface_tilt.index)


@pytest.fixture
def shading_loss(surface_tilt):
    # diffuse shading loss values for the average_masking_angle fixture
    return pd.Series([0, 0.00395338, 0.01439098], index=surface_tilt.index)


def test_masking_angle_series(surface_tilt, masking_angle):
    # series inputs and outputs
    masking_angle_actual = shading.masking_angle(surface_tilt, 0.5, 0.25)
    assert_series_equal(masking_angle_actual, masking_angle)


def test_masking_angle_scalar(surface_tilt, masking_angle):
    # scalar inputs and outputs, including zero
    for tilt, angle in zip(surface_tilt, masking_angle):
        masking_angle_actual = shading.masking_angle(tilt, 0.5, 0.25)
        assert np.isclose(masking_angle_actual, angle)


def test_masking_angle_zero_gcr(surface_tilt):
    # scalar inputs and outputs, including zero
    for tilt in surface_tilt:
        masking_angle_actual = shading.masking_angle(tilt, 0, 0.25)
        assert np.isclose(masking_angle_actual, 0)


def test_masking_angle_passias_series(surface_tilt, average_masking_angle):
    # pandas series inputs and outputs
    masking_angle_actual = shading.masking_angle_passias(surface_tilt, 0.5)
    assert_series_equal(masking_angle_actual, average_masking_angle)


def test_masking_angle_passias_scalar(surface_tilt, average_masking_angle):
    # scalar inputs and outputs, including zero
    for tilt, angle in zip(surface_tilt, average_masking_angle):
        masking_angle_actual = shading.masking_angle_passias(tilt, 0.5)
        assert np.isclose(masking_angle_actual, angle)


def test_sky_diffuse_passias_series(average_masking_angle, shading_loss):
    # pandas series inputs and outputs
    actual_loss = shading.sky_diffuse_passias(average_masking_angle)
    assert_series_equal(shading_loss, actual_loss)


def test_sky_diffuse_passias_scalar(average_masking_angle, shading_loss):
    # scalar inputs and outputs
    for angle, loss in zip(average_masking_angle, shading_loss):
        actual_loss = shading.sky_diffuse_passias(angle)
        assert np.isclose(loss, actual_loss)


@pytest.fixture
def true_tracking_angle_and_inputs_NREL():
    # data from NREL 'Slope-Aware Backtracking for Single-Axis Trackers'
    # doi.org/10.2172/1660126 ; Accessed on 2023-11-06.
    tzinfo = timezone(timedelta(hours=-5))
    axis_tilt_angle = 9.666  # deg
    axis_azimuth_angle = 195.0  # deg
    timedata = pd.DataFrame(
        columns=("Apparent Elevation", "Solar Azimuth", "True-Tracking"),
        data=(
            (2.404287, 122.791770, -84.440),
            (11.263058, 133.288729, -72.604),
            (18.733558, 145.285552, -59.861),
            (24.109076, 158.939435, -45.578),
            (26.810735, 173.931802, -28.764),
            (26.482495, 189.371536, -8.475),
            (23.170447, 204.136810, 15.120),
            (17.296785, 217.446538, 39.562),
            (9.461862, 229.102218, 61.587),
            (0.524817, 239.330401, 79.530),
        ),
    )
    timedata.index = pd.date_range(
        "2019-01-01T08", "2019-01-01T17", freq="1h", tz=tzinfo
    )
    timedata["Apparent Zenith"] = 90.0 - timedata["Apparent Elevation"]
    return (axis_tilt_angle, axis_azimuth_angle, timedata)


@pytest.fixture
def projected_solar_zenith_angle_edge_cases():
    premises_and_result_matrix = pd.DataFrame(
        data=[
            # s_zen | s_azm | ax_tilt | ax_azm | psza
            [   0,       0,      0,        0,      0],
            [   0,     180,      0,        0,      0],
            [   0,       0,      0,      180,      0],
            [   0,     180,      0,      180,      0],
            [  45,       0,      0,      180,      0],
            [  45,      90,      0,      180,    -45],
            [  45,     270,      0,      180,     45],
            [  45,      90,     90,      180,    -90],
            [  45,     270,     90,      180,     90],
            [  45,      90,     90,        0,     90],
            [  45,     270,     90,        0,    -90],
            [  45,      45,     90,      180,   -135],
            [  45,     315,     90,      180,    135],
        ],
        columns=["solar_zenith", "solar_azimuth", "axis_tilt", "axis_azimuth",
                 "psza"],
    )
    return premises_and_result_matrix


def test_projected_solar_zenith_angle_numeric(
    true_tracking_angle_and_inputs_NREL,
    projected_solar_zenith_angle_edge_cases
):
    psza_func = shading.projected_solar_zenith_angle
    axis_tilt, axis_azimuth, timedata = true_tracking_angle_and_inputs_NREL
    # test against data provided by NREL
    psz = psza_func(
        timedata["Apparent Zenith"],
        timedata["Solar Azimuth"],
        axis_tilt,
        axis_azimuth,
    )
    assert_allclose(psz, timedata["True-Tracking"], atol=1e-3)
    # test by changing axis azimuth and tilt
    psza = psza_func(
        timedata["Apparent Zenith"],
        timedata["Solar Azimuth"],
        -axis_tilt,
        axis_azimuth - 180,
    )
    assert_allclose(psza, -timedata["True-Tracking"], atol=1e-3)

    # test edge cases
    solar_zenith, solar_azimuth, axis_tilt, axis_azimuth, psza_expected = (
        v for _, v in projected_solar_zenith_angle_edge_cases.items()
    )
    psza = psza_func(
        solar_zenith,
        solar_azimuth,
        axis_tilt,
        axis_azimuth,
    )
    assert_allclose(psza, psza_expected, atol=1e-9)


@pytest.mark.parametrize(
    "cast_type, cast_func",
    [
        (float, lambda x: float(x)),
        (np.ndarray, lambda x: np.array([x])),
        (pd.Series, lambda x: pd.Series(data=[x])),
    ],
)
def test_projected_solar_zenith_angle_datatypes(
    cast_type, cast_func, true_tracking_angle_and_inputs_NREL
):
    psz_func = shading.projected_solar_zenith_angle
    axis_tilt, axis_azimuth, timedata = true_tracking_angle_and_inputs_NREL
    sun_apparent_zenith = timedata["Apparent Zenith"].iloc[0]
    sun_azimuth = timedata["Solar Azimuth"].iloc[0]

    axis_tilt, axis_azimuth, sun_apparent_zenith, sun_azimuth = (
        cast_func(sun_apparent_zenith),
        cast_func(sun_azimuth),
        cast_func(axis_tilt),
        cast_func(axis_azimuth),
    )
    psz = psz_func(sun_apparent_zenith, axis_azimuth, axis_tilt, axis_azimuth)
    assert isinstance(psz, cast_type)


@pytest.fixture
def sf1d_premises_and_expected():
    """Data comprised of solar position, rows parameters and terrain slope
    with respective shade fractions (sf). Returns a 2-tuple with the premises
    to be used directly in shaded_fraction1d(...) in the first element and
    the expected shaded fractions in the second element.
    See [1] in shaded_fraction1d()
    Test data sourced from http://doi.org/10.5281/zenodo.10513987
    """
    test_data = pd.DataFrame(
        columns=["x_L", "z_L", "theta_L", "x_R", "z_R", "theta_R", "z_0", "l",
                 "theta_s", "f_s"],
        data=(
            (1, 0.2,  50, 0,   0,  25,    0, 0.5,  80, 1),
            (1, 0.1,  50, 0,   0,  25, 0.05, 0.5,  80, 0.937191),
            (1,   0,  50, 0, 0.1,  25,    0, 0.5,  80, 0.30605),
            (1,   0,  50, 0, 0.2,  25,    0, 0.5,  80, 0),
            (1, 0.2, -25, 0,   0, -50,    0, 0.5, -80, 0),
            (1, 0.1, -25, 0,   0, -50,    0, 0.5, -80, 0.30605),
            (1,   0, -25, 0, 0.1, -50,  0.1, 0.5, -80, 0.881549),
            (1,   0, -25, 0, 0.2, -50,    0, 0.5, -80, 1),
            (1, 0.2,   5, 0,   0,  25, 0.05, 0.5,  80, 0.832499),
            (1, 0.2, -25, 0,   0,  25, 0.05, 0.5,  80, 0.832499),
            (1, 0.2,   5, 0,   0, -45, 0.05, 0.5,  80, 0.832499),
            (1, 0.2, -25, 0,   0, -45, 0.05, 0.5,  80, 0.832499),
            (1,   0, -25, 0, 0.2,  25, 0.05, 0.5, -80, 0.832499),
            (1,   0, -25, 0, 0.2,  -5, 0.05, 0.5, -80, 0.832499),
            (1,   0,  45, 0, 0.2,  25, 0.05, 0.5, -80, 0.832499),
            (1,   0,  45, 0, 0.2,  -5, 0.05, 0.5, -80, 0.832499),
        ),
    )  # fmt: skip

    test_data["cross_axis_slope"] = atand(
        (test_data["z_R"] - test_data["z_L"])
        / (test_data["x_L"] - test_data["x_R"])
    )
    test_data["pitch"] = test_data["x_L"] - test_data["x_R"]
    # switch Left/Right rows if needed to make the right one the shaded
    where_switch = test_data["theta_s"] >= 0
    test_data["theta_L"], test_data["theta_R"] = np.where(
        where_switch,
        (test_data["theta_L"], test_data["theta_R"]),
        (test_data["theta_R"], test_data["theta_L"]),
    )
    test_data.rename(
        columns={
            "theta_L": "shading_row_rotation",
            "theta_R": "shaded_row_rotation",
            "z_0": "surface_to_axis_offset",
            "l": "collector_width",
            "theta_s": "solar_zenith",  # for the projected solar zenith angle
            "f_s": "shaded_fraction",
        },
        inplace=True,
    )
    test_data.drop(columns=["x_L", "z_L", "x_R", "z_R"], inplace=True)
    # for the projected solar zenith angle
    # this returns the same psz angle as test_data["solar_zenith"]
    test_data["solar_azimuth"], test_data["axis_azimuth"] = 180, 90

    # return 1st: premises dataframe first and 2nd: shaded fraction series
    return (
        test_data.drop(columns=["shaded_fraction"]),
        test_data["shaded_fraction"],
    )


def test_shaded_fraction1d(sf1d_premises_and_expected):
    """Tests shaded_fraction1d"""
    # unwrap sf_premises_and_expected values premises and expected results
    premises, expected_sf_array = sf1d_premises_and_expected
    # test scalar input
    expected_result = expected_sf_array.iloc[0]
    sf = shading.shaded_fraction1d(**premises.iloc[0])
    assert_approx_equal(sf, expected_result)
    assert isinstance(sf, float)

    # test Series inputs
    sf_vec = shading.shaded_fraction1d(**premises)
    assert_allclose(sf_vec, expected_sf_array, atol=1e-6)
    assert isinstance(sf_vec, pd.Series)


def test_shaded_fraction1d_unprovided_shading_row_rotation():
    """Tests shaded_fraction1d without providing shading_row_rotation"""
    test_data = pd.DataFrame(
        columns=[
            "shaded_row_rotation", "surface_to_axis_offset", "collector_width",
            "solar_zenith", "cross_axis_slope", "pitch", "solar_azimuth",
            "axis_azimuth", "expected_sf",
        ],
        data=[
            (30, 0, 5.7735, 60, 0, 5, 90, 180, 0),
            (30, 0, 5.7735, 79, 0, 5, 90, 180, 0.5),
            (30, 0, 5.7735, 90, 0, 5, 90, 180, 1),
        ],
    )  # fmt: skip
    expected_sf = test_data["expected_sf"]
    premises = test_data.drop(columns=["expected_sf"])
    sf = shading.shaded_fraction1d(**premises)
    assert_allclose(sf, expected_sf, atol=1e-2)


@pytest.fixture
def direct_martinez_Table2():
    """
    Original data used in [1] (see pvlib.shading.direct_martinez) to validate
    the model. Some of the data is provided in Table 2.
    Returns tuple with (input: pandas.DataFrame, output: pandas.Series)
    Output is power loss: 1 - (P_shaded / P_unshaded)
    """
    test_data = pd.DataFrame(
        columns=[
            "F_GS-H",
            "F_GS-V",
            "shaded_blocks",
            "poa_direct",
            "poa_diffuse",
            "power_loss_model",
        ],
        data=[
            # F-H, F-V, Nsb, direct, diffuse, power_loss
            # original data sourced from researchers
            [1.00, 0.09, 16, 846.59, 59.42, 0.8844],
            [1.00, 0.18, 16, 841.85, 59.69, 0.8888],
            [1.00, 0.36, 16, 843.38, 59.22, 0.8994],
            [0.04, 0.64,  1, 851.90, 59.40, 0.0783],
            [0.17, 0.45,  3, 862.86, 58.40, 0.2237],
            [0.29, 0.27,  5, 864.14, 58.11, 0.3282],
            [0.50, 0.09,  8, 863.23, 58.31, 0.4634],
            [0.13, 1.00,  2, 870.14, 58.02, 0.2137],
            [0.25, 1.00,  4, 876.57, 57.98, 0.4000],
            [0.38, 1.00,  6, 866.86, 58.89, 0.5577],
            [0.50, 1.00,  8, 874.58, 58.44, 0.6892],
            [0.58, 0.82, 10, 876.80, 58.16, 0.7359],
            [0.75, 0.73, 12, 866.89, 58.73, 0.8113],
            [0.92, 0.64, 15, 861.48, 59.66, 0.8894],
            # custom edge cases
            [0.00, 0.00,  0, 800.00, 50.00, 0.0000],
            [1.00, 1.00, 16, 900.00, 00.00, 1.0000],
            [0.00, 1.00, 16, 000.00, 00.00, np.nan],
            [1.00, 0.00,  0, 000.00, 00.00, np.nan],
            [1.00, 0.00,  0, -50.00, 50.00, np.nan],  # zero poa_global
            [1.00, 0.00,  0, 50.00, -50.00, np.nan],  # zero poa_global
        ]
    )  # fmt: skip
    test_data["total_blocks"] = 16  # total blocks is 16 for all cases
    test_data["shaded_fraction"] = test_data["F_GS-H"] * test_data["F_GS-V"]
    test_data["poa_global"] = (
        test_data["poa_direct"] + test_data["poa_diffuse"]
    )
    test_data = test_data.drop(columns=["F_GS-H", "F_GS-V", "poa_diffuse"])
    return (
        test_data.drop(columns="power_loss_model"),
        test_data["power_loss_model"],
    )


def test_direct_martinez(direct_martinez_Table2):
    """Tests pvlib.shading.direct_martinez"""
    test_data, power_losses_expected = direct_martinez_Table2
    power_losses = shading.direct_martinez(**test_data)
    assert_allclose(power_losses, power_losses_expected, atol=5e-3)
