import numpy as np
import pandas as pd

from pandas.testing import assert_series_equal
from numpy.testing import assert_allclose
import pytest
from datetime import timezone, timedelta

import pvlib
from pvlib import shading


@pytest.fixture
def test_system():
    syst = {'height': 1.0,
            'pitch': 2.,
            'surface_tilt': 30.,
            'surface_azimuth': 180.,
            'rotation': -30.}  # rotation of right edge relative to horizontal
    syst['gcr'] = 1.0 / syst['pitch']
    return syst


def test__ground_angle(test_system):
    ts = test_system
    x = np.array([0., 0.5, 1.0])
    angles = shading.ground_angle(
        ts['surface_tilt'], ts['gcr'], x)
    expected_angles = np.array([0., 5.866738789543952, 9.896090638982903])
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
        "2019-01-01T08", "2019-01-01T17", freq="1H", tz=tzinfo
    )
    timedata["Apparent Zenith"] = 90.0 - timedata["Apparent Elevation"]
    return (axis_tilt_angle, axis_azimuth_angle, timedata)


@pytest.fixture
def singleaxis_psz_implementation_port_data():
    # data generated with the PSZ angle implementation in tracking.singleaxis
    # See GitHub issue #1734 & PR #1904
    axis_tilt_angle = 12.224
    axis_azimuth_angle = 187.2

    singleaxis_result = pd.DataFrame(
        columns=[
            "Apparent Zenith",
            "Solar Azimuth",
            "tracker_theta",
            "surface_azimuth",
            "surface_tilt",
        ],
        data=[
            [88.86131915, 116.14911543, -84.67346, 98.330924, 84.794565],
            [85.67558254, 119.46577753, -80.544188, 99.219659, 80.760477],
            [82.4784391, 122.90558458, -76.226064, 100.171259, 76.5443],
            [79.37555806, 126.48822166, -71.79054, 101.184411, 72.217365],
            [76.40491865, 130.23239671, -67.237442, 102.276947, 67.781439],
            [73.59273783, 134.15525777, -62.55178, 103.476096, 63.224495],
            [70.96318968, 138.2715258, -57.713941, 104.819827, 58.53107],
            [68.54068323, 142.59233032, -52.702658, 106.361922, 53.685798],
            [66.35031258, 147.12377575, -47.496592, 108.18131, 48.676053],
            [64.41759166, 151.8653323, -42.07579, 110.39903, 43.495367],
            [62.76775062, 156.80824414, -36.423404, 113.210504, 38.148938],
            [61.42469841, 161.9342438, -30.527799, 116.950922, 32.663696],
            [60.40974474, 167.21493901, -24.385012, 122.236817, 27.108957],
            [59.74022062, 172.61222482, -18.001341, 130.288224, 21.645102],
            [59.42818646, 178.07994717, -11.395651, 143.610698, 16.652493],
            [59.47944177, 183.56677914, -4.600779, 166.390187, 13.048796],
            [59.89302187, 189.01995634, 2.336615, 198.108, 12.441979],
            [60.66128258, 194.38926277, 9.358232, 225.094855, 15.351466],
            [61.77055542, 199.63057627, 16.398369, 241.465486, 20.352345],
            [63.20224386, 204.70842576, 23.389598, 251.116742, 26.231294],
            [64.93416116, 209.59729217, 30.268795, 257.259578, 32.425598],
            [66.94189859, 214.28170196, 36.982274, 261.49605, 38.674352],
            [69.20004673, 218.75538494, 43.489104, 264.617474, 44.841832],
            [71.68314725, 223.01986867, 49.762279, 267.042188, 50.852813],
            [74.36628597, 227.08285659, 55.787916, 269.007999, 56.666604],
            [77.22520074, 230.95665462, 61.562937, 270.658956, 62.264111],
            [80.23550305, 234.65680797, 67.091395, 272.086933, 67.639267],
            [83.3693091, 238.20102038, 72.378024, 273.352342, 72.790188],
            [86.57992299, 241.60837123, 77.408775, 274.492262, 77.698775],
            [89.70940444, 244.89880789, 82.045935, 275.505443, 82.227402],
        ],
    )
    singleaxis_result.index = pd.date_range(
        "2024-01-25 08:40",
        "2024-01-25 18:20",
        freq="20min",
        tz=timezone(timedelta(hours=1)),
    )
    return (axis_tilt_angle, axis_azimuth_angle, singleaxis_result)


def test_projected_solar_zenith_angle_numeric(
    true_tracking_angle_and_inputs_NREL, singleaxis_psz_implementation_port_data
):
    psz_func = shading.projected_solar_zenith_angle
    axis_tilt, axis_azimuth, timedata = true_tracking_angle_and_inputs_NREL
    # test against data provided by NREL
    psz = psz_func(
        axis_tilt,
        axis_azimuth,
        timedata["Apparent Zenith"],
        timedata["Solar Azimuth"],
    )
    assert_allclose(psz, timedata["True-Tracking"], atol=1e-3)
    # test by changing axis azimuth and tilt
    psz = psz_func(
        -axis_tilt,
        axis_azimuth - 180,
        timedata["Apparent Zenith"],
        timedata["Solar Azimuth"],
    )
    assert_allclose(psz, -timedata["True-Tracking"], atol=1e-3)

    # test implementation port from tracking.singleaxis
    axis_tilt, axis_azimuth, singleaxis = singleaxis_psz_implementation_port_data
    psz = pvlib.tracking.singleaxis(
        singleaxis["Apparent Zenith"],
        singleaxis["Solar Azimuth"],
        axis_tilt,
        axis_azimuth,
        backtrack=False,
    )
    assert_allclose(psz["tracker_theta"], singleaxis["tracker_theta"], atol=1e-6)


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
        cast_func(axis_tilt),
        cast_func(axis_azimuth),
        cast_func(sun_apparent_zenith),
        cast_func(sun_azimuth),
    )
    psz = psz_func(axis_tilt, axis_azimuth, sun_apparent_zenith, axis_azimuth)
    assert isinstance(psz, cast_type)
