"""
test ants2d
"""

import numpy as np
import pandas as pd
import pvlib
from pvlib.bifacial import ants2d

import pytest


def test__shaded_fraction():
    
    # special angles
    tracker_rotation = np.array([60, 60, 60, 60])
    phi = np.array([60, 60, 60, 60])
    gcr = np.array([1, 0.75, 2/3, 0.5])
    expected = np.array([0.5, 1/3, 0.25, 0])
    fs = ants2d._shaded_fraction(tracker_rotation, phi, gcr)
    np.testing.assert_allclose(fs, expected)
    fs = ants2d._shaded_fraction(-tracker_rotation, -phi, gcr)
    np.testing.assert_allclose(fs, expected)

    # sun too high for shade
    assert 0 == ants2d._shaded_fraction(10, 20, 0.5)
    assert 0 == ants2d._shaded_fraction(-10, -20, 0.5)

    # sun behind the modules (AOI > 90)
    # (debatable whether this should be zero or one)
    assert 0 == ants2d._shaded_fraction(45, -50, 0.5)
    assert 0 == ants2d._shaded_fraction(-45, 50, 0.5)

    # edge cases
    tracker_rotation = np.array([0, 0, 0, 90, 90, 90, -90, -90, -90])
    phi = np.array([0, 90, -90, 0, 90, -90, 0, 90, -90])
    gcr = 0.5
    # (some of these are debatable as well)
    expected = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1])
    fs = ants2d._shaded_fraction(tracker_rotation, phi, gcr)
    np.testing.assert_allclose(fs, expected)


def test__shaded_fraction_x0x1():
    fs = ants2d._shaded_fraction(np.array([60, -60]), np.array([60, -60]),
                                 2/3, x0=[0, 0.5], x1=[0.5, 1])
    np.testing.assert_allclose(fs, np.array([[0.5, 0.0], [0.0, 0.5]]))


@pytest.mark.parametrize('model', ['perez', 'haydavies'])
def test__apply_sky_diffuse_model(model):
    inputs = {'dni': 900, 'dhi': 150, 'solar_zenith': 41, 'solar_azimuth': 67,
              'dni_extra': 1360, 'airmass': 1.324}
    dni_adj, dhi_adj = ants2d._apply_sky_diffuse_model(**inputs, model=model)
    # ensure that the adjusted values+isotropic yield the same poa_global as
    # the target model
    kwargs = inputs.copy()
    kwargs.pop('dni')
    kwargs.pop('dhi')
    if model != 'perez':
        kwargs.pop('airmass')
    kwargs['surface_tilt'] = 20
    kwargs['surface_azimuth'] = 180
    adj = pvlib.irradiance.get_total_irradiance(dni=dni_adj, dhi=dhi_adj,
                                                ghi=1000,  # doesn't matter
                                                model='isotropic', **kwargs)
    func = {'perez': pvlib.irradiance.perez,
            'haydavies': pvlib.irradiance.haydavies}[model]
    diffuse = func(dni=inputs['dni'], dhi=inputs['dhi'], **kwargs,
                   return_components=True)
    aoi_proj = pvlib.irradiance.aoi_projection(kwargs['surface_tilt'],
                                               kwargs['surface_azimuth'],
                                               kwargs['solar_zenith'],
                                               kwargs['solar_azimuth'])
    poa_direct = inputs['dni'] * aoi_proj + diffuse['circumsolar']
    poa_sky_diffuse = diffuse['isotropic']
    poa_ground = 1000 * 0.25 * (1 - pvlib.tools.cosd(20)) / 2
    assert adj['poa_direct'] == pytest.approx(poa_direct, abs=1e-10)
    assert adj['poa_sky_diffuse'] == pytest.approx(poa_sky_diffuse, abs=1e-10)
    assert adj['poa_ground_diffuse'] == pytest.approx(poa_ground, abs=1e-10)
    # sum of components, ignoring horizon brightening per ANTS-2D assumption
    assert adj['poa_global'] == pytest.approx(
        poa_direct + poa_sky_diffuse + poa_ground, abs=1e-10)



def test__apply_sky_diffuse_model_errors():
    with pytest.raises(ValueError, match='Must supply dni_extra'):
        ants2d._apply_sky_diffuse_model(0, 0, 'haydavies', None,
                                        None, None, None)
    with pytest.raises(ValueError, match='Must supply dni_extra and airmass'):
        ants2d._apply_sky_diffuse_model(0, 0, 'perez', None,
                                        None, None, None)
    with pytest.raises(ValueError, match='Invalid model: not_a_model'):
        ants2d._apply_sky_diffuse_model(0, 0, 'not_a_model', None,
                                        None, None, None)


def test__apply_ground_slope_cross_axis_slope():
    # use an outrageous cross_axis_slope, just for testing
    inputs = {
        'height': 2, 'pitch': 4, 'gcr': 0.5, 'tracker_rotation': 50,
        'ghi': 600, 'dni': 900, 'dhi': 150, 'solar_zenith': 60,
        'solar_azimuth': 270,
    }
    outputs = ants2d._apply_ground_slope(axis_tilt=0, axis_azimuth=180,
                                         cross_axis_slope=60, **inputs)
    # cosd(60) gives a factor of 2 difference in various inputs
    expected = {'height': inputs['height'] / 2, 'pitch': inputs['pitch'] * 2,
                'gcr': inputs['gcr'] / 2, 'tracker_rotation': -10,
                # zenith aligns with cross_axis_slope:
                'ghi': inputs['dni'] + inputs['dhi']}
    for (key, exp), actual in zip(expected.items(), outputs):
        assert actual == pytest.approx(exp, abs=1e-10), key

    # now flip it around and check negative cross-axis slope
    outputs = ants2d._apply_ground_slope(axis_tilt=0, axis_azimuth=180,
                                         cross_axis_slope=-60, **inputs)
    expected = {'height': inputs['height'] / 2, 'pitch': inputs['pitch'] * 2,
                'gcr': inputs['gcr'] / 2, 'tracker_rotation': 110,
                'ghi': inputs['dhi']}
    for (key, exp), actual in zip(expected.items(), outputs):
        assert actual == pytest.approx(exp, abs=1e-10), key


def test__apply_ground_slope_axis_tilt():
    # use an outrageous axis_tilt, just for testing
    inputs = {
        'height': 2, 'pitch': 4, 'gcr': 0.5, 'tracker_rotation': 50,
        'ghi': 600, 'dni': 900, 'dhi': 150, 'solar_zenith': 60,
        'solar_azimuth': 180,
    }
    outputs = ants2d._apply_ground_slope(axis_tilt=60, axis_azimuth=180,
                                         cross_axis_slope=0, **inputs)
    expected = {'height': inputs['height'] / 2, 'pitch': inputs['pitch'],
                'gcr': inputs['gcr'],
                'tracker_rotation': inputs['tracker_rotation'],
                # zenith aligns with axis_tilt:
                'ghi': inputs['dni'] + inputs['dhi']}
    for (key, exp), actual in zip(expected.items(), outputs):
        assert actual == pytest.approx(exp, abs=1e-10), key

    # now flip it around and check negative axis tilt
    outputs = ants2d._apply_ground_slope(axis_tilt=-60, axis_azimuth=180,
                                         cross_axis_slope=0, **inputs)

    expected = {'height': inputs['height'] / 2, 'pitch': inputs['pitch'],
                'gcr': inputs['gcr'],
                'tracker_rotation': inputs['tracker_rotation'],
                'ghi': inputs['dhi']}
    for (key, exp), actual in zip(expected.items(), outputs):
        assert actual == pytest.approx(exp, abs=1e-10), key


def test__apply_ground_slope_both():
    inputs = {
        'height': 2, 'pitch': 4, 'gcr': 0.5, 'tracker_rotation': 50,
        'ghi': 600, 'dni': 900, 'dhi': 150, 'solar_zenith': 15,
        # the azimuth that results from the tilt/slope parameters below
        'solar_azimuth': 234.73561031724535,
    }
    outputs = ants2d._apply_ground_slope(axis_tilt=45, axis_azimuth=180,
                                         cross_axis_slope=45, **inputs)
    expected = {'height': inputs['height'] / 2,
                'pitch': inputs['pitch'] * 2**0.5,
                'gcr': inputs['gcr'] / 2**0.5,
                'tracker_rotation': inputs['tracker_rotation'] - 45,
                # zenith aligns with axis_tilt:
                'ghi': inputs['dni'] / 2**0.5 + inputs['dhi']}
    for (key, exp), actual in zip(expected.items(), outputs):
        assert actual == pytest.approx(exp, abs=1e-10), key


def test__apply_ground_slope_zero():
    inputs = [2, 4, 0.5, 45, 600, 900, 150, 60, 210]
    outputs = ants2d._apply_ground_slope(*inputs, axis_tilt=0,
                                         axis_azimuth=180,
                                         cross_axis_slope=0)
    for input, output in zip(inputs, outputs):
        assert pytest.approx(input, abs=1e-10) == output


@pytest.fixture
def ants_params():
    # parameters for get_irradiance
    times = pd.date_range("2019-06-01 11:30", freq="h", periods=2)
    inputs = {
        'tracker_rotation': [45, -45],
        'axis_azimuth': 180,
        'solar_zenith': [60, 60],
        'solar_azimuth': [225, 135],
        'gcr': 0.5, 'height': 2.5, 'pitch': 4.0,
        'ghi': [700, 700],
        'dni': [1000, 1000],
        'dhi': [200, 200],
        'albedo': 0.2,
        'dni_extra': [1360, 1360],
        'airmass': [2, 2],
    }
    for k, v in inputs.items():
        if isinstance(v, list):
            inputs[k] = pd.Series(v, index=times)
    return inputs


def test_get_irradiance_return_type(ants_params):
    # verify pandas in -> pandas out, and shapes of numpy outputs
    out = ants2d.get_irradiance(**ants_params, n_row_segments=1)
    assert isinstance(out, pd.DataFrame)  # DataFrame, since n_row_segments=1
    expected_keys = ['poa_front', 'poa_front_direct', 'poa_front_diffuse',
       'poa_front_sky_diffuse', 'poa_front_ground_diffuse',
       'shaded_fraction_front', 'poa_back', 'poa_back_direct',
       'poa_back_diffuse', 'poa_back_sky_diffuse', 'poa_back_ground_diffuse',
       'shaded_fraction_back']
    assert set(out.columns) == set(expected_keys)
    assert len(out) == 2  # 2 timestamps

    out = ants2d.get_irradiance(**ants_params, n_row_segments=3)
    assert isinstance(out, dict)  # dict, since n_row_segments>1
    assert set(out.keys()) == set(expected_keys)
    for k, v in out.items():
        assert v.shape == (3, 2), k  # 3 row segments, 2 timestamps


def test_get_irradiance_symmetry(ants_params):
    # check symmetries for normal tracker
    out = ants2d.get_irradiance(**ants_params, n_row_segments=1)
    # symmetrical/mirrored inputs should produce equal outputs
    pd.testing.assert_series_equal(out.iloc[0, :], out.iloc[1, :],
                                   check_names=False)


@pytest.mark.parametrize('solar_zenith', [
    60,  # partial ground shading, no module shading
    80,  # full ground shading, partial module shading
])
@pytest.mark.parametrize('tracker_rotation', [+90, -90])
def test_get_irradiance_vertical(ants_params, solar_zenith, tracker_rotation):
    # check symmetries for vertical panels (tilt=90)
    ants_params['solar_zenith'] = pd.Series(solar_zenith,
                                            index=ants_params['ghi'].index)
    ants_params['tracker_rotation'] = pd.Series(tracker_rotation,
                                                index=ants_params['ghi'].index)
    out = ants2d.get_irradiance(**ants_params, n_row_segments=1)
    # inputs are symmetrical morning/afternoon, so morning front should equal
    # afternoon back, and vice versa
    front_keys = ['poa_front', 'poa_front_direct', 'poa_front_diffuse',
       'poa_front_sky_diffuse', 'poa_front_ground_diffuse',
       'shaded_fraction_front']
    for front_key in front_keys:
        back_key = front_key.replace("front", "back")
        assert np.isclose(out.iloc[0][front_key], out.iloc[1][back_key])
        assert np.isclose(out.iloc[1][front_key], out.iloc[0][back_key])

    # now with >1 row segment
    out = ants2d.get_irradiance(**ants_params, n_row_segments=2)
    lower_half = 0
    upper_half = 1
    morning = 0
    afternoon = 1
    for front_key in front_keys:
        back_key = front_key.replace("front", "back")
        assert np.isclose(out[front_key][lower_half, morning],
                          out[back_key][lower_half, afternoon])
        assert np.isclose(out[front_key][upper_half, morning],
                          out[back_key][upper_half, afternoon])
        assert np.isclose(out[back_key][lower_half, morning],
                          out[front_key][lower_half, afternoon])
        assert np.isclose(out[back_key][upper_half, morning],
                          out[front_key][upper_half, afternoon])


def test_get_irradiance_limit(ants_params):
    # check that diffuse components of front-side irradiance are lower
    # than what get_total_irradiance predicts
    surface_tilt = ants_params['tracker_rotation'].abs()
    surface_azimuth = np.where(ants_params['tracker_rotation'] > 0, 270, 90)
    irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt, surface_azimuth,
        ants_params['solar_zenith'], ants_params['solar_azimuth'],
        ants_params['dni'], ants_params['ghi'], ants_params['dhi'],
        albedo=ants_params['albedo'], model='isotropic')

    ants = ants2d.get_irradiance(**ants_params, n_row_segments=1,
                                 model='isotropic')
    # 15 W/m2 happens to be just below the difference (determined empirically)
    diff_sky = irrad['poa_sky_diffuse'] - ants['poa_front_sky_diffuse']
    diff_ground = irrad['poa_ground_diffuse'] - ants['poa_front_ground_diffuse']
    assert all(diff_sky > 15)
    assert all(diff_ground > 15)

    # but as pitch->infinity, front-side irradiance converges to
    # output of get_total_irradiance
    ants_params['pitch'] *= 1000
    ants_params['gcr'] /= 1000
    ants = ants2d.get_irradiance(**ants_params, n_row_segments=1,
                                 model='isotropic')

    colmap = {'poa_front': 'poa_global', 'poa_front_direct': 'poa_direct',
              'poa_front_diffuse': 'poa_diffuse',
              'poa_front_sky_diffuse': 'poa_sky_diffuse',
              'poa_front_ground_diffuse': 'poa_ground_diffuse'}
    ants_front = ants[list(colmap)].rename(columns=colmap)
    pd.testing.assert_frame_equal(ants_front, irrad, atol=0.1)
    

@pytest.fixture
def ants_params_fixed():
    # parameters for get_irradiance, for a fixed-tilt system
    inputs = {
        'tracker_rotation': 30,
        'axis_azimuth': 90,
        'solar_zenith': 60,
        'solar_azimuth': 175,
        'gcr': 0.6, 'height': 1.5, 'pitch': 3.5,
        'ghi': 700,
        'dni': 1000,
        'dhi': 200,
        'albedo': 0.2,
        'dni_extra': 1360,
        'airmass': 2,
    }
    return inputs


def test_get_irradiance_horizontal(ants_params_fixed):
    # check that no issues pop up with tracker_rotation=solar_zenith=0
    ants_params_fixed['tracker_rotation'] = 0
    ants_params_fixed['solar_zenith'] = 0
    ants_params_fixed['solar_azimuth'] = 180
    zero = ants2d.get_irradiance(**ants_params_fixed)

    ants_params_fixed['tracker_rotation'] = 0.0001
    ants_params_fixed['solar_zenith'] = 0.0001
    ants_params_fixed['solar_azimuth'] = 180.0001
    pos_epsilon = ants2d.get_irradiance(**ants_params_fixed)

    ants_params_fixed['tracker_rotation'] = -0.0001
    ants_params_fixed['solar_zenith'] = -0.0001
    ants_params_fixed['solar_azimuth'] = 179.9999
    neg_epsilon = ants2d.get_irradiance(**ants_params_fixed)

    for key in zero:
        np.testing.assert_allclose(zero[key], pos_epsilon[key], atol=0.01)
        np.testing.assert_allclose(zero[key], neg_epsilon[key], atol=0.01)


def test_get_irradiance_direct_shading(ants_params_fixed):
    # check that direct shading increases as sun approaches horizon
    ants_params_fixed.pop('solar_zenith')
    out60 = ants2d.get_irradiance(solar_zenith=60, **ants_params_fixed)
    out80 = ants2d.get_irradiance(solar_zenith=80, **ants_params_fixed)
    assert out80['poa_front_direct'] < out60['poa_front_direct']


def test_get_irradiance_multiple_row_segments(ants_params_fixed):
    # check that granular sims average to the same value as n_row_segments=1
    out4 = ants2d.get_irradiance(**ants_params_fixed, n_row_segments=4)
    out2 = ants2d.get_irradiance(**ants_params_fixed, n_row_segments=2)
    out1 = ants2d.get_irradiance(**ants_params_fixed, n_row_segments=1)

    for k in out4:
        # check two bottom quarters average to the bottom half, and top
        # two quarters average to the top half
        assert np.isclose(np.mean(out4[k][0:2, 0]), out2[k][0, 0])
        assert np.isclose(np.mean(out4[k][2:4, 0]), out2[k][1, 0])

        # check that two halves average to the whole
        assert np.isclose(np.mean(out2[k][:, 0]), out1[k])


def test_get_irradiance_slope(ants_params_fixed):
    # check the slope affects direct & diffuse shading
    flat = ants2d.get_irradiance(cross_axis_slope=0, **ants_params_fixed)
    # negative slope with axis_azimuth=90 means sloping down to the north
    tilt = ants2d.get_irradiance(cross_axis_slope=-10, **ants_params_fixed)
    assert tilt['shaded_fraction_front'] > flat['shaded_fraction_front']
    assert tilt['poa_front_direct'] < flat['poa_front_direct']
    assert tilt['poa_front_sky_diffuse'] < flat['poa_front_sky_diffuse']


def test_get_irradiance_nonuniform_albedo():
    # check that specifying albedo for each ground segment works

    # horizontal array, very close to the ground, with different albedo
    # on left and right sides. check that ground-reflected irradiance at
    # the edges of the module match the corresponding albedos
    inputs = {
        'tracker_rotation': 0,
        'axis_azimuth': 180,
        'solar_zenith': 0,
        'solar_azimuth': 180,
        'gcr': 0.1, 'height': 0.05, 'pitch': 20,
        'ghi': 1000,
        'dni': 1000,
        'dhi': 0,
        'albedo': np.array([[0.5]*10 + [0.1]*10]).T,
        'model': 'isotropic'
    }
    out = ants2d.get_irradiance(n_ground_segments=20,
                                n_row_segments=10000,
                                max_rows=2,
                                **inputs)
    # check far left and right segments, on the edge of the module.
    # need a large n_row_segments so that these segments are very thin
    left, right = out['poa_back_ground_diffuse'][[0, -1], 0]
    # divide by two because ~half the visible ground is fully shaded
    np.testing.assert_allclose(left, 0.1 * 1000 / 2, rtol=0.002)
    np.testing.assert_allclose(right, 0.5 * 1000 / 2, rtol=0.002)


def test_get_irradiance_nonuniform_albedo_limit():
    # nonuniform albedo averages to uniform albedo, when sufficiently far away
    base_inputs = {
        'tracker_rotation': 45,
        'axis_azimuth': 180,
        'solar_zenith': 10,
        'solar_azimuth': 215,
        'gcr': 0.5, 'height': 1000, 'pitch': 4,
        'ghi': 300,
        'dni': 0,  # set dni to zero so that shadows don't confound results
        'dhi': 300,
        'n_ground_segments': 2,
        'max_rows': 10000,
        'model': 'isotropic',
   }
    out_uni = ants2d.get_irradiance(albedo=0.3,
                                    **base_inputs)
    out_non = ants2d.get_irradiance(albedo=np.array([[0.5, 0.1]]).T,
                                    **base_inputs)
    for key in out_non:
        np.testing.assert_allclose(out_non[key], out_uni[key], atol=1e-6)

