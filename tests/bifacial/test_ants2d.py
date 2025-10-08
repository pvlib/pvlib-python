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
    expected = np.array([[0.5, 1/3, 0.25, 0]])
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
    expected = np.array([[0, 0, 0, 0, 1, 1, 0, 1, 1]])
    fs = ants2d._shaded_fraction(tracker_rotation, phi, gcr)
    np.testing.assert_allclose(fs, expected)


def test__shaded_fraction_x0x1():
    fs = ants2d._shaded_fraction(np.array([60, -60]), np.array([60, -60]),
                                 2/3, x0=[0, 0.5], x1=[0.5, 1])
    np.testing.assert_allclose(fs, np.array([[0.5, 0.0], [0.0, 0.5]]))


def test__ants2d_singleside():
    pass


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

