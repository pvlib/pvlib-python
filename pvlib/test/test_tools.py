import pytest
import numpy as np

from numpy.testing import assert_allclose
from pvlib import tools


@pytest.mark.parametrize('keys, input_dict, expected', [
    (['a', 'b'], {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2}),
    (['a', 'b', 'd'], {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2}),
    (['a'], {}, {}),
    (['a'], {'b': 2}, {})
])
def test_build_kwargs(keys, input_dict, expected):
    kwargs = tools._build_kwargs(keys, input_dict)
    assert kwargs == expected


def test_latitude_conversions():
    geocentric_lats = np.radians(np.array([0, 30.2, 45, 90]))
    geodetic_lats = np.radians(np.array([0, 30.36759, 45.19243, 90]))
    temp = tools.latitude_to_geodetic(geocentric_lats)
    assert_allclose(np.degrees(temp), np.degrees(geodetic_lats))
    temp2 = tools.latitude_to_geocentric(geodetic_lats)
    assert_allclose(np.degrees(temp2), np.degrees(geocentric_lats))


def test_basis_conversions():
    test_lle = np.array([[30, -45, 900],
                         [20, 85, 10],
                         [-50, 45, 5]])
    test_xyz = np.array([[3909619.9, -3909619.9, 3170821.2],
                         [522572.4, 5973029.8, 2167700],
                         [2904700.9, 2904700.9, -4862792.5]])

    actual_xyz = tools.lle_to_xyz(test_lle)
    assert_allclose(actual_xyz, test_xyz)

    actual_lle = tools.xyz_to_lle(test_xyz)
    assert_allclose(actual_lle, test_lle, rtol=1e-2)


def test_polar_to_cart():
    test_rho = np.array([10, 10, 50, 20])
    test_phi = np.radians(np.array([180, -30, 45, 270]))
    expected_x = np.array([-10, 5*3**.5, 50.0/2**.5, 0])
    expected_y = np.array([0, -5, 50.0/2**.5, -20])
    actual_x, actual_y = tools.polar_to_cart(test_rho,
                                             test_phi)
    assert_allclose(actual_x, expected_x, atol=1e-7)
    assert_allclose(actual_y, expected_y, atol=1e-7)
