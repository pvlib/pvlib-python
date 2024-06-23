"""Tests spatial submodule."""

from pvlib import spatial
from numpy.testing import assert_allclose
import shapely

import pytest


@pytest.mark.parametrize(
    "azimuth, zenith, expected",
    [
        # vertical vector
        (0, 0, [0, 0, 1]),
        (90, 0, [0, 0, 1]),
        (180, 0, [0, 0, 1]),
        (270, 0, [0, 0, 1]),
        (0, 90, [0, 1, 0]),
        (90, 90, [1, 0, 0]),
        (180, 90, [0, -1, 0]),
        (270, 90, [-1, 0, 0]),
        (0, 45, [0, 0.70710678, 0.70710678]),
        (90, 45, [0.70710678, 0, 0.70710678]),
        (180, 45, [0, -0.70710678, 0.70710678]),
        (270, 45, [-0.70710678, 0, 0.70710678]),
    ],
)
def test__solar_vector(azimuth, zenith, expected):
    vec = spatial._solar_vector(zenith=zenith, azimuth=azimuth)
    assert_allclose(vec, expected, atol=1e-15)


@pytest.mark.parametrize(
    "azimuth, tilt, expected",
    [
        # horizontal surfaces
        (0, 0, [0, 0, 1]),
        (90, 0, [0, 0, 1]),
        (180, 0, [0, 0, 1]),
        (270, 0, [0, 0, 1]),
        # vertical surfaces
        (0, 90, [0, 1, 0]),
        (90, 90, [1, 0, 0]),
        (180, 90, [0, -1, 0]),
        (270, 90, [-1, 0, 0]),
        # tilted surfaces
        (0, 45, [0, 0.70710678, 0.70710678]),
        (90, 45, [0.70710678, 0, 0.70710678]),
        (180, 45, [0, -0.70710678, 0.70710678]),
        (270, 45, [-0.70710678, 0, 0.70710678]),
    ],
)
def test__plane_normal_vector(azimuth, tilt, expected):
    vec = spatial._plane_normal_vector(tilt=tilt, azimuth=azimuth)
    assert_allclose(vec, expected, atol=1e-8)


def test_FlatSurface__init__():
    # construct with native shapely polygon
    surface_azimuth = 180
    surface_tilt = 0
    polygon = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    surface = spatial.FlatSurface(
        azimuth=surface_azimuth,
        tilt=surface_tilt,
        polygon_boundaries=polygon,
    )
    assert surface.azimuth == surface_azimuth
    assert surface.tilt == surface_tilt
    assert surface.polygon == polygon
    assert isinstance(surface.polygon, shapely.Polygon)
    # construct from coordinates
    surface_azimuth = 180
    surface_tilt = 0
    polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
    surface = spatial.FlatSurface(
        azimuth=surface_azimuth, tilt=surface_tilt, polygon_boundaries=polygon
    )
    assert surface.azimuth == surface_azimuth
    assert surface.tilt == surface_tilt
    assert surface.polygon == shapely.Polygon(polygon)
    assert isinstance(surface.polygon, shapely.Polygon)


def test_FlatSurface_readonly_properties():
    surface_azimuth = 180
    surface_tilt = 0
    polygon = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    surface = spatial.FlatSurface(
        azimuth=surface_azimuth,
        tilt=surface_tilt,
        polygon_boundaries=polygon,
    )
    with pytest.raises(AttributeError):
        surface.azimuth = 0
    with pytest.raises(AttributeError):
        surface.tilt = 0
    with pytest.raises(AttributeError):
        surface.polygon = polygon


def test_FlatSurface_get_3D_shades_from():
    pass


def test_FlatSurface_get_2D_shades_from():
    pass


def test_FlatSurface_combine_2D_shades():
    pass


def test_FlatSurface_plot():
    pass


def test_RectangularSurface__init__():
    # construct with native shapely polygon
    center = [0, 0, 0]
    surface_azimuth = 180
    surface_tilt = 0
    axis_tilt = 0
    width = 1
    length = 1
    surface = spatial.RectangularSurface(
        center=center,
        azimuth=surface_azimuth,
        tilt=surface_tilt,
        axis_tilt=axis_tilt,
        width=width,
        length=length,
    )
    assert surface.reference_point == center
    assert surface.azimuth == surface_azimuth
    assert surface.tilt == surface_tilt
    assert surface.roll == axis_tilt
    # construct from coordinates
    center = [0, 0, 0]
    surface_azimuth = 180
    surface_tilt = 0
    axis_tilt = 0
    width = 1
    length = 1
    surface = spatial.RectangularSurface(
        center=center,
        azimuth=surface_azimuth,
        tilt=surface_tilt,
        axis_tilt=axis_tilt,
        width=width,
        length=length,
    )
    assert surface.reference_point == center
    assert surface.azimuth == surface_azimuth
    assert surface.tilt == surface_tilt
    assert surface.roll == axis_tilt


def test_RectangularSurface__calc_surface_tilt_and_azimuth():
    center = [0, 0, 0]
    width = length = 2
    azimuth = 180
    tilt = 45
    axis_tilt = 45
    surface = spatial.RectangularSurface(
        center=center,
        azimuth=azimuth,
        tilt=tilt,
        axis_tilt=axis_tilt,
        width=width,
        length=length,
    )
    assert surface.tilt == tilt
    assert surface.azimuth == azimuth
