"""
test ants2d
"""

import numpy as np
import pandas as pd
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

