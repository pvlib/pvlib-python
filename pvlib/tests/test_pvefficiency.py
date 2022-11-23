import numpy as np
import pandas as pd

from .conftest import assert_series_equal
from numpy.testing import assert_allclose

import pytest

from pvlib import pvefficiency


def test_adr():
    g = [1000, 200, 1000, 200, 1000, 200]
    t = [25, 25, 50, 50, 75, 75]
    p = [1.0, -6.602189, 0.018582, 0.071889, 0.054049]

    e = [1.0, 0.949147, 0.928114, 0.876456, 0.855693, 0.80323]

    result = pvefficiency.adr(g, t, *p)
    assert_allclose(result, e, rtol=1e-5)


def test_fit_pvefficiency_adr():
    g = [1000, 200, 1000, 200, 1000, 200]
    t = [25, 25, 50, 50, 75, 75]
    e = [1.0, 0.949147, 0.928114, 0.876456, 0.855693, 0.80323]

    p = [1.0, -6.602189, 0.018582, 0.071889, 0.054049]

    result = pvefficiency.fit_pvefficiency_adr(g, t, e, dict_output=False)
    assert_allclose(result, p, rtol=1e-5)

    result = pvefficiency.fit_pvefficiency_adr(g, t, e, dict_output=True)
    assert 'k_a' in result


def test_adr_round_trip():
    g = [1000, 200, 1000, 200, 1000, 200]
    t = [25, 25, 50, 50, 75, 75]
    e = [1.0, 0.949147, 0.928114, 0.876456, 0.855693, 0.80323]

    p = pvefficiency.fit_pvefficiency_adr(g, t, e, dict_output=False)
    result = pvefficiency.adr(g, t, *p)
    assert_allclose(result, e, rtol=1e-5)
