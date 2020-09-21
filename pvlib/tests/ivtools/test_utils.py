import numpy as np
import pandas as pd
import pytest
from pvlib.ivtools.utils import _numdiff, rectify_iv_curve
from pvlib.ivtools.utils import _schumaker_qspline

from conftest import DATA_DIR


@pytest.fixture
def ivcurve():
    voltage = np.array([0., 1., 5., 10., 25., 25.00001, 30., 28., 45., 47.,
                        49., 51., np.nan])
    current = np.array([7., 6., 6., 5., 4., 3., 2.7, 2.5, np.nan, 0.5, -1., 0.,
                        np.nan])
    return voltage, current


def test__numdiff():
    iv = pd.read_csv(DATA_DIR / 'ivtools_numdiff.csv',
                     names=['I', 'V', 'dIdV', 'd2IdV2'], dtype=float)
    df, d2f = _numdiff(iv.V, iv.I)
    assert np.allclose(iv.dIdV, df, equal_nan=True)
    assert np.allclose(iv.d2IdV2, d2f, equal_nan=True)


def test_rectify_iv_curve(ivcurve):
    voltage, current = ivcurve

    vexp_no_dec = np.array([0., 1.,  5., 10., 25., 25.00001, 28., 30., 47.,
                            51.])
    iexp_no_dec = np.array([7., 6., 6., 5., 4., 3., 2.5, 2.7, 0.5, 0.])
    v, i = rectify_iv_curve(voltage, current)
    np.testing.assert_allclose(v, vexp_no_dec, atol=.0001)
    np.testing.assert_allclose(i, iexp_no_dec, atol=.0001)

    vexp = np.array([0., 1., 5., 10., 25., 28., 30., 47., 51.])
    iexp = np.array([7., 6., 6., 5., 3.5, 2.5, 2.7, 0.5, 0.])
    v, i = rectify_iv_curve(voltage, current, decimals=4)
    np.testing.assert_allclose(v, vexp, atol=.0001)
    np.testing.assert_allclose(i, iexp, atol=.0001)


@pytest.mark.parametrize('x,y,expected', [
    (np.array([0., 1., 2., 3., 4., 1., 2., 3., 4., 5.]),
     np.array([2., 1., 0., 1., 2., 3., 2., 1., 2., 3.]),
     (np.array([[0., -1., 2.], [-0.5, -1., 1.], [-0.75, -0.5, 3.],
                [0.75, -1.5, 0.375], [0.125, -1.25, 2.5625], [1.5, 0., 0.],
                [-0.5, -1., 2.], [-0.25, 1.5, 0.375], [0.75, -1.5, 1.375],
                [0.5, 1., 1.], [1.5, 0., 1.], [0.0278, -0.3333, 2.1667],
                [-0.75, 1.5, 1.625], [-0.25, 1.5, 1.375], [0.1667, 0., 2.],
                [0., 1., 2.]]),
      np.array([0., 1., 1., 1.5, 1.5, 2., 2., 2.5, 2.5, 3., 3., 3., 3.5, 3.5,
                4., 4., 5.]),
      np.array([2., 1., 3., 0.375, 2.5625, 0., 2., 0.375, 1.375, 1., 1.,
                2.1667, 1.625, 1.375, 2., 2., 3.]),
      np.array([0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0.,
                0.]))),
    (np.array([1., 2., 3., 4., 5.]),
     np.array([-2., -1., 0., 1., 2.]),
     (np.array([[0., 1., -2.], [0., 1., -1.], [0., 1., 0.], [0., 1., 1.]]),
      np.array([1., 2., 3., 4., 5.]),
      np.array([-2., -1., 0., 1., 2.]),
      np.array([0., 0., 0., 0., 0.]))),
    (np.array([-.5, -.1, 0., .2, .3]),
     np.array([-5., -1., .2, .5, 2.]),
     (np.array([[2.2727, 9.0909, -5.], [63.0303, 10.9091, -1.],
                [-72.7273, 17.2121, -.297], [-11.8182, 2.6667, .2],
                [6.0606, .303, .3485], [122.7273, 2.7273, .5]]),
      np.array([-.5, -.1, -.05, 0., .1, .2, .3]),
      np.array([-5., -1., -.297, .2, .3485, .5, 2.]),
      np.array([0., 0., 1., 0., 1., 0., 0.])))])
def test__schmumaker_qspline(x, y, expected):
    [t, c, yhat, kflag] = _schumaker_qspline(x, y)
    np.testing.assert_allclose(c, expected[0], atol=0.0001)
    np.testing.assert_allclose(t, expected[1], atol=0.0001)
    np.testing.assert_allclose(yhat, expected[2], atol=0.0001)
    np.testing.assert_allclose(kflag, expected[3], atol=0.0001)
