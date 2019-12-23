import os
import numpy as np
import pandas as pd
import pytest
from pvlib.ivtools.utility import numdiff, rectify_iv_curve


BASEDIR = os.path.dirname(__file__)
TESTDIR = os.path.dirname(BASEDIR)
PROJDIR = os.path.dirname(TESTDIR)
DATADIR = os.path.join(PROJDIR, 'data')
TESTDATA = os.path.join(DATADIR, 'ivtools_numdiff.dat')


@pytest.fixture
def ivcurve():
    voltage = np.array([0., 1., 5., 10., 25., 25.00001, 30., 28., 45., 47.,
                        49., 51., np.nan])
    current = np.array([7., 6., 6., 5., 4., 3., 2.7, 2.5, np.nan, 0.5, -1., 0.,
                        np.nan])
    return voltage, current


def test_numdiff():
    iv = pd.read_csv(
        TESTDATA, names=['I', 'V', 'dIdV', 'd2IdV2'], dtype=float)
    df, d2f = numdiff(iv.V, iv.I)
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
