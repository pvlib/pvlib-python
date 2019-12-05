import os
import pandas as pd
from pvlib.ivtools.PVsyst_parameter_estimation import numdiff
import numpy as np

TESTDIR = os.path.dirname(__file__)
IVTOOLS = os.path.dirname(TESTDIR)
PROJDIR = os.path.dirname(IVTOOLS)
DATADIR = os.path.join(PROJDIR, 'data')
TESTDATA = os.path.join(DATADIR, 'ivtools_numdiff.dat')

def test_numdiff():
    iv = pd.read_csv(
        TESTDATA, names=['I', 'V', 'dIdV', 'd2IdV2'], dtype=float)
    df, d2f = numdiff(iv.V, iv.I)
    assert np.allclose(iv.dIdV, df, equal_nan=True)
    assert np.allclose(iv.d2IdV2, d2f, equal_nan=True)
