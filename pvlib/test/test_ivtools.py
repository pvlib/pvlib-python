# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:51:15 2019

@author: cwhanse
"""

import numpy as np
from pvlib import pvsystem
from pvlib.ivtools import fit_sde_sandia


def get_test_iv_params():
    return {'IL': 8.0, 'I0': 5e-10, 'Rsh': 1000, 'Rs': 0.2, 'nNsVth': 1.61864}


def test_fit_sde_sandia():
    test_params = get_test_iv_params()
    testcurve = pvsystem.singlediode(photocurrent=test_params['IL'],
                                     saturation_current=test_params['I0'],
                                     resistance_shunt=test_params['Rsh'],
                                     resistance_series=test_params['Rs'],
                                     nNsVth=test_params['nNsVth'],
                                     ivcurve_pnts=300)
    expected = tuple(test_params[k] for k in ['IL', 'I0', 'Rsh', 'Rs',
                     'nNsVth'])
    result = fit_sde_sandia(V=testcurve['v'], I=testcurve['i'],
                            Voc=testcurve['v_oc'], Isc=testcurve['i_sc'],
                            Vmp=testcurve['v_mp'], Imp=testcurve['i_mp'])
    assert np.allclose(result, expected, rtol=5e-5)
