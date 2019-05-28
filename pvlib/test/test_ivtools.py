# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:51:15 2019

@author: cwhanse
"""

import numpy as np
import pandas as pd
import pytest
from pvlib import pvsystem
from pvlib import ivtools
from conftest import requires_scipy


def get_test_iv_params():
    return {'IL': 8.0, 'I0': 5e-10, 'Rsh': 1000, 'Rs': 0.2, 'nNsVth': 1.61864}


def get_cec_params_cansol_cs5p_220p():
    return {'V_mp_ref': 46.6, 'I_mp_ref': 4.73, 'V_oc_ref': 58.3,
            'I_sc_ref': 5.05, 'alpha_isc': 0.000495, 'beta_voc': -0.003372,
            'gamma_pmp': -0.43, 'cells_in_series': 96}


@pytest.fixture()
def sam_data():
    return pvsystem.retrieve_sam('cecmod')


@pytest.fixture()
def cec_module_parameters(sam_data):
    modules = sam_data
    module = "Canadian_Solar_CS5P_220P"
    module_parameters = modules[module]
    return module_parameters


@requires_scipy
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
    result = ivtools.fit_sde_sandia(v=testcurve['v'], i=testcurve['i'],
                                    v_oc=testcurve['v_oc'],
                                    i_sc=testcurve['i_sc'],
                                    v_mp=testcurve['v_mp'],
                                    i_mp=testcurve['i_mp'])
    assert np.allclose(result, expected, rtol=5e-5)


@requires_pysam
def test_fit_cec_with_sam(cec_module_parameters):
    sam_parameters = cec_module_parameters
    cec_list_data = get_cec_params_cansol_cs5p_220p()
    # convert from %/C to A/C and V/C
    alpha_sc = cec_list_data['alpha_isc'] * cec_list_data['I_sc_ref']
    beta_oc = cec_list_data['beta_voc'] * cec_list_data['V_oc_ref']

    I_L_ref, I_o_ref, R_sh_ref, R_s, a_ref, Adjust = \
        ivtools.fit_cec_with_sam(celltype='polySi',
            v_mp=cec_list_data['V_mp_ref'], i_mp=cec_list_data['I_mp_ref'],
            v_oc=cec_list_data['V_oc_ref'], i_sc=cec_list_data['I_sc_ref'],
            alpha_sc=alpha_sc, beta_voc=beta_oc,
            gamma_pmp=cec_list_data['gamma_pmp'],
            cells_in_series=cec_list_data['cells_in_series'], temp_ref=25)
    modeled = pd.Series(index=sam_parameters.index, data=cec_list_data)
    modeled['a_ref'] = a_ref
    modeled['I_L_ref'] = I_L_ref
    modeled['I_o_ref'] = I_o_ref
    modeled['R_sh_ref'] = R_sh_ref
    modeled['R_s'] = R_s
    modeled['Adjust'] = Adjust
    modeled['alpha_sc'] = alpha_sc
    modeled['beta_oc'] = beta_oc
    modeled['gamma_r'] = cec_list_data['gamma_pmp']
    modeled['N_s'] = cec_list_data['cells_in_series']
    modeled = modeled.dropna()
    expected = pd.Series(index=modeled.index, data=np.nan)
    for k in modeled.keys():
        expected[k] = sam_parameters[k]
    assert np.allclose(modeled.values, expected.values, rtol=5e-2)
