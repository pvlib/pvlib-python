'''
written : 220111 Steve Ransome (SRCL)
tests functions from mlfm.py

TESTED
def mlfm_meas_to_norm(dmeas, ref, qty_mlfm_vars):

def mlfm_6(dmeas, c_1, c_2, c_3, c_4, c_5, c_6):

def mlfm_norm_to_stack(dnorm, ref, qty_mlfm_vars):

NOT TESTED YET
def mlfm_fit(dmeas, dnorm, mlfm_sel):

def plot_mlfm_scatter(dmeas, dnorm, mlfm_file_name, qty_mlfm_vars):

def plot_mlfm_stack(dmeas, dnorm, dstack, ref,
                    mlfm_file_name, qty_mlfm_vars,
                    xaxis_labels=12, is_i_sc_self_ref=False,
                    is_v_oc_temp_module_corr=True):
'''

'''
Define tolerance for checking.
Most values are normalised ~1
0.000001 is probably good
'''

import pandas as pd
from pvlib.mlfm import mlfm_meas_to_norm, mlfm_6, mlfm_norm_to_stack
from numpy.testing import assert_allclose  # assert_almost_equal,
import pytest


tolerance = 0.000001

qty_mlfm_vars = 6  # check all 6 mlfm params from iv curves

@pytest.fixture
def reference():
    # get reference module STC values for normalisation
    ref = dict(
        module_id='g78',
        i_sc=5.35,
        i_mp=4.9,
        v_mp=36.8,
        v_oc=44.2,
        alpha_i_sc=0.0005,
        alpha_i_mp=0,  # often not known, not used here
        beta_v_mp=0,  # often not known, not used here
        beta_v_oc=-0.0035,
        gamma_p_mp=-0.0045,  # = alpha_i_mp + beta_v_mp
        delta_ff=0,  # often not known, not used here
    )
    # create p_mp and ff
    ref['p_mp'] = ref['i_mp'] * ref['v_mp']
    ref['ff'] = ref['p_mp'] / (ref['i_sc'] * ref['v_oc'])
    return ref


@pytest.fixture
def measured():
    # get measured data
    data_meas = {
        # 'date_time':      ['2016-03-23 09:00:00-07:00'],
        'module_id':        [78],
        'poa_global':       [591.3868886],
        'wind_speed':       [4.226408028],
        # 'temp_air':       [17.42457581],
        'temp_module':      [27.82861328],
        'v_oc':             [43.52636044],
        'i_sc':             [3.14995479],
        'i_mp':             [2.949264766],
        'v_mp':             [35.76882896],
        'r_sc':             [674.5517322],
        'r_oc':             [1.355690858],
    }

    meas = pd.DataFrame(data_meas)

    # create p_mp and ff in case they don't exist
    meas['poa_global_kwm2'] = meas['poa_global'] / 1000
    meas['p_mp'] = meas['i_mp'] * meas['v_mp']
    meas['ff'] = meas['p_mp'] / (meas['i_sc'] * meas['v_oc'])

    return meas


@pytest.fixture
def normalized():
    data_norm_target = {
        # 'date_time':     ['2016-03-23 09:00:00-07:00'],
        'pr_dc':           [0.989242790817207],
        'pr_dc_temp_corr': [1.00183462464583],
        'i_mp':            [0.93628796685047],
        'v_mp':            [0.821773945683017],
        'i_sc':            [0.995586151149719],
        'v_oc':            [0.98475928597285],
        'v_oc_temp_corr':  [0.994508547151521],
        'r_sc':            [0.981487711004909],
        'r_oc':            [0.903706382978424],
        'i_ff':            [0.953947722780796],
        'v_ff':            [0.909337325885234],
    }

    norm_target = pd.DataFrame(data_norm_target)

    return norm_target


@pytest.fixture
def stacked():
    # get stack data
    data_stack_target = {
        # 'date_time':       ['2016-03-23 09:00:00-07:00'],
        'pr_dc':             [0.989242790817207],
        'i_sc':              [0.0052435168594609],
        'r_sc':              [0.0219920307073518],
        'i_ff':              [0.049708690806242],
        'i_v':               [0.01],
        'v_ff':              [0.102704472076433],
        'r_oc':              [0.114393859291095],
        'v_oc':              [0.0181055001343123],
        'temp_module_corr':  [0.0115818228244058],
    }

    stack_target = pd.DataFrame(data_stack_target)

    return stack_target


@pytest.fixture
def mlfm_6_coeffs():
    # test mlfm coefficients
    c_1 = +1.0760136800094817
    c_2 = -0.004619443769147978
    c_3 = +0.018343135214876096
    c_4 = -0.07613482929987923
    c_5 = -0.0006626101399079871
    c_6 = -0.014752223616684625
    expected = 0.9859917396312191

    return c_1, c_2, c_3, c_4, c_5, c_6, expected


def test_mlfm_meas_to_norm(mlfm_6_coeffs, reference, measured, normalized):
    norm_calc = mlfm_meas_to_norm(measured, reference, 6)
    assert_allclose(norm_calc, normalized, atol=1e-6)


def test_mlfm_6(measured, mlfm_6_coeffs):
    c_1, c_2, c_3, c_4, c_5, c_6, expected = mlfm_6_coeffs
    result = mlfm_6(measured, c_1, c_2, c_3, c_4, c_5, c_6)
    assert_allclose(expected, result[0], atol=1e-6)


def test_mlfm_norm_to_stack(normalized, reference, stacked):
    stack_calc = mlfm_norm_to_stack(normalized, reference, 6)
    assert_allclose(stack_calc, stacked, atol=1e-6)
