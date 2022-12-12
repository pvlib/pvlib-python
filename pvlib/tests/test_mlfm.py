import numpy as np
import pandas as pd

from pvlib import mlfm

import pytest

from conftest import requires_mpl, assert_frame_equal

from numpy.testing import assert_allclose

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
        beta_v_oc=-0.0035,  # 1/C
        gamma_pdc=-0.0045,  # = alpha_i_mp + beta_v_mp
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
    # meas['poa_global_kwm2'] = meas['poa_global'] / 1000
    meas['p_mp'] = meas['i_mp'] * meas['v_mp']
    meas['ff'] = meas['p_mp'] / (meas['i_sc'] * meas['v_oc'])

    return meas


@pytest.fixture
def normalized():
    data_norm_target = {
        # 'date_time':     ['2016-03-23 09:00:00-07:00'],
        'pr_dc':           [0.989242790817207],
        'pr_dc_temp_corr': [1.00183462464583],
        'i_sc':            [0.995586151149719],
        'i_mp':            [0.93628796685047],
        'v_oc':            [0.98475928597285],
        'v_mp':            [0.821773945683017],
        'v_oc_temp_corr':  [0.994508547151521],
        'r_sc':            [0.981487711004909],
        'r_oc':            [0.903706382978424],
        'i_ff':            [0.953947722780796],
        'v_ff':            [0.909337325885234],
    }

    norm_target = pd.DataFrame(data_norm_target)

    return norm_target


@pytest.fixture
def stacked6():
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
def stacked4():
    data_stack_target = {
        # 'date_time':       ['2016-03-23 09:00:00-07:00'],
        'pr_dc':             [0.989242790817],
        'i_sc':              [0.0054355995322],
        'i_mp':              [0.0734605702031],
        'i_v':               [0.01],
        'v_mp':              [0.214483151855],
        'v_oc':              [0.0187687482844],
        'temp_module_corr':  [0.012006092936],
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


@pytest.fixture
def matrix_data():
    # sample ghi, tmod, ws and pr_dc to fit
    # this data selectable from mlfm.ipynb
    # ---
    # select one of the following meas files
    # meas_file = 2  # <<< change from 0 to 2
    # ---

    return pd.DataFrame(np.array(
        [[100., 15, 0, 0.935774123487434],
         [200., 15, 0, 0.978281104560968],
         [400., 15, 0, 1.00721377598511],
         [600., 15, 0, 1.02254628193195],
         [800., 15, 0, 1.02710983555693],
         [1000., 15, 0, 1.02655910642259],
         [100., 25, 0, 0.907539559416693],
         [200., 25, 0, 0.94849519081601],   # LIC
         [400., 25, 0, 0.980840831523425],
         [600., 25, 0, 0.994311717861206],
         [800., 25, 0, 0.998914055228048],
         [1000., 25, 0, 1],                  # STC
         [1100., 25, 0, 0.998984571122331],
         [100., 50, 0, 0.833074775054297],
         [200., 50, 0, 0.879615265280794],
         [400., 50, 0, 0.908004964318957],
         [600., 50, 0, 0.920260626745268],
         [800., 50, 0, 0.925496431895749],
         [1000., 50, 0, 0.927551970214086],
         [1100., 50, 0, 0.926324993653569],
         [100., 75, 0, 0.746819733167856],
         [200., 75, 0, 0.792739683524666],
         [400., 75, 0, 0.826481538938877],
         [600., 75, 0, 0.842744854690247],
         [800., 75, 0, 0.847735029475644],
         [1000., 75, 0, 0.849053676698728],
         [1100., 75, 0, 0.849039573519871]]),
        columns=[
            'poa_global', 'temp_module', 'wind_speed', 'pr_dc'])


@pytest.fixture
def mlfm_6_fit():
    # fit matrix
    '''
    Excel fit GRG linear values
    c_1 = +1.0573318761708000
    c_2 = -0.0030251199627269
    c_3 = +0.1228522267570000
    c_4 = -0.0545505400372862
    c_5 = 0  # this is in conflict with the data which include wind_speed
    c_6 = -0.002394779219883
    rmse = 0.280%
    '''
    c_1 = +1.0579328401731174
    c_2 = -0.0030248261647759975
    c_3 = +0.12378885001559799
    c_4 = -0.05521716508715758
    c_5 = 0.
    c_6 = -0.0023546463713093836
    expected = 0.9845007615699125

    cc_target = [c_1, c_2, c_3, c_4, c_5, c_6]
    return c_1, c_2, c_3, c_4, c_5, c_6, expected, cc_target


def test_mlfm_meas_to_norm(mlfm_6_coeffs, reference, measured, normalized):
    norm_calc = mlfm.mlfm_meas_to_norm(measured, reference)
    assert_frame_equal(norm_calc, normalized, atol=1e-6)


def test_mlfm_6(measured, mlfm_6_coeffs):
    c_1, c_2, c_3, c_4, c_5, c_6, expected = mlfm_6_coeffs
    result = mlfm.mlfm_6(measured, c_1, c_2, c_3, c_4, c_5, c_6)
    assert_allclose(expected, result[0], atol=1e-6)


def test_mlfm_norm_to_stack(normalized, reference, stacked6, stacked4):
    stack_calc = mlfm.mlfm_norm_to_stack(normalized, reference['ff'])
    assert_frame_equal(stack_calc, stacked6, atol=1e-6)
    # test without 'i_ff', 'r_sc', 'v_ff', 'r_oc'
    # v_mp = v_ff * r_oc and i_mp = i_ff * r_sc
    norm = normalized.drop(columns=['i_ff', 'r_sc', 'v_ff', 'r_oc'])
    short_stack_calc = mlfm.mlfm_norm_to_stack(norm, reference['ff'])
    assert_frame_equal(short_stack_calc, stacked4, check_less_precise=True)


def test_mlfm_fit(matrix_data, mlfm_6_fit):
    c_1, c_2, c_3, c_4, c_5, c_6, expected, cc_target = mlfm_6_fit
    # choose which parameter to fit - usually pr_dc
    mlfm_sel = 'pr_dc'
    # drop wind_speed since it's always zero
    matrix_data = matrix_data.drop(columns=['wind_speed'])
    predictions, cc_fit, residuals, perr = mlfm.mlfm_fit(
        matrix_data, mlfm_sel)
    # atol is large due to different behavior in conda_linux Python 3.6 env.
    assert_allclose(cc_fit, cc_target, atol=5e-3)


@requires_mpl
def test_plot_mlfm_scatter(measured, normalized):
    import matplotlib.pyplot as plt
    fig = mlfm.plot_mlfm_scatter(measured, normalized, 'norm plot')
    assert isinstance(fig, plt.Figure)


@requires_mpl
def test_plot_mlfm_stack(measured, normalized, stacked6, stacked4, reference):
    # stacked plot requires at least index length of 2
    m = pd.concat([measured, measured])
    m.index = [0, 1]
    n = pd.concat([normalized, normalized])
    n.index = [0, 1]
    s6 = pd.concat([stacked6, stacked6])
    s6.index = [0, 1]
    import matplotlib.pyplot as plt
    fig = mlfm.plot_mlfm_stack(m, n, s6, reference['ff'], 'stacked 6 plot')
    assert isinstance(fig, plt.Figure)
    s4 = pd.concat([stacked4, stacked4])
    s4.index = [0, 1]
    import matplotlib.pyplot as plt
    fig = mlfm.plot_mlfm_stack(m, n, s4, reference['ff'], 'stacked 4 plot')
    assert isinstance(fig, plt.Figure)


"""

remove

reference()
measured()
normalized()
stacked6()
stacked4()
mlfm_6_coeffs()
matrix_data()
mlfm_6_fit()
test_mlfm_meas_to_norm(mlfm_6_coeffs, reference, measured, normalized)
test_mlfm_6(measured, mlfm_6_coeffs)
test_mlfm_norm_to_stack(normalized, reference, stacked6, stacked4)
test_mlfm_fit(matrix_data, mlfm_6_fit)
test_plot_mlfm_scatter(measured, normalized)
test_plot_mlfm_stack(measured, normalized, stacked6, stacked4, reference)
"""