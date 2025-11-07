import numpy as np
import pandas as pd
from scipy import optimize

import pytest
from numpy.testing import assert_allclose

from pvlib.ivtools import sdm
from pvlib import pvsystem

from tests.conftest import requires_statsmodels


def test_fit_desoto():
    result, _ = sdm.fit_desoto(v_mp=31.0, i_mp=8.71, v_oc=38.3, i_sc=9.43,
                               alpha_sc=0.005658, beta_voc=-0.13788,
                               cells_in_series=60)
    result_expected = {'I_L_ref': 9.45232,
                       'I_o_ref': 3.22460e-10,
                       'R_s': 0.297814,
                       'R_sh_ref': 125.798,
                       'a_ref': 1.59128,
                       'alpha_sc': 0.005658,
                       'EgRef': 1.121,
                       'dEgdT': -0.0002677,
                       'irrad_ref': 1000,
                       'temp_ref': 25}
    assert np.allclose(pd.Series(result), pd.Series(result_expected),
                       rtol=1e-4)


def test_fit_desoto_init_guess(mocker):
    init_guess_array = np.array([9.4, 3.0e-10, 0.3, 125., 1.6])
    init_guess = {k: v for k, v in zip(
        ['IL_0', 'Io_0', 'Rs_0', 'Rsh_0', 'a_0'], init_guess_array)}
    spy = mocker.spy(optimize, 'root')
    result, _ = sdm.fit_desoto(v_mp=31.0, i_mp=8.71, v_oc=38.3, i_sc=9.43,
                               alpha_sc=0.005658, beta_voc=-0.13788,
                               cells_in_series=60, init_guess=init_guess)
    np.testing.assert_array_equal(init_guess_array, spy.call_args[1]['x0'])


def test_fit_desoto_init_bad_key():
    init_guess = {'IL_0': 6., 'bad_key': 0}
    with pytest.raises(ValueError, match='is not a valid name;'):
        result, _ = sdm.fit_desoto(v_mp=31.0, i_mp=8.71, v_oc=38.3, i_sc=9.43,
                                   alpha_sc=0.005658, beta_voc=-0.13788,
                                   cells_in_series=60, init_guess=init_guess)


def test_fit_desoto_failure():
    with pytest.raises(RuntimeError) as exc:
        sdm.fit_desoto(v_mp=31.0, i_mp=8.71, v_oc=38.3, i_sc=9.43,
                       alpha_sc=0.005658, beta_voc=-0.13788,
                       cells_in_series=10)
    assert ('Parameter estimation failed') in str(exc.value)


@requires_statsmodels
def test_fit_desoto_sandia(cec_params_cansol_cs5p_220p):
    # this test computes a set of IV curves for the input fixture, fits
    # the De Soto model to the calculated IV curves, and compares the fitted
    # parameters to the starting values
    params = cec_params_cansol_cs5p_220p['params']
    params.pop('Adjust')
    specs = cec_params_cansol_cs5p_220p['specs']
    effective_irradiance = np.array([400., 500., 600., 700., 800., 900.,
                                     1000.])
    temp_cell = np.array([15., 25., 35., 45.])
    ee = np.tile(effective_irradiance, len(temp_cell))
    tc = np.repeat(temp_cell, len(effective_irradiance))
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
        ee, tc, alpha_sc=specs['alpha_sc'], **params)
    ivcurve_params = dict(photocurrent=IL, saturation_current=I0,
                          resistance_series=Rs, resistance_shunt=Rsh,
                          nNsVth=nNsVth)
    sim_ivcurves = pvsystem.singlediode(**ivcurve_params).to_dict('series')
    v = np.linspace(0., sim_ivcurves['v_oc'], 300)
    i = pvsystem.i_from_v(voltage=v, **ivcurve_params)
    sim_ivcurves.update(v=v.T, i=i.T, ee=ee, tc=tc)

    result = sdm.fit_desoto_sandia(sim_ivcurves, specs)
    modeled = pd.Series(index=params.keys(), data=np.nan)
    modeled['a_ref'] = result['a_ref']
    modeled['I_L_ref'] = result['I_L_ref']
    modeled['I_o_ref'] = result['I_o_ref']
    modeled['R_s'] = result['R_s']
    modeled['R_sh_ref'] = result['R_sh_ref']
    expected = pd.Series(params)
    assert np.allclose(modeled[params.keys()].values,
                       expected[params.keys()].values, rtol=5e-2)
    assert_allclose(result['dEgdT'], -0.0002677)
    assert_allclose(result['EgRef'], 1.3112547292120638)
    assert_allclose(result['cells_in_series'], specs['cells_in_series'])


def test_fit_desoto_batzelis():
    params = {'i_sc': 15.98, 'v_oc': 50.26, 'i_mp': 15.27, 'v_mp': 42.57,
              'alpha_sc': 0.007351, 'beta_voc': -0.120624}
    expected = {  # calculated with the function itself
        'alpha_sc': 0.007351,
        'a_ref': 1.7257632483733483,
        'I_L_ref': 15.985408866796396,
        'I_o_ref': 3.594308384705643e-12,
        'R_sh_ref': 389.4379947026243,
        'R_s': 0.13181590981241956,
    }
    out = sdm.fit_desoto_batzelis(**params)
    for k in expected:
        assert out[k] == pytest.approx(expected[k])

    # ensure the STC values are reproduced
    iv = pvsystem.singlediode(out['I_L_ref'], out['I_o_ref'], out['R_s'],
                              out['R_sh_ref'], out['a_ref'])
    assert iv['i_sc'] == pytest.approx(params['i_sc'])
    assert iv['i_mp'] == pytest.approx(params['i_mp'], rel=3e-3)
    assert iv['v_oc'] == pytest.approx(params['v_oc'], rel=3e-4)
    assert iv['v_mp'] == pytest.approx(params['v_mp'], rel=4e-3)
