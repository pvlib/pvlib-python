import os
import numpy as np
import pandas as pd
from collections import OrderedDict

import pytest

from pvlib.ivtools import fit_sdm
from pvlib import pvsystem

from pvlib.test.conftest import requires_scipy, requires_pysam
from pvlib.test.conftest import requires_statsmodels

BASEDIR = os.path.dirname(__file__)
TESTDIR = os.path.dirname(BASEDIR)
PROJDIR = os.path.dirname(TESTDIR)
DATADIR = os.path.join(PROJDIR, 'data')
TESTDATA = os.path.join(DATADIR, 'ivtools_numdiff.dat')


@pytest.fixture
def get_test_iv_params():
    return {'IL': 8.0, 'I0': 5e-10, 'Rsh': 1000, 'Rs': 0.2, 'nNsVth': 1.61864}


@pytest.fixture
def get_cec_params_cansol_cs5p_220p():
    return {'input': {'V_mp_ref': 46.6, 'I_mp_ref': 4.73, 'V_oc_ref': 58.3,
                      'I_sc_ref': 5.05, 'alpha_sc': 0.0025,
                      'beta_voc': -0.19659, 'gamma_pmp': -0.43,
                      'cells_in_series': 96},
            'output': {'a_ref': 2.3674, 'I_L_ref': 5.056, 'I_o_ref': 1.01e-10,
                       'R_sh_ref': 837.51, 'R_s': 1.004, 'Adjust': 2.3}}


@requires_pysam
def test_fit_sdm_cec_sam(get_cec_params_cansol_cs5p_220p):
    input_data = get_cec_params_cansol_cs5p_220p['input']
    I_L_ref, I_o_ref, R_sh_ref, R_s, a_ref, Adjust = \
        fit_sdm.fit_sdm_cec_sam(
            celltype='polySi', v_mp=input_data['V_mp_ref'],
            i_mp=input_data['I_mp_ref'], v_oc=input_data['V_oc_ref'],
            i_sc=input_data['I_sc_ref'], alpha_sc=input_data['alpha_sc'],
            beta_voc=input_data['beta_voc'],
            gamma_pmp=input_data['gamma_pmp'],
            cells_in_series=input_data['cells_in_series'])
    expected = pd.Series(get_cec_params_cansol_cs5p_220p['output'])
    modeled = pd.Series(index=expected.index, data=np.nan)
    modeled['a_ref'] = a_ref
    modeled['I_L_ref'] = I_L_ref
    modeled['I_o_ref'] = I_o_ref
    modeled['R_sh_ref'] = R_sh_ref
    modeled['R_s'] = R_s
    modeled['Adjust'] = Adjust
    assert np.allclose(modeled.values, expected.values, rtol=5e-2)
    # test for fitting failure
    with pytest.raises(RuntimeError):
        I_L_ref, I_o_ref, R_sh_ref, R_s, a_ref, Adjust = \
            fit_sdm.fit_sdm_cec_sam(
                celltype='polySi', v_mp=0.45, i_mp=5.25, v_oc=0.55, i_sc=5.5,
                alpha_sc=0.00275, beta_voc=0.00275, gamma_pmp=0.0055,
                cells_in_series=1, temp_ref=25)


@requires_scipy
def test_fit_sdm_desoto():
    result, _ = fit_sdm.fit_sdm_desoto(v_mp=31.0, i_mp=8.71, v_oc=38.3,
                                       i_sc=9.43, alpha_sc=0.005658,
                                       beta_voc=-0.13788,
                                       cells_in_series=60)
    result_expected = {'I_L_ref': 9.45232,
                       'I_o_ref': 3.22460e-10,
                       'a_ref': 1.59128,
                       'R_sh_ref': 125.798,
                       'R_s': 0.297814,
                       'alpha_sc': 0.005658,
                       'EgRef': 1.121,
                       'dEgdT': -0.0002677,
                       'irrad_ref': 1000,
                       'temp_ref': 25}
    assert np.allclose(pd.Series(result), pd.Series(result_expected),
                       rtol=1e-4)


@requires_scipy
def test_fit_sdm_desoto_failure():
    with pytest.raises(RuntimeError) as exc:
        fit_sdm.fit_sdm_desoto(v_mp=31.0, i_mp=8.71, v_oc=38.3, i_sc=9.43,
                               alpha_sc=0.005658, beta_voc=-0.13788,
                               cells_in_series=10)
    assert ('Parameter estimation failed') in str(exc.value)


@requires_scipy
@requires_statsmodels
def test_fit_pvsyst_sandia(disp=False, npts=3000):
    spec_list = ['ns', 'aisc', 'bvoc', 'descr']
    iv_specs = dict.fromkeys(spec_list)
    keylist = ['isc', 'imp', 'vmp', 'voc', 'poa', 'tc', 'ee']
    ivcurves = dict.fromkeys(keylist)

    with open(os.path.join(BASEDIR, 'PVsyst_demo.txt'), 'r') as f:

        Ns, aIsc, bVoc, descr = f.readline().split(',')

        iv_specs.update(
            ns=int(Ns), aisc=float(aIsc), bvoc=float(bVoc), descr=descr)

        strN, strM = f.readline().split(',')
        N = int(strN)
        M = int(strM)

        isc = np.empty(N)
        imp = np.empty(N)
        vmp = np.empty(N)
        voc = np.empty(N)
        ee = np.empty(N)
        poa = np.empty(N)
        tc = np.empty(N)
        v = np.empty((N, M))
        i = np.empty((N, M))
        v[:] = np.nan  # fill with nan
        i[:] = np.nan

        for k in range(N):
            tmp = (float(x) for x in f.readline().split(','))
            isc[k], imp[k], vmp[k], voc[k], poa[k], tc[k], ee[k] = tmp
            # read voltage and current
            tmp = [float(x) for x in f.readline().split(',')]
            while len(tmp) < M:
                tmp.append(np.nan)
            v[k, :] = tmp
            tmp = [float(x) for x in f.readline().split(',')]
            while len(tmp) < M:
                tmp.append(np.nan)
            i[k, :] = tmp

    ivcurves['isc'] = isc[:npts]
    ivcurves['imp'] = imp[:npts]
    ivcurves['voc'] = voc[:npts]
    ivcurves['vmp'] = vmp[:npts]
    ivcurves['ee'] = ee[:npts]
    ivcurves['tc'] = tc[:npts]
    ivcurves['v'] = v[:npts]
    ivcurves['i'] = i[:npts]

    pvsyst_specs = dict.fromkeys(spec_list)
    paramlist = [
        'IL_ref', 'Io_ref', 'eG', 'Rsh_ref', 'Rsh0', 'Rshexp', 'Rs_ref',
        'gamma_ref', 'mugamma']
    varlist = ['Iph', 'Io', 'Rsh', 'Rs', 'u']
    pvsyst = OrderedDict(key=(paramlist + varlist))

    with open(os.path.join(BASEDIR, 'PVsyst_demo_model.txt'), 'r') as f:

        Ns, aIsc, bVoc, descr = f.readline().split(',')

        pvsyst_specs.update(
            ns=int(Ns), aisc=float(aIsc), bvoc=float(bVoc), descr=descr)

        tmp = [float(x) for x in f.readline().split(',')]
        # IL_ref, Io_ref, eG, Rsh_ref, Rsh0, Rshexp, Rs_ref, gamma_ref, mugamma
        pvsyst.update(zip(paramlist, tmp))

        strN = f.readline()
        N = int(strN)

        Iph = np.empty(N)
        Io = np.empty(N)
        Rsh = np.empty(N)
        Rs = np.empty(N)
        u = np.empty(N)

        for k in range(N):
            tmp = [float(x) for x in f.readline().split(',')]
            Iph[k], Io[k], Rsh[k], Rs[k], u[k] = tmp

    pvsyst.update(zip(varlist, [Iph, Io, Rsh, Rs, u]))

    expected = fit_sdm.fit_pvsyst_sandia(ivcurves, iv_specs)
    param_res = pvsystem.calcparams_pvsyst(
        effective_irradiance=ivcurves['ee'], temp_cell=ivcurves['tc'],
        alpha_sc=iv_specs['aisc'], gamma_ref=expected['gamma_ref'],
        mu_gamma=expected['mugamma'], I_L_ref=expected['IL_ref'],
        I_o_ref=expected['Io_ref'], R_sh_ref=expected['Rsh_ref'],
        R_sh_0=expected['Rsh0'], R_s=expected['Rs_ref'],
        cells_in_series=iv_specs['ns'], EgRef=expected['eG'])
    iv_res = pvsystem.singlediode(*param_res)

    ivcurves['pmp'] = ivcurves['vmp'] * ivcurves['imp']  # power
    if disp:
        return expected, pvsyst, ivcurves, iv_specs, param_res, iv_res

    # assertions
    assert np.allclose(
        ivcurves['pmp'], iv_res['p_mp'], equal_nan=True, rtol=0.038)
    assert np.allclose(
        ivcurves['vmp'], iv_res['v_mp'], equal_nan=True, rtol=0.029)
    assert np.allclose(
        ivcurves['imp'], iv_res['i_mp'], equal_nan=True, rtol=0.021)
    assert np.allclose(
        ivcurves['isc'], iv_res['i_sc'], equal_nan=True, rtol=0.003)
    assert np.allclose(
        ivcurves['voc'], iv_res['v_oc'], equal_nan=True, rtol=0.019)
    # ns, aisc, bvoc, descr
    assert all((iv_specs[spec] == pvsyst_specs[spec]) for spec in spec_list)
    # IL_ref, Io_ref, eG, Rsh_ref, Rsh0, Rshexp, Rs_ref, gamma_ref, mugamma
    assert np.isclose(expected['IL_ref'], pvsyst['IL_ref'], rtol=6.5e-5)
    assert np.isclose(expected['Io_ref'], pvsyst['Io_ref'], rtol=0.15)
    assert np.isclose(expected['Rs_ref'], pvsyst['Rs_ref'], rtol=0.0035)
    assert np.isclose(expected['Rsh_ref'], pvsyst['Rsh_ref'], rtol=0.091)
    assert np.isclose(expected['Rsh0'], pvsyst['Rsh0'], rtol=0.013)
    assert np.isclose(expected['eG'], pvsyst['eG'], rtol=0.037)
    assert np.isclose(expected['gamma_ref'], pvsyst['gamma_ref'], rtol=0.0045)
    assert np.isclose(expected['mugamma'], pvsyst['mugamma'], rtol=0.064)

    # Iph, Io, Rsh, Rs, u
    mask = np.ones(expected['u'].shape, dtype=bool)
    # exclude one curve with different convergence
    umask = mask.copy()
    umask[2540] = False
    assert all(expected['u'][umask] == pvsyst['u'][:npts][umask])
    assert np.allclose(
        expected['Iph'][expected['u']], pvsyst['Iph'][:npts][expected['u']],
        equal_nan=True, rtol=0.0009)
    assert np.allclose(
        expected['Io'][expected['u']], pvsyst['Io'][:npts][expected['u']],
        equal_nan=True, rtol=0.096)
    assert np.allclose(
        expected['Rs'][expected['u']], pvsyst['Rs'][:npts][expected['u']],
        equal_nan=True, rtol=0.035)
    # exclude one curve with Rsh outside 63% tolerance
    rshmask = expected['u'].copy()
    rshmask[2545] = False
    assert np.allclose(
        expected['Rsh'][rshmask], pvsyst['Rsh'][:npts][rshmask],
        equal_nan=True, rtol=0.63)
