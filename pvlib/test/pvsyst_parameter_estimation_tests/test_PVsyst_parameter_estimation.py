from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import pvlib
from pvlib.ivtools.PVsyst_parameter_estimation import numdiff
from pvlib.ivtools import PVsyst_parameter_estimation
from conftest import requires_scipy, requires_statsmodels

BASEDIR = os.path.dirname(__file__)
TESTDIR = os.path.dirname(BASEDIR)
PROJDIR = os.path.dirname(TESTDIR)
DATADIR = os.path.join(PROJDIR, 'data')
TESTDATA = os.path.join(DATADIR, 'ivtools_numdiff.dat')


def test_numdiff():
    iv = pd.read_csv(
        TESTDATA, names=['I', 'V', 'dIdV', 'd2IdV2'], dtype=float)
    df, d2f = numdiff(iv.V, iv.I)
    assert np.allclose(iv.dIdV, df, equal_nan=True)
    assert np.allclose(iv.d2IdV2, d2f, equal_nan=True)


@requires_scipy
@requires_statsmodels
def test_pvsyst_parameter_estimation(disp=False, npts=3000):
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

    expected = PVsyst_parameter_estimation.pvsyst_parameter_estimation(
        ivcurves, iv_specs)
    param_res = pvlib.pvsystem.calcparams_pvsyst(
        effective_irradiance=ivcurves['ee'], temp_cell=ivcurves['tc'],
        alpha_sc=iv_specs['aisc'], gamma_ref=expected['gamma_ref'],
        mu_gamma=expected['mugamma'], I_L_ref=expected['IL_ref'],
        I_o_ref=expected['Io_ref'], R_sh_ref=expected['Rsh_ref'],
        R_sh_0=expected['Rsh0'], R_s=expected['Rs_ref'],
        cells_in_series=iv_specs['ns'], EgRef=expected['eG'])
    iv_res = pvlib.pvsystem.singlediode(*param_res)

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
    assert all(expected['u'] == pvsyst['u'][:npts])
    assert np.allclose(
        expected['Iph'][expected['u']], pvsyst['Iph'][:npts][expected['u']],
        equal_nan=True, rtol=0.0009)
    assert np.allclose(
        expected['Io'][expected['u']], pvsyst['Io'][:npts][expected['u']],
        equal_nan=True, rtol=0.096)
    assert np.allclose(
        expected['Rs'][expected['u']], pvsyst['Rs'][:npts][expected['u']],
        equal_nan=True, rtol=0.03)
    assert np.allclose(
        expected['Rsh'][expected['u']], pvsyst['Rsh'][:npts][expected['u']],
        equal_nan=True, rtol=0.63)


if __name__ == "__main__":
    (expected,
     pvsyst,
     ivcurves,
     iv_specs,
     param_res,
     iv_res) = test_pvsyst_parameter_estimation(disp=True)
