import numpy as np

from numpy.testing import assert_allclose

from pvlib.ivtools import sdm
from pvlib import pvsystem

from tests.conftest import requires_statsmodels, TESTS_DATA_DIR


def _read_iv_curves_for_test(datafile, npts):
    """ read constants and npts IV curves from datafile """
    iv_specs = dict.fromkeys(['cells_in_series', 'alpha_sc', 'beta_voc',
                              'descr'])
    ivcurves = dict.fromkeys(['i_sc', 'i_mp', 'v_mp', 'v_oc', 'poa', 'tc',
                              'ee'])

    infilen = TESTS_DATA_DIR / datafile
    with infilen.open(mode='r') as f:

        Ns, aIsc, bVoc, descr = f.readline().split(',')

        iv_specs.update(
            cells_in_series=int(Ns), alpha_sc=float(aIsc),
            beta_voc=float(bVoc), descr=descr)

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

    ivcurves['i_sc'] = isc[:npts]
    ivcurves['i_mp'] = imp[:npts]
    ivcurves['v_oc'] = voc[:npts]
    ivcurves['v_mp'] = vmp[:npts]
    ivcurves['ee'] = ee[:npts]
    ivcurves['tc'] = tc[:npts]
    ivcurves['v'] = v[:npts]
    ivcurves['i'] = i[:npts]
    ivcurves['p_mp'] = ivcurves['v_mp'] * ivcurves['i_mp']  # power

    return iv_specs, ivcurves


def _read_pvsyst_expected(datafile):
    """ Read Pvsyst model parameters and diode equation values for each
    IV curve
    """
    pvsyst_specs = dict.fromkeys(['cells_in_series', 'alpha_sc', 'beta_voc',
                                  'descr'])
    # order required to match file being read
    paramlist = [
        'I_L_ref', 'I_o_ref', 'EgRef', 'R_sh_ref', 'R_sh_0', 'R_sh_exp', 'R_s',
        'gamma_ref', 'mu_gamma']
    varlist = ['iph', 'io', 'rs', 'rsh', 'u']
    pvsyst = dict.fromkeys(paramlist + varlist)

    infilen = TESTS_DATA_DIR / datafile
    with infilen.open(mode='r') as f:

        Ns, aIsc, bVoc, descr = f.readline().split(',')

        pvsyst_specs.update(
            cells_in_series=int(Ns), alpha_sc=float(aIsc),
            beta_voc=float(bVoc), descr=descr)

        tmp = [float(x) for x in f.readline().split(',')]
        # I_L_ref, I_o_ref, EgRef, R_s, R_sh_ref, R_sh_0, R_sh_exp, gamma_ref,
        # mu_gamma
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

    pvsyst.update(zip(varlist, [Iph, Io, Rs, Rsh, u]))

    return pvsyst_specs, pvsyst


@requires_statsmodels
def test_fit_pvsyst_sandia(npts=3000):

    # get IV curve data
    iv_specs, ivcurves = _read_iv_curves_for_test('PVsyst_demo.csv', npts)

    # get known Pvsyst model parameters and five parameters from each fitted
    # IV curve
    pvsyst_specs, pvsyst = _read_pvsyst_expected('PVsyst_demo_model.csv')

    modeled = sdm.fit_pvsyst_sandia(ivcurves, iv_specs)

    # calculate IV curves using the fitted model, for comparison with input
    # IV curves
    param_res = pvsystem.calcparams_pvsyst(
        effective_irradiance=ivcurves['ee'], temp_cell=ivcurves['tc'],
        alpha_sc=iv_specs['alpha_sc'], gamma_ref=modeled['gamma_ref'],
        mu_gamma=modeled['mu_gamma'], I_L_ref=modeled['I_L_ref'],
        I_o_ref=modeled['I_o_ref'], R_sh_ref=modeled['R_sh_ref'],
        R_sh_0=modeled['R_sh_0'], R_s=modeled['R_s'],
        cells_in_series=iv_specs['cells_in_series'], EgRef=modeled['EgRef'])
    iv_res = pvsystem.singlediode(*param_res)

    # assertions
    assert np.allclose(
        ivcurves['p_mp'], iv_res['p_mp'], equal_nan=True, rtol=0.038)
    assert np.allclose(
        ivcurves['v_mp'], iv_res['v_mp'], equal_nan=True, rtol=0.029)
    assert np.allclose(
        ivcurves['i_mp'], iv_res['i_mp'], equal_nan=True, rtol=0.021)
    assert np.allclose(
        ivcurves['i_sc'], iv_res['i_sc'], equal_nan=True, rtol=0.003)
    assert np.allclose(
        ivcurves['v_oc'], iv_res['v_oc'], equal_nan=True, rtol=0.019)
    # cells_in_series, alpha_sc, beta_voc, descr
    assert all((iv_specs[k] == pvsyst_specs[k]) for k in iv_specs.keys())
    # I_L_ref, I_o_ref, EgRef, R_sh_ref, R_sh_0, R_sh_exp, R_s, gamma_ref,
    # mu_gamma
    assert np.isclose(modeled['I_L_ref'], pvsyst['I_L_ref'], rtol=6.5e-5)
    assert np.isclose(modeled['I_o_ref'], pvsyst['I_o_ref'], rtol=0.15)
    assert np.isclose(modeled['R_s'], pvsyst['R_s'], rtol=0.0035)
    assert np.isclose(modeled['R_sh_ref'], pvsyst['R_sh_ref'], rtol=0.091)
    assert np.isclose(modeled['R_sh_0'], pvsyst['R_sh_0'], rtol=0.013)
    assert np.isclose(modeled['EgRef'], pvsyst['EgRef'], rtol=0.037)
    assert np.isclose(modeled['gamma_ref'], pvsyst['gamma_ref'], rtol=0.0045)
    assert np.isclose(modeled['mu_gamma'], pvsyst['mu_gamma'], rtol=0.064)

    # Iph, Io, Rsh, Rs, u
    mask = np.ones(modeled['u'].shape, dtype=bool)
    # exclude one curve with different convergence
    umask = mask.copy()
    umask[2540] = False
    assert all(modeled['u'][umask] == pvsyst['u'][:npts][umask])
    assert np.allclose(
        modeled['iph'][modeled['u']], pvsyst['iph'][:npts][modeled['u']],
        equal_nan=True, rtol=0.0009)
    assert np.allclose(
        modeled['io'][modeled['u']], pvsyst['io'][:npts][modeled['u']],
        equal_nan=True, rtol=0.096)
    assert np.allclose(
        modeled['rs'][modeled['u']], pvsyst['rs'][:npts][modeled['u']],
        equal_nan=True, rtol=0.035)
    # exclude one curve with Rsh outside 63% tolerance
    rshmask = modeled['u'].copy()
    rshmask[2545] = False
    assert np.allclose(
        modeled['rsh'][rshmask], pvsyst['rsh'][:npts][rshmask],
        equal_nan=True, rtol=0.63)


def test_pvsyst_temperature_coeff():
    # test for consistency with dP/dT estimated with secant rule
    params = {'alpha_sc': 0., 'gamma_ref': 1.1, 'mu_gamma': 0.,
              'I_L_ref': 6., 'I_o_ref': 5.e-9, 'R_sh_ref': 200.,
              'R_sh_0': 2000., 'R_s': 0.5, 'cells_in_series': 60}
    expected = -0.004886706494879083
    # params defines a Pvsyst model for a notional module.
    # expected value is created by calculating power at 1000 W/m2, and cell
    # temperature of 24 and 26C, using pvsystem.calcparams_pvsyst and
    # pvsystem.singlediode. The derivative (value for expected) is estimated
    # as the slope (p_mp at 26C - p_mp at 24C) / 2
    # using the secant rule for derivatives.
    gamma_pdc = sdm.pvsyst_temperature_coeff(
        params['alpha_sc'], params['gamma_ref'], params['mu_gamma'],
        params['I_L_ref'], params['I_o_ref'], params['R_sh_ref'],
        params['R_sh_0'], params['R_s'], params['cells_in_series'])
    assert_allclose(gamma_pdc, expected, rtol=0.0005)
