"""
testing single-diode methods using JW Bishop 1988
"""

import numpy as np
from pvlib import pvsystem
from pvlib.singlediode_methods import bishop88, estimate_voc, VOLTAGE_BUILTIN
from conftest import requires_scipy

POA = 888
TCELL = 55
CECMOD = pvsystem.retrieve_sam('cecmod')


@requires_scipy
def test_newton_spr_e20_327():
    """test pvsystem.singlediode with Newton method on SPR-E20-327"""
    spr_e20_327 = CECMOD.SunPower_SPR_E20_327
    x = pvsystem.calcparams_desoto(
        effective_irradiance=POA, temp_cell=TCELL,
        alpha_sc=spr_e20_327.alpha_sc, a_ref=spr_e20_327.a_ref,
        I_L_ref=spr_e20_327.I_L_ref, I_o_ref=spr_e20_327.I_o_ref,
        R_sh_ref=spr_e20_327.R_sh_ref, R_s=spr_e20_327.R_s,
        EgRef=1.121, dEgdT=-0.0002677)
    il, io, rs, rsh, nnsvt = x
    pvs = pvsystem.singlediode(*x, method='lambertw')
    out = pvsystem.singlediode(*x, method='newton')
    isc, voc, imp, vmp, pmp, ix, ixx = out.values()
    assert np.isclose(pvs['i_sc'], isc)
    assert np.isclose(pvs['v_oc'], voc)
    # the singlediode method doesn't actually get the MPP correct
    pvs_imp = pvsystem.i_from_v(rsh, rs, nnsvt, vmp, io, il, method='lambertw')
    pvs_vmp = pvsystem.v_from_i(rsh, rs, nnsvt, imp, io, il, method='lambertw')
    assert np.isclose(pvs_imp, imp)
    assert np.isclose(pvs_vmp, vmp)
    assert np.isclose(pvs['p_mp'], pmp)
    assert np.isclose(pvs['i_x'], ix)
    pvs_ixx = pvsystem.i_from_v(rsh, rs, nnsvt, (voc + vmp)/2, io, il,
                                method='lambertw')
    assert np.isclose(pvs_ixx, ixx)
    return isc, voc, imp, vmp, pmp, pvs


@requires_scipy
def test_newton_fs_495():
    """test pvsystem.singlediode with Newton method on FS495"""
    fs_495 = CECMOD.First_Solar_FS_495
    x = pvsystem.calcparams_desoto(
        effective_irradiance=POA, temp_cell=TCELL,
        alpha_sc=fs_495.alpha_sc, a_ref=fs_495.a_ref, I_L_ref=fs_495.I_L_ref,
        I_o_ref=fs_495.I_o_ref, R_sh_ref=fs_495.R_sh_ref, R_s=fs_495.R_s,
        EgRef=1.475, dEgdT=-0.0003)
    il, io, rs, rsh, nnsvt = x
    x += (101, )
    pvs = pvsystem.singlediode(*x, method='lambertw')
    out = pvsystem.singlediode(*x, method='newton')
    isc, voc, imp, vmp, pmp, ix, ixx, i, v = out.values()
    assert np.isclose(pvs['i_sc'], isc)
    assert np.isclose(pvs['v_oc'], voc)
    # the singlediode method doesn't actually get the MPP correct
    pvs_imp = pvsystem.i_from_v(rsh, rs, nnsvt, vmp, io, il, method='lambertw')
    pvs_vmp = pvsystem.v_from_i(rsh, rs, nnsvt, imp, io, il, method='lambertw')
    assert np.isclose(pvs_imp, imp)
    assert np.isclose(pvs_vmp, vmp)
    assert np.isclose(pvs['p_mp'], pmp)
    assert np.isclose(pvs['i_x'], ix)
    pvs_ixx = pvsystem.i_from_v(rsh, rs, nnsvt, (voc + vmp)/2, io, il,
                                method='lambertw')
    assert np.isclose(pvs_ixx, ixx)
    return isc, voc, imp, vmp, pmp, i, v, pvs


@requires_scipy
def test_brentq_spr_e20_327():
    """test pvsystem.singlediode with Brent method on SPR-E20-327"""
    spr_e20_327 = CECMOD.SunPower_SPR_E20_327
    x = pvsystem.calcparams_desoto(
        effective_irradiance=POA, temp_cell=TCELL,
        alpha_sc=spr_e20_327.alpha_sc, a_ref=spr_e20_327.a_ref,
        I_L_ref=spr_e20_327.I_L_ref, I_o_ref=spr_e20_327.I_o_ref,
        R_sh_ref=spr_e20_327.R_sh_ref, R_s=spr_e20_327.R_s,
        EgRef=1.121, dEgdT=-0.0002677)
    il, io, rs, rsh, nnsvt = x
    pvs = pvsystem.singlediode(*x, method='lambertw')
    out = pvsystem.singlediode(*x, method='brentq')
    isc, voc, imp, vmp, pmp, ix, ixx = out.values()
    assert np.isclose(pvs['i_sc'], isc)
    assert np.isclose(pvs['v_oc'], voc)
    # the singlediode method doesn't actually get the MPP correct
    pvs_imp = pvsystem.i_from_v(rsh, rs, nnsvt, vmp, io, il, method='lambertw')
    pvs_vmp = pvsystem.v_from_i(rsh, rs, nnsvt, imp, io, il, method='lambertw')
    assert np.isclose(pvs_imp, imp)
    assert np.isclose(pvs_vmp, vmp)
    assert np.isclose(pvs['p_mp'], pmp)
    assert np.isclose(pvs['i_x'], ix)
    pvs_ixx = pvsystem.i_from_v(rsh, rs, nnsvt, (voc + vmp)/2, io, il,
                                method='lambertw')
    assert np.isclose(pvs_ixx, ixx)
    return isc, voc, imp, vmp, pmp, pvs


@requires_scipy
def test_brentq_fs_495():
    """test pvsystem.singlediode with Brent method on SPR-E20-327"""
    fs_495 = CECMOD.First_Solar_FS_495
    x = pvsystem.calcparams_desoto(
        effective_irradiance=POA, temp_cell=TCELL,
        alpha_sc=fs_495.alpha_sc, a_ref=fs_495.a_ref, I_L_ref=fs_495.I_L_ref,
        I_o_ref=fs_495.I_o_ref, R_sh_ref=fs_495.R_sh_ref, R_s=fs_495.R_s,
        EgRef=1.475, dEgdT=-0.0003)
    il, io, rs, rsh, nnsvt = x
    x += (101, )
    pvs = pvsystem.singlediode(*x, method='lambertw')
    out = pvsystem.singlediode(*x, method='brentq')
    isc, voc, imp, vmp, pmp, ix, ixx, i, v = out.values()
    assert np.isclose(pvs['i_sc'], isc)
    assert np.isclose(pvs['v_oc'], voc)
    # the singlediode method doesn't actually get the MPP correct
    pvs_imp = pvsystem.i_from_v(rsh, rs, nnsvt, vmp, io, il, method='lambertw')
    pvs_vmp = pvsystem.v_from_i(rsh, rs, nnsvt, imp, io, il, method='lambertw')
    assert np.isclose(pvs_imp, imp)
    assert np.isclose(pvs_vmp, vmp)
    assert np.isclose(pvs['p_mp'], pmp)
    assert np.isclose(pvs['i_x'], ix)
    pvs_ixx = pvsystem.i_from_v(rsh, rs, nnsvt, (voc + vmp)/2, io, il,
                                method='lambertw')
    assert np.isclose(pvs_ixx, ixx)
    return isc, voc, imp, vmp, pmp, i, v, pvs


def pvsyst_fs_495():
    """
    PVSyst First Solar FS-495 parameters.

    Returns
    -------
    dictionary of PVSyst First Solar FS-495 parameters
    """
    fs_495 = dict(d2mutau=1.31, alpha_sc=0.00039, gamma_ref=1.48,
                  mu_gamma=0.001, I_o_ref=0.962e-9, R_sh_ref=5000,
                  R_sh_0=12500, R_sh_exp=3.1, R_s=4.6, beta_oc=-0.2116,
                  EgRef=1.475, cells_in_series=108, cells_in_parallel=2,
                  I_sc_ref=1.55, V_oc_ref=86.5, I_mp_ref=1.4,
                  V_mp_ref=67.85,
                  temp_ref=25, irrad_ref=1000)
    Vt = 0.025693001600485238  # thermal voltage at reference (V)
    nNsVt = fs_495['cells_in_series'] * fs_495['gamma_ref'] * Vt
    Vd = fs_495['I_sc_ref'] * fs_495['R_s']  # diode voltage at short circuit
    Id = fs_495['I_o_ref'] * (np.exp(Vd / nNsVt) - 1)  # diode current (A)
    Ish = Vd / fs_495['R_sh_ref']  # shunt current (A)
    # builtin potential difference (V)
    dv = VOLTAGE_BUILTIN * fs_495['cells_in_series'] - Vd
    # calculate photo-generated current at reference condition (A)
    fs_495['I_L_ref'] = (
        (fs_495['I_sc_ref'] + Id + Ish) / (1 - fs_495['d2mutau'] / dv)
    )
    return fs_495


PVSYST_FS_495 = pvsyst_fs_495()  # PVSyst First Solar FS-495 parameters


def test_pvsyst_fs_495_recombination_loss():
    """test pvsystem.singlediode with Brent method on SPR-E20-327"""
    poa, temp_cell = 1000.0, 25.0  # test conditions

    # first evaluate DeSoto model
    cec_fs_495 = CECMOD.First_Solar_FS_495  # CEC parameters for
    il_cec, io_cec, rs_cec, rsh_cec, nnsvt_cec = pvsystem.calcparams_desoto(
        effective_irradiance=poa, temp_cell=temp_cell,
        alpha_sc=cec_fs_495.alpha_sc, a_ref=cec_fs_495.a_ref,
        I_L_ref=cec_fs_495.I_L_ref, I_o_ref=cec_fs_495.I_o_ref,
        R_sh_ref=cec_fs_495.R_sh_ref, R_s=cec_fs_495.R_s,
        EgRef=1.475, dEgdT=-0.0003
    )
    voc_est_cec = estimate_voc(photocurrent=il_cec, saturation_current=io_cec,
                               nNsVth=nnsvt_cec)
    vd_cec = np.linspace(0, voc_est_cec, 1000)
    desoto = bishop88(
        diode_voltage=vd_cec, photocurrent=il_cec, saturation_current=io_cec,
        resistance_series=rs_cec, resistance_shunt=rsh_cec, nNsVth=nnsvt_cec
    )

    # now evaluate PVSyst model with thin-film recombination loss current
    pvsyst_fs_495 = PVSYST_FS_495
    x = pvsystem.calcparams_pvsyst(
        effective_irradiance=poa, temp_cell=temp_cell,
        alpha_sc=pvsyst_fs_495['alpha_sc'],
        gamma_ref=pvsyst_fs_495['gamma_ref'],
        mu_gamma=pvsyst_fs_495['mu_gamma'], I_L_ref=pvsyst_fs_495['I_L_ref'],
        I_o_ref=pvsyst_fs_495['I_o_ref'], R_sh_ref=pvsyst_fs_495['R_sh_ref'],
        R_sh_0=pvsyst_fs_495['R_sh_0'], R_sh_exp=pvsyst_fs_495['R_sh_exp'],
        R_s=pvsyst_fs_495['R_s'],
        cells_in_series=pvsyst_fs_495['cells_in_series'],
        EgRef=pvsyst_fs_495['EgRef']
    )
    il_pvsyst, io_pvsyst, rs_pvsyst, rsh_pvsyst, nnsvt_pvsyst = x
    voc_est_pvsyst = estimate_voc(photocurrent=il_pvsyst,
                                  saturation_current=io_pvsyst,
                                  nNsVth=nnsvt_pvsyst)
    vd_pvsyst = np.linspace(0, voc_est_pvsyst, 1000)
    pvsyst = bishop88(
        diode_voltage=vd_pvsyst, photocurrent=il_pvsyst,
        saturation_current=io_pvsyst, resistance_series=rs_pvsyst,
        resistance_shunt=rsh_pvsyst, nNsVth=nnsvt_pvsyst,
        d2mutau=pvsyst_fs_495['d2mutau'],
        cells_in_series=pvsyst_fs_495['cells_in_series']
    )

    # test expected change in max power
    assert np.isclose(max(desoto[2]) - max(pvsyst[2]), 0.01949420697212645)
    # test expected change in short circuit current
    isc_desoto = np.interp(0, desoto[1], desoto[0])
    isc_pvsyst = np.interp(0, pvsyst[1], pvsyst[0])
    assert np.isclose(isc_desoto - isc_pvsyst, -7.955827628380874e-05)
    # test expected change in open circuit current
    voc_desoto = np.interp(0, desoto[0][::-1], desoto[1][::-1])
    voc_pvsyst = np.interp(0, pvsyst[0][::-1], pvsyst[1][::-1])
    assert np.isclose(voc_desoto - voc_pvsyst, -0.04184247739671321)
    return desoto, pvsyst


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a, b = test_pvsyst_fs_495_recombination_loss()
    ab0 = np.interp(a[1], b[1], b[0])
    pmpa, pmpb = max(a[2]), max(b[2])
    isca = np.interp(0, a[1], a[0])
    iscb = np.interp(0, b[1], b[0])
    voca = np.interp(0, a[0][::-1], a[1][::-1])
    vocb = np.interp(0, b[0][::-1], b[1][::-1])
    f1 = plt.figure('power')
    plt.plot(a[1], a[2], a[1], ab0 * a[1], b[1], b[2], '--',
             a[1], a[2] - ab0 * a[1])
    plt.plot([a[1][0], a[1][-1]], [pmpa]*2, ':',
             [b[1][0], b[1][-1]], [pmpb]*2, ':')
    plt.legend(['DeSoto', 'PVSyst interpolated', 'PVSyst', '$\Delta P$',
                '$P_{mp,DeSoto}=%4.1f$' % pmpa,
                '$P_{mp,PVSyst}=%4.1f$' % pmpb])
    plt.grid()
    plt.xlabel('voltage (V)')
    plt.ylabel('power (W)')
    plt.title('FS-495 power, DeSoto vs. PVSyst with recombination loss')
    f1.show()
    f2 = plt.figure('current')
    plt.plot(a[1], a[0], a[1], ab0, b[1], b[0], '--', a[1], a[0] - ab0)
    plt.plot([a[1][0], a[1][-1]], [isca]*2, ':',
             [b[1][0], b[1][-1]], [iscb]*2, ':')
    plt.plot([voca]*2, [a[0][0], a[0][-1]], ':',
             [vocb]*2, [b[0][0], b[0][-1]], ':')
    plt.legend(['DeSoto', 'PVSyst interpolated', 'PVSyst', '$\Delta I$',
                '$I_{sc,DeSoto}=%4.2f$' % isca,
                '$I_{sc,PVSyst}=%4.2f$' % iscb,
                '$V_{oc,DeSoto}=%4.1f$' % voca,
                '$V_{oc,PVSyst}=%4.1f$' % vocb])
    plt.grid()
    plt.xlabel('voltage (V)')
    plt.ylabel('current (A)')
    plt.title('FS-495 IV-curve, DeSoto vs. PVSyst with recombination loss')
    f2.show()
