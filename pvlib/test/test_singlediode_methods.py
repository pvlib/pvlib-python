"""
testing single-diode methods using JW Bishop 1988
"""

import numpy as np
from pvlib import pvsystem
from pvlib.singlediode_methods import bishop88, estimate_voc
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


def test_fs_495_recombination_loss():
    """test pvsystem.singlediode with Brent method on SPR-E20-327"""
    fs_495 = CECMOD.First_Solar_FS_495
    d2mutau_fs_495 = 1.3  # pvsyst recombination loss parameter (volts)
    x = pvsystem.calcparams_desoto(
        effective_irradiance=POA, temp_cell=TCELL,
        alpha_sc=fs_495.alpha_sc, a_ref=fs_495.a_ref, I_L_ref=fs_495.I_L_ref,
        I_o_ref=fs_495.I_o_ref, R_sh_ref=fs_495.R_sh_ref, R_s=fs_495.R_s,
        EgRef=1.475, dEgdT=-0.0003)
    il, io, rs, rsh, nnsvt = x
    voc_est = estimate_voc(photocurrent=il, saturation_current=io,
                           nNsVth=nnsvt)
    vd = np.linspace(0, voc_est, 1000)
    noloss = bishop88(
        diode_voltage=vd, photocurrent=il, saturation_current=io,
        resistance_series=rs, resistance_shunt=rsh, nNsVth=nnsvt,
             d2mutau=0, cells_in_series=fs_495.N_s/2, gradients=False)
    wloss = bishop88(
        diode_voltage=vd, photocurrent=il, saturation_current=io,
        resistance_series=rs, resistance_shunt=rsh, nNsVth=nnsvt,
        d2mutau=d2mutau_fs_495, cells_in_series=fs_495.N_s/2, gradients=False)
    # test expected change in max power
    assert np.isclose(max(noloss[2]) - max(wloss[2]), 2.906285338766608)
    # test expected change in short circuit current
    isc_noloss = np.interp(0, noloss[1], noloss[0])
    isc_wloss = np.interp(0, wloss[1], wloss[0])
    assert np.isclose(isc_noloss - isc_wloss, 0.020758929386382796)
    # test expected change in open circuit current
    voc_noloss = np.interp(0, noloss[0][::-1], noloss[1][::-1])
    voc_wloss = np.interp(0, wloss[0][::-1], wloss[1][::-1])
    assert np.isclose(voc_noloss - voc_wloss, 0.2098581317348902)
    return noloss, wloss


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a, b = test_fs_495_recombination_loss()
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
    plt.legend(['no loss', 'w/loss interp', 'w/loss', '$\Delta P_{mp}$',
                '$P_{mp,noloss}=%4.1f$' % pmpa,
                '$P_{mp,wloss}=%4.1f$' % pmpb])
    plt.grid()
    plt.xlabel('voltage (V)')
    plt.ylabel('power (P)')
    plt.title('FS-495 power with and w/o recombination loss')
    f1.show()
    f2 = plt.figure('current')
    plt.plot(a[1], a[0], a[1], ab0, b[1], b[0], '--', a[1], a[0] - ab0)
    plt.plot([a[1][0], a[1][-1]], [isca]*2, ':',
             [b[1][0], b[1][-1]], [iscb]*2, ':')
    plt.plot([voca]*2, [a[0][0], a[0][-1]], ':',
             [vocb]*2, [b[0][0], b[0][-1]], ':')
    plt.legend(['no loss', 'w/loss interp', 'w/loss', '$\Delta I$',
                '$I_{sc,noloss}=%4.2f$' % isca,
                '$I_{sc,wloss}=%4.2f$' % iscb,
                '$V_{oc,noloss}=%4.1f$' % voca,
                '$V_{oc,wloss}=%4.1f$' % vocb])
    plt.grid()
    plt.xlabel('voltage (V)')
    plt.ylabel('current (I)')
    plt.title('FS-495 IV-curve with and w/o recombination loss')
    f2.show()
