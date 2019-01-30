"""
testing single-diode methods using JW Bishop 1988
"""

import numpy as np
from pvlib import pvsystem
from pvlib.singlediode import bishop88, estimate_voc, VOLTAGE_BUILTIN
import pytest
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
    """test pvsystem.singlediode with Brent method on FS495"""
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


def get_pvsyst_fs_495():
    """
    PVsyst parameters for First Solar FS-495 module from PVSyst-6.7.2 database.

    I_L_ref derived from Isc_ref conditions::

        I_L_ref = (I_sc_ref + Id + Ish) / (1 - d2mutau/(Vbi*N_s - Vd))

    where::

        Vd = I_sc_ref * R_s
        Id = I_o_ref * (exp(Vd / nNsVt) - 1)
        Ish = Vd / R_sh_ref

    """
    return {
        'd2mutau': 1.31, 'alpha_sc': 0.00039, 'gamma_ref': 1.48,
        'mu_gamma': 0.001, 'I_o_ref': 9.62e-10, 'R_sh_ref': 5000,
        'R_sh_0': 12500, 'R_sh_exp': 3.1, 'R_s': 4.6, 'beta_oc': -0.2116,
        'EgRef': 1.5, 'cells_in_series': 108, 'cells_in_parallel': 2,
        'I_sc_ref': 1.55, 'V_oc_ref': 86.5, 'I_mp_ref': 1.4, 'V_mp_ref': 67.85,
        'temp_ref': 25, 'irrad_ref': 1000, 'I_L_ref': 1.5743233463848496
    }


@pytest.mark.parametrize(
    'poa, temp_cell, expected, tol', [
        (
            get_pvsyst_fs_495()['irrad_ref'],
            get_pvsyst_fs_495()['temp_ref'],
            {
                'pmp': (get_pvsyst_fs_495()['I_mp_ref'] *
                        get_pvsyst_fs_495()['V_mp_ref']),
                'isc': get_pvsyst_fs_495()['I_sc_ref'],
                'voc': get_pvsyst_fs_495()['V_oc_ref']
            },
            (5e-4, 0.04)
        ),
        (POA, TCELL, {'pmp': 76.26, 'isc': 1.387, 'voc': 79.29}, (1e-3, 1e-3))]
)  # DeSoto @(888[W/m**2], 55[degC]) = {Pmp: 72.71, Isc: 1.402, Voc: 75.42)
def test_pvsyst_recombination_loss(poa, temp_cell, expected, tol):
    """test PVSst recombination loss"""
    pvsyst_fs_495 = get_pvsyst_fs_495()
    # first evaluate PVSyst model with thin-film recombination loss current
    # at reference conditions
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
        NsVbi=VOLTAGE_BUILTIN*pvsyst_fs_495['cells_in_series']
    )
    # test max power
    assert np.isclose(max(pvsyst[2]), expected['pmp'], *tol)
    # test short circuit current
    isc_pvsyst = np.interp(0, pvsyst[1], pvsyst[0])
    assert np.isclose(isc_pvsyst, expected['isc'], *tol)
    # test open circuit current
    voc_pvsyst = np.interp(0, pvsyst[0][::-1], pvsyst[1][::-1])
    assert np.isclose(voc_pvsyst, expected['voc'], *tol)
