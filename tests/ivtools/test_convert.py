import numpy as np
from pvlib.ivtools import convert


def test_convert_cec_pvsyst():
    cells_in_series = 66
    trina660_cec = {'I_L_ref': 18.4759, 'I_o_ref': 5.31e-12,
                    'EgRef': 1.121, 'dEgdT': -0.0002677,
                    'R_s': 0.159916, 'R_sh_ref': 113.991, 'a_ref': 1.8,
                    'Adjust': 6.42247, 'alpha_sc': 0.00629}
    trina660_pvsyst_est = convert.convert_cec_pvsyst(trina660_cec,
                                                     cells_in_series)
    pvsyst_expected = {'alpha_sc': 0.0096671,
                       'I_L_ref': 18.19305,
                       'I_o_ref': 6.94494e-12,
                       'EgRef': 1.121,
                       'R_s': 0.16318,
                       'R_sh_ref': 1000.947,
                       'R_sh_0': 8593.35,
                       'R_sh_exp': 5.5,
                       'gamma_ref': 1.0724,
                       'mu_gamma': -0.00074595,
                       'cells_in_series': 66}

    # set up dict of rtol, because some parameters are more sensitive to
    # optimization process than others
    rtol_d = {'alpha_sc': 1e-4,
              'I_L_ref': 1e-4,
              'I_o_ref': 1e-4,
              'EgRef': 1e-4,
              'R_s': 1e-4,
              'R_sh_ref': 1e-4,
              'R_sh_0': 1e-1,
              'R_sh_exp': 1e-3,
              'gamma_ref': 1e-4,
              'mu_gamma': 1e-4,
              'cells_in_series': 1e-8}
    
    assert np.all([np.isclose(trina660_pvsyst_est[k], pvsyst_expected[k],
                              rtol=rtol_d[k], atol=0.)
                   for k in pvsyst_expected])


def test_convert_pvsyst_cec():
    trina660_pvsyst = {'alpha_sc': 0.0074, 'I_L_ref': 18.464391,
                       'I_o_ref': 3.3e-11, 'EgRef': 1.121,
                       'R_s': 0.156, 'R_sh_ref': 200, 'R_sh_0': 800,
                       'R_sh_exp': 5.5, 'gamma_ref': 1.002, 'mu_gamma': 1e-3,
                       'cells_in_series': 66}
    trina660_cec_est = convert.convert_pvsyst_cec(trina660_pvsyst)

    cec_expected = {'alpha_sc': 0.0074,
                    'I_L_ref': 18.09421,
                    'I_o_ref': 2.46522e-14,
                    'EgRef': 1.121,
                    'dEgdT': -0.0002677,
                    'R_s': 0.098563,
                    'R_sh_ref': 268.508,
                    'a_ref': 1.2934,
                    'Adjust': 0.0065145}

    assert np.all([np.isclose(trina660_cec_est[k], cec_expected[k],
                              rtol=1e-4, atol=0.)
                   for k in cec_expected])
