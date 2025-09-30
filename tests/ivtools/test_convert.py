import numpy as np
from pvlib.ivtools import sdm


def test_convert_cec_pvsyst():
    cells_in_series = 66
    trina660_cec = {'I_L_ref': 18.4759, 'I_o_ref': 5.31e-12,
                    'EgRef': 1.121, 'dEgdT': -0.0002677,
                    'R_s': 0.159916, 'R_sh_ref': 113.991, 'a_ref': 1.59068,
                    'Adjust': 6.42247, 'alpha_sc': 0.00629}
    trina660_pvsyst_est = sdm.convert_cec_pvsyst(trina660_cec,
                                                 cells_in_series)
    pvsyst_expected = {'alpha_sc': 0.007478218748188788,
                       'I_L_ref': 18.227679597516214,
                       'I_o_ref': 2.7418999402908e-11,
                       'EgRef': 1.121,
                       'R_s': 0.16331908293164496,
                       'R_sh_ref': 5267.928954454954,
                       'R_sh_0': 60171.206687871425,
                       'R_sh_exp': 5.5,
                       'gamma_ref': 1.0,
                       'mu_gamma': -6.349173477135307e-05,
                       'cells_in_series': 66}

    assert np.all([np.isclose(trina660_pvsyst_est[k], pvsyst_expected[k],
                              rtol=1e-3)
                   for k in pvsyst_expected])


def test_convert_pvsyst_cec():
    trina660_pvsyst = {'alpha_sc': 0.0074, 'I_o_ref': 3.3e-11, 'EgRef': 1.121,
                       'R_s': 0.156, 'R_sh_ref': 200, 'R_sh_0': 800,
                       'R_sh_exp': 5.5, 'gamma_ref': 1.002, 'mu_gamma': 1e-3,
                       'cells_in_series': 66}
    trina660_cec_est = sdm.convert_pvsyst_cec(trina660_pvsyst)
    cec_expected = {'alpha_sc': 0.0074,
                    'I_L_ref': 18.05154226834071,
                    'I_o_ref': 2.6863417875143392e-14,
                    'EgRef': 1.121,
                    'dEgdT': -0.0002677,
                    'R_s': 0.09436341848926795,
                    'a_ref': 1.2954800250731866,
                    'Adjust': 0.0011675969492410047,
                    'cells_in_series': 66}

    assert np.all([np.isclose(trina660_cec_est[k], cec_expected[k],
                              rtol=1e-3)
                   for k in cec_expected])
