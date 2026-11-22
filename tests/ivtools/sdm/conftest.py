import pytest


@pytest.fixture
def cec_params_cansol_cs5p_220p():
    return {'ivcurve': {'V_mp_ref': 46.6, 'I_mp_ref': 4.73, 'V_oc_ref': 58.3,
                        'I_sc_ref': 5.05},
            'specs': {'alpha_sc': 0.0025, 'beta_voc': -0.19659,
                      'gamma_pmp': -0.43, 'cells_in_series': 96},
            'params': {'I_L_ref': 5.056, 'I_o_ref': 1.01e-10,
                       'R_sh_ref': 837.51, 'R_s': 1.004, 'a_ref': 2.3674,
                       'Adjust': 2.3}}
