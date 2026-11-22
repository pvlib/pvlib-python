import numpy as np
import pandas as pd

import pytest

from pvlib.ivtools import sdm

from tests.conftest import requires_pysam


@requires_pysam
def test_fit_cec_sam(cec_params_cansol_cs5p_220p):
    input_data = cec_params_cansol_cs5p_220p['ivcurve']
    specs = cec_params_cansol_cs5p_220p['specs']
    I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = \
        sdm.fit_cec_sam(
            celltype='polySi', v_mp=input_data['V_mp_ref'],
            i_mp=input_data['I_mp_ref'], v_oc=input_data['V_oc_ref'],
            i_sc=input_data['I_sc_ref'], alpha_sc=specs['alpha_sc'],
            beta_voc=specs['beta_voc'],
            gamma_pmp=specs['gamma_pmp'],
            cells_in_series=specs['cells_in_series'])
    expected = pd.Series(cec_params_cansol_cs5p_220p['params'])
    modeled = pd.Series(index=expected.index, data=np.nan)
    modeled['a_ref'] = a_ref
    modeled['I_L_ref'] = I_L_ref
    modeled['I_o_ref'] = I_o_ref
    modeled['R_s'] = R_s
    modeled['R_sh_ref'] = R_sh_ref
    modeled['Adjust'] = Adjust
    assert np.allclose(modeled.values, expected.values, rtol=5e-2)


@requires_pysam
def test_fit_cec_sam_estimation_failure(cec_params_cansol_cs5p_220p):
    # Failing to estimate the parameters for the CEC SDM model should raise an
    # exception.
    with pytest.raises(RuntimeError):
        sdm.fit_cec_sam(celltype='polySi', v_mp=0.45, i_mp=5.25, v_oc=0.55,
                        i_sc=5.5, alpha_sc=0.00275, beta_voc=0.00275,
                        gamma_pmp=0.0055, cells_in_series=1, temp_ref=25)
