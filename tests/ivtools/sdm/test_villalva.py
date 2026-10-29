import numpy as np
from numpy.testing import assert_allclose

from pvlib.ivtools.sdm import fit_villalva


def test_fit_villalva_roundtrip():
    """Recover archived SDM parameters from synthetic STC key points."""
    # Reference values archived from
    # docs/examples/iv-modeling/plot_villalva.py.
    #
    # The key points were generated with pvlib.pvsystem.singlediode
    # from the Canadian Solar CS5P-220M SDM parameters used in the
    # gallery validation example.
    params, _ = fit_villalva(
        v_mp=46.90263243704791,
        i_mp=4.6898061801383335,
        v_oc=59.40065126750111,
        i_sc=5.099770128570936,
        alpha_sc=0.004539,
        beta_voc=-0.22216,
        cells_in_series=96,
        diode_factor=1.0692532995822863,
        temp_ref=25.0,
        irrad_ref=1000.0,
        rs_step=1e-4,
        rs_max=1.5,
    )

    expected = {
        "I_L_ref": 5.113999993730909,
        "I_o_ref": 8.196000001605032e-10,
        "R_s": 1.065,
        "R_sh_ref": 381.68001775625714,
        "a_ref": 2.6373,
    }

    for key, value in expected.items():
        assert_allclose(params[key], value, rtol=1e-6, atol=0.0)
