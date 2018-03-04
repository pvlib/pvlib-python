from pvlib import constants


def test_elementary_charge():
    # This would catch when the value provided by scipy changes
    assert constants.elementary_charge_C == 1.6021766208e-19


def test_boltzmann():
    # This would catch when the value provided by scipy changes
    assert constants.boltzmann_J_per_K == 1.38064852e-23


def test_degC_to_K_offset():
    assert constants.degC_to_K_offset == 273.15


def test_T_stc_C():
    assert constants.T_stc_C == 25.


def test_T_stc_K():
    assert constants.T_stc_K == 298.15
