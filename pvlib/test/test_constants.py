from pvlib import constants


def test_elementary_charge():
    # This catches when the value changes
    assert constants.elementary_charge_C == 1.6021766208e-19


def test_boltzmann():
    # This catches when the value changes
    assert constants.boltzmann_J_per_K == 1.38064852e-23


def test_T_stc_degC():
    # This catches when the standard value changes
    assert constants.T_stc_degC == 25.


def test_T_stc_K():
    # This catches when the standard value changes
    assert constants.T_stc_K == 298.15
