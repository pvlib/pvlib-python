import numpy as np
from pvlib.temperature import noct_sam

def test_noct_sam_effective_irr_reduces_temp():
    # Effective irradiance should lower predicted module temp
    t_full = noct_sam(
        poa_global=800, temp_air=20, wind_speed=2,
        noct=45, module_efficiency=0.20
    )

    t_eff_600 = noct_sam(
        poa_global=800, temp_air=20, wind_speed=2,
        noct=45, module_efficiency=0.20,
        effective_irradiance=600
    )

    t_eff_200 = noct_sam(
        poa_global=800, temp_air=20, wind_speed=2,
        noct=45, module_efficiency=0.20,
        effective_irradiance=200
    )

    assert t_eff_600 < t_full
    assert t_eff_200 < t_eff_600


def test_noct_sam_oc_case_reduces_temp():
    # Open-circuit (efficiency=0) should still respond to effective irradiance
    t_oc_full = noct_sam(
        poa_global=800, temp_air=20, wind_speed=2,
        noct=45, module_efficiency=0.0
    )

    t_oc_eff = noct_sam(
        poa_global=800, temp_air=20, wind_speed=2,
        noct=45, module_efficiency=0.0,
        effective_irradiance=300
    )

    assert t_oc_eff < t_oc_full


def test_noct_sam_not_below_ambient_for_small_eff():
    # Very small effective irradiance should not predict cooling below ambient
    t_small = noct_sam(
        poa_global=800, temp_air=20, wind_speed=2,
        noct=45, module_efficiency=0.20,
        effective_irradiance=1.0
    )

    assert t_small >= 20.0 - 1e-6  # small numerical tolerance
