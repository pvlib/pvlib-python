import numpy as np
from pvlib.temperature import noct_sam
def test_noct_sam_effective_irr_reduces_temp():
    # Test that effective irradiance lowers predicted cell temperature
    t_full = noct_sam(poa_global=800, temp_air=20, wind_speed=2,
                      noct=45, module_efficiency=0.20)
    # Lower effective irradiance should reduce modeled temperature
    t_eff_600 = noct_sam(poa_global=800, temp_air=20, wind_speed=2,
                         noct=45, module_efficiency=0.20,
                         effective_irradiance=600)
    t_eff_200 = noct_sam(poa_global=800, temp_air=20, wind_speed=2,
                         noct=45, module_efficiency=0.20,
                         effective_irradiance=200)
    assert t_eff_600 < t_full
    assert t_eff_200 < t_eff_600
def test_noct_sam_oc_case_reduces_temp():
    # Test that open-circuit modules (efficiency=0) still respond to effective irradiance
    t_oc_full = noct_sam(poa_global=800, temp_air=20, wind_speed=2,
                         noct=45, module_efficiency=0.0)
    # Reducing effective irradiance should still lower modeled temperature
    t_oc_eff = noct_sam(poa_global=800, temp_air=20, wind_speed=2,
                        noct=45, module_efficiency=0.0,
                        effective_irradiance=300)
    assert t_oc_eff < t_oc_full

def test_noct_sam_not_below_ambient_for_small_eff():
    # Test that extremely small effective irradiance never predicts temperature below ambient
    t_small = noct_sam(poa_global=800, temp_air=20, wind_speed=2,
                       noct=45, module_efficiency=0.20,
                       effective_irradiance=1.0)
    # Allow tiny numerical tolerance
    assert t_small >= 20.0 - 1e-6
