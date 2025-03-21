import numpy as np
from pvlib import temperature
from pvlib import pvsystem

# Get the pvsyst_cell function directly
from pvlib.temperature import pvsyst_cell


def test_pvsyst_cell_semi_integrated():
    """Test semi-integrated PVsyst cell temperature model."""
    poa = 1000  # W/m^2
    temp_air = 25  # °C
    wind_speed = 1  # m/s
    
    # Test with default parameters for semi_integrated
    result = pvsyst_cell(poa, temp_air, wind_speed, u_c=20.0, u_v=0.0)
    expected = temp_air + ((poa * 0.9 * (1 - 0.1)) / (20.0 + 0.0 * wind_speed))
    assert np.isclose(result, expected)
    
    # Test with temperature.pvsyst_cell wrapper
    temp_params = {'u_c': 20.0, 'u_v': 0.0}
    mount = pvsystem.FixedMount(surface_tilt=30, surface_azimuth=180,
                                racking_model='semi_integrated')
    array = pvsystem.Array(mount=mount, module_type='glass_glass',
                          temperature_model_parameters=temp_params)
    result = array.get_cell_temperature(poa, temp_air, wind_speed, 'pvsyst')
    assert np.isclose(result, expected)