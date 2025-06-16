import pvlib
import pandas as pd
import numpy as np

def test_ground_diffuse_integration():
    """Test that get_ground_diffuse works correctly with poa_components."""

    # Create sample data
    surface_tilt = 30  # degrees
    ghi = 1000  # W/m²
    albedo = 0.25

    # Calculate ground diffuse using get_ground_diffuse
    ground_diffuse = pvlib.irradiance.get_ground_diffuse(
        surface_tilt=surface_tilt,
        ghi=ghi,
        albedo=albedo
    )

    # Create other required inputs for poa_components
    aoi = 20  # degrees
    dni = 800  # W/m²
    poa_sky_diffuse = 200  # W/m²

    # Use poa_components with the calculated ground_diffuse
    result = pvlib.irradiance.poa_components(
        aoi=aoi,
        dni=dni,
        poa_sky_diffuse=poa_sky_diffuse,
        poa_ground_diffuse=ground_diffuse
    )

    # Print results
    print("\nTest Results:")
    print("=" * 50)
    print(f"Ground Diffuse (from get_ground_diffuse): {ground_diffuse:.2f} W/m²")
    print("\nPOA Components:")
    for key, value in result.items():
        print(f"{key}: {value:.2f} W/m²")

    # Verify the results make sense
    assert ground_diffuse > 0, "Ground diffuse should be positive"
    assert result['poa_ground_diffuse'] == ground_diffuse, "Ground diffuse should match input"
    assert result['poa_global'] > 0, "Total POA should be positive"
    assert result['poa_global'] == result['poa_direct'] + result['poa_diffuse'], "Total should equal sum of components"
    assert result['poa_diffuse'] == result['poa_sky_diffuse'] + result[
        'poa_ground_diffuse'], "Diffuse should equal sum of sky and ground"

if __name__ == "__main__":
    test_ground_diffuse_integration()
