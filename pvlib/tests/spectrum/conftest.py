import numpy as np
import pytest
import pandas as pd

from ..conftest import DATA_DIR

SPECTRL2_TEST_DATA = DATA_DIR / 'spectrl2_example_spectra.csv'


@pytest.fixture
def spectrl2_data():
    # reference spectra generated with solar_utils==0.3
    """
    expected = solar_utils.spectrl2(
        units=1,
        location=[40, -80, -5],
        datetime=[2020, 3, 15, 10, 45, 59],
        weather=[1013, 15],
        orientation=[0, 180],
        atmospheric_conditions=[1.14, 0.65, 0.344, 0.1, 1.42],
        albedo=[0.3, 0.7, 0.8, 1.3, 2.5, 4.0] + [0.2]*6,
    )
    """
    kwargs = {
        'surface_tilt': 0,
        'relative_airmass': 1.4899535986910446,
        'apparent_zenith': 47.912086486816406,
        'aoi': 47.91208648681641,
        'ground_albedo': 0.2,
        'surface_pressure': 101300,
        'ozone': 0.344,
        'precipitable_water': 1.42,
        'aerosol_turbidity_500nm': 0.1,
        'dayofyear': 75
    }
    df = pd.read_csv(SPECTRL2_TEST_DATA, index_col=0)
    # convert um to nm
    df['wavelength'] = np.round(df['wavelength'] * 1000, 1)
    df[['specdif', 'specdir', 'specetr', 'specglo']] /= 1000
    return kwargs, df
