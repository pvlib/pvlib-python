import pytest
import numpy as np
from pvlib.iotools import download_SRTM
from ..conftest import DATA_DIR

from pvlib._deprecation import pvlibDeprecationWarning

NM_lat, NM_lon = 35, -107
NM_elevations = [1418, 2647, 3351]
NM_points = ((1000, 2000, 3000),
             (1000, 2000, 3000))


@pytest.mark.remote_data
def test_cgiar_download():
    DEM, file_path = download_SRTM(35, -107, path_to_save=DATA_DIR)
    assert np.allclose(DEM[NM_points], NM_elevations)
