"""
tests for :mod:`pvlib.iotools.merra2`
"""

import pandas as pd
import numpy as np
import pytest
import os
from pvlib.iotools import read_merra2, get_merra2
from ..conftest import (DATA_DIR, RERUNS, RERUNS_DELAY, assert_index_equal,
                        requires_merra2_credentials, requires_xarray)


@pytest.fixture(scope="module")
def merra2_credentials():
    """Supplies pvlib-python's EarthData login credentials.
    Users should obtain their own credentials as described in the `get_merra2`
    documentation."""
    return (os.environ["MERRA2_USERNAME"], os.environ["MERRA2_PASSWORD"])


@requires_xarray
def test_read_merra2():
    #  data, meta = \
    #    read_merra2(DATA_DIR / 'MERRA2_400.tavg1_2d_rad_Nx.20200101.nc4')
    assert True


@requires_xarray
@requires_merra2_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2():
    assert True
