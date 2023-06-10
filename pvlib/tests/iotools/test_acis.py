"""
tests for :mod:`pvlib.iotools.acis`
"""

import pandas as pd
import pytest
from pvlib.iotools import get_acis_precipitation
from ..conftest import (RERUNS, RERUNS_DELAY, assert_series_equal)
from requests import HTTPError

@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_acis_precipitation():
    precipitation, meta = get_acis_precipitation(40, -80, '2012-01-01',
                                                 '2012-01-05', 21)
    idx = pd.date_range('2012-01-01', '2012-01-05', freq='d')
    expected = pd.Series([0.6, 1.8, 1.9, 1.2, 0.0], index=idx,
                         name='precipitation')
    assert_series_equal(precipitation, expected)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_acis_precipitation_error():
    with pytest.raises(HTTPError, match='invalid grid'):
        # 50 is not a valid dataset (or "grid", in ACIS lingo)
        get_acis_precipitation(40, -80, '2012-01-01', '2012-01-05', 50)
