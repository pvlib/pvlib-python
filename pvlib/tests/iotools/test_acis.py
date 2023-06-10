"""
tests for :mod:`pvlib.iotools.acis`
"""

import pandas as pd
import pytest
from pvlib.iotools import get_acis_precipitation
from ..conftest import (RERUNS, RERUNS_DELAY, assert_series_equal)
from requests import HTTPError


@pytest.mark.parametrize('dataset,expected,lat,lon', [
    (1, [0.76, 1.78, 1.52, 0.76, 0.0], 40.0, -80.0),
    (2, [0.05, 2.74, 1.43, 0.92, 0.0], 40.0083, -79.9653),
    (3, [0.0, 2.79, 1.52, 1.02, 0.0], 40.0, -80.0),
    (21, [0.6, 1.8, 1.9, 1.2, 0.0], 40.0, -80.0),
])
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_acis_precipitation(dataset, expected, lat, lon):
    st = '2012-01-01'
    ed = '2012-01-05'
    precipitation, meta = get_acis_precipitation(40, -80, st, ed, dataset)
    idx = pd.date_range(st, ed, freq='d')
    idx.name = 'date'
    idx.freq = None
    expected = pd.Series(expected, index=idx, name='precipitation')
    assert_series_equal(precipitation, expected)
    assert meta == {'latitude': lat, 'longitude': lon}


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_acis_precipitation_error():
    with pytest.raises(HTTPError, match='invalid grid'):
        # 50 is not a valid dataset (or "grid", in ACIS lingo)
        get_acis_precipitation(40, -80, '2012-01-01', '2012-01-05', 50)
