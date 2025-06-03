import pytest

from pvlib.iotools import epw
from tests.conftest import TESTS_DATA_DIR, RERUNS, RERUNS_DELAY

from pvlib._deprecation import pvlibDeprecationWarning

epw_testfile = TESTS_DATA_DIR / 'NLD_Amsterdam062400_IWEC.epw'


def test_read_epw():
    df, meta = epw.read_epw(epw_testfile)
    assert len(df) == 8760
    assert 'ghi' in df.columns
    assert meta['latitude'] == 52.3


def test_read_epw_buffer():
    with open(epw_testfile, 'r') as f:
        df, meta = epw.read_epw(f)
    assert len(df) == 8760
    assert 'ghi' in df.columns
    assert meta['latitude'] == 52.3


def test_parse_epw_deprecated():
    with pytest.warns(pvlibDeprecationWarning, match='Use read_epw instead'):
        with open(epw_testfile, 'r') as f:
            df, meta = epw.parse_epw(f)
    assert len(df) == 8760
    assert 'ghi' in df.columns
    assert meta['latitude'] == 52.3


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_read_epw_remote():
    url = 'https://energyplus-weather.s3.amazonaws.com/europe_wmo_region_6/NLD/NLD_Amsterdam.062400_IWEC/NLD_Amsterdam.062400_IWEC.epw'
    epw.read_epw(url)


def test_read_epw_coerce_year():
    coerce_year = 1987
    data, _ = epw.read_epw(epw_testfile, coerce_year=coerce_year)
    assert (data.index.year == 1987).all()
