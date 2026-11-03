import pandas as pd
import pytest

from pvlib.iotools import epw
from tests.conftest import TESTS_DATA_DIR, RERUNS, RERUNS_DELAY

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


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_read_epw_remote():
    url = 'https://energyplus-weather.s3.amazonaws.com/europe_wmo_region_6/NLD/NLD_Amsterdam.062400_IWEC/NLD_Amsterdam.062400_IWEC.epw'
    epw.read_epw(url)


def test_read_epw_coerce_year():
    coerce_year = 1987
    data, _ = epw.read_epw(epw_testfile, coerce_year=coerce_year)
    assert (data.index.year == 1987).all()


def _mock_epw_urlopen(mocker, payload):
    """Patch pvlib.iotools.epw.urlopen to serve the given bytes."""
    mock_response = mocker.MagicMock()
    mock_response.read.return_value = payload
    return mocker.patch('pvlib.iotools.epw.urlopen',
                        return_value=mock_response)


def test_read_epw_http(mocker):
    # Mock the download; the HTTP branch must parse exactly like the
    # local file entry point.
    with open(epw_testfile, 'rb') as f:
        payload = f.read()
    mock_urlopen = _mock_epw_urlopen(mocker, payload)

    url = 'https://example.com/NLD_Amsterdam062400_IWEC.epw'
    df, meta = epw.read_epw(url)

    request = mock_urlopen.call_args.args[0]
    assert request.full_url == url
    expected_df, expected_meta = epw.read_epw(epw_testfile)
    pd.testing.assert_frame_equal(df, expected_df)
    assert meta == expected_meta


def test_read_epw_http_coerce_year(mocker):
    # coerce_year must be honored on the HTTP branch as well.
    with open(epw_testfile, 'rb') as f:
        payload = f.read()
    _mock_epw_urlopen(mocker, payload)

    url = 'https://example.com/NLD_Amsterdam062400_IWEC.epw'
    df, meta = epw.read_epw(url, coerce_year=2015)

    assert (df['year'] == 2015).all()
    assert (df.index.year == 2015).all()
    expected_df, expected_meta = epw.read_epw(epw_testfile,
                                              coerce_year=2015)
    pd.testing.assert_frame_equal(df, expected_df)
    assert meta == expected_meta
