import os

from pandas.util.testing import network

from pvlib.iotools import epw
from conftest import data_dir

epw_testfile = os.path.join(data_dir, 'NLD_Amsterdam062400_IWEC.epw')


def test_read_epw():
    epw.read_epw(epw_testfile)


@network
def test_read_epw_remote():
    url = 'https://energyplus.net/weather-download/europe_wmo_region_6/NLD//NLD_Amsterdam.062400_IWEC/NLD_Amsterdam.062400_IWEC.epw'
    epw.read_epw(url)


def test_read_epw_coerce_year():
    coerce_year = 1987
    data, meta = epw.read_epw(epw_testfile, coerce_year=coerce_year)
    assert (data.index.year == 1987).all()
