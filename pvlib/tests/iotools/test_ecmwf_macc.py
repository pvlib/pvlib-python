"""
tests for :mod:`pvlib.iotools.ecmwf_macc`
"""

import os
import datetime
import numpy as np
import pytest
from conftest import requires_netCDF4, DATA_DIR
from pvlib.iotools import ecmwf_macc

TESTDATA = 'aod550_tcwv_20121101_test.nc'

# for creating test data
START = END = datetime.date(2012, 11, 1)
DATAFILE = 'aod550_tcwv_20121101.nc'
RESIZE = 4
LON_BND = (0, 360.0)
LAT_BND = (90, -90)


@pytest.fixture
def expected_test_data():
    return DATA_DIR / TESTDATA


@requires_netCDF4
def test_get_nearest_indices(expected_test_data):
    """Test getting indices given latitude, longitude from ECMWF_MACC data."""
    data = ecmwf_macc.ECMWF_MACC(expected_test_data)
    ilat, ilon = data.get_nearest_indices(38, -122)
    assert ilat == 17
    assert ilon == 79


@requires_netCDF4
def test_interp_data(expected_test_data):
    """Test interpolating UTC time from ECMWF_MACC data."""
    data = ecmwf_macc.ECMWF_MACC(expected_test_data)
    test9am = data.interp_data(
        38, -122, datetime.datetime(2012, 11, 1, 9, 0, 0), 'aod550')
    assert np.isclose(test9am, data.data.variables['aod550'][2, 17, 79])
    test12pm = data.interp_data(
        38, -122, datetime.datetime(2012, 11, 1, 12, 0, 0), 'aod550')
    assert np.isclose(test12pm, data.data.variables['aod550'][3, 17, 79])
    test113301 = data.interp_data(
        38, -122, datetime.datetime(2012, 11, 1, 11, 33, 1), 'aod550')
    expected = test9am + (2 + (33 + 1 / 60) / 60) / 3 * (test12pm - test9am)
    assert np.isclose(test113301, expected)  # 0.15515305836696536


@requires_netCDF4
def test_read_ecmwf_macc(expected_test_data):
    """Test reading ECMWF_MACC data from netCDF4 file."""
    data = ecmwf_macc.read_ecmwf_macc(
        expected_test_data, 38, -122)
    expected_times = [
        1351738800, 1351749600, 1351760400, 1351771200, 1351782000, 1351792800,
        1351803600, 1351814400]
    assert np.allclose(data.index.astype(int) // 1000000000, expected_times)
    expected_aod = np.array([
        0.39531226, 0.22371339, 0.18373083, 0.15010143, 0.130809, 0.11172834,
        0.09741255, 0.0921606])
    expected_tcwv = np.array([
        26.56172238, 22.75563109, 19.37884304, 16.19186269, 13.31990346,
        11.65635338, 10.94879802, 10.55725756])
    assert np.allclose(data.aod550.values, expected_aod)
    assert np.allclose(data.tcwv.values, expected_tcwv)
    assert np.allclose(data.precipitable_water.values, expected_tcwv / 10.0)
    datetimes = (datetime.datetime(2012, 11, 1, 9, 0, 0),
                 datetime.datetime(2012, 11, 1, 12, 0, 0))
    data_9am_12pm = ecmwf_macc.read_ecmwf_macc(
        expected_test_data, 38, -122, datetimes)
    assert np.allclose(data_9am_12pm.aod550.values, expected_aod[2:4])
    assert np.allclose(data_9am_12pm.tcwv.values, expected_tcwv[2:4])


def _create_test_data(datafile=DATAFILE, testfile=TESTDATA, start=START,
                      end=END, resize=RESIZE):  # pragma: no cover
    """
    Create test data from downloaded data.

    Downloaded data from ECMWF for a single day is 3MB. This creates a subset
    of the downloaded data that is only 100kb.
    """

    import netCDF4

    if not os.path.exists(datafile):
        ecmwf_macc.get_ecmwf_macc(datafile, ("aod550", "tcwv"), start, end)

    data = netCDF4.Dataset(datafile)
    testdata = netCDF4.Dataset(testfile, 'w', format="NETCDF3_64BIT_OFFSET")

    # attributes
    testdata.Conventions = data.Conventions
    testdata.history = "intentionally blank"

    # longitude
    lon_name = 'longitude'
    lon_test = data.variables[lon_name][::resize]
    lon_size = lon_test.size
    lon = testdata.createDimension(lon_name, lon_size)
    assert not lon.isunlimited()
    assert lon_test[0] == LON_BND[0]
    assert (LON_BND[-1] - lon_test[-1]) == (LON_BND[-1] / lon_size)
    longitudes = testdata.createVariable(lon_name, "f4", (lon_name,))
    longitudes.units = data.variables[lon_name].units
    longitudes.long_name = lon_name
    longitudes[:] = lon_test

    # latitude
    lat_name = 'latitude'
    lat_test = data.variables[lat_name][::resize]
    lat = testdata.createDimension(lat_name, lat_test.size)
    assert not lat.isunlimited()
    assert lat_test[0] == LAT_BND[0]
    assert lat_test[-1] == LAT_BND[-1]
    latitudes = testdata.createVariable(lat_name, "f4", (lat_name,))
    latitudes.units = data.variables[lat_name].units
    latitudes.long_name = lat_name
    latitudes[:] = lat_test

    # time
    time_name = 'time'
    time_test = data.variables[time_name][:]
    time = testdata.createDimension(time_name, None)
    assert time.isunlimited()
    times = testdata.createVariable(time_name, 'i4', (time_name,))
    times.units = data.variables[time_name].units
    times.long_name = time_name
    times.calendar = data.variables[time_name].calendar
    times[:] = time_test

    # aod
    aod_name = 'aod550'
    aod_vars = data.variables[aod_name]
    aod_dims = (time_name, lat_name, lon_name)
    aod_fill_value = getattr(aod_vars, '_FillValue')
    aods = testdata.createVariable(
        aod_name, 'i2', aod_dims, fill_value=aod_fill_value)
    for attr in aod_vars.ncattrs():
        if attr.startswith('_'):
            continue
        setattr(aods, attr, getattr(aod_vars, attr))
    aods[:] = aod_vars[:, ::resize, ::resize]

    # tcwv
    tcwv_name = 'tcwv'
    tcwv_vars = data.variables[tcwv_name]
    tcwv_dims = (time_name, lat_name, lon_name)
    tcwv_fill_value = getattr(tcwv_vars, '_FillValue')
    tcwvs = testdata.createVariable(
        tcwv_name, 'i2', tcwv_dims, fill_value=tcwv_fill_value)
    for attr in tcwv_vars.ncattrs():
        if attr.startswith('_'):
            continue
        setattr(tcwvs, attr, getattr(tcwv_vars, attr))
    tcwvs[:] = tcwv_vars[:, ::resize, ::resize]

    data.close()
    testdata.close()
