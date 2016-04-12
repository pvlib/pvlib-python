import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import numpy as np
from numpy import nan
import pandas as pd

from nose.tools import raises, assert_almost_equals
from nose.plugins.skip import SkipTest
from pandas.util.testing import assert_frame_equal

from pvlib.location import Location
from pvlib import solarposition
from pvlib import tracking


def test_solar_noon():
    apparent_zenith = pd.Series([10])
    apparent_azimuth = pd.Series([180])
    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=0, axis_azimuth=0,
                                       max_angle=90, backtrack=True, 
                                       gcr=2.0/7.0)
    
    expect = pd.DataFrame({'aoi': 10, 'surface_azimuth': 90,
                           'surface_tilt': 0, 'tracker_theta': 0},
                           index=[0], dtype=np.float64)
    
    assert_frame_equal(expect, tracker_data)


def test_azimuth_north_south():
    apparent_zenith = pd.Series([60])
    apparent_azimuth = pd.Series([90])

    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=0, axis_azimuth=180,
                                       max_angle=90, backtrack=True, 
                                       gcr=2.0/7.0)
    
    expect = pd.DataFrame({'aoi': 0, 'surface_azimuth': 90,
                           'surface_tilt': 60, 'tracker_theta': -60},
                           index=[0], dtype=np.float64)
                           
    assert_frame_equal(expect, tracker_data)
    
    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=0, axis_azimuth=0,
                                       max_angle=90, backtrack=True, 
                                       gcr=2.0/7.0)
    
    expect['tracker_theta'] *= -1
                           
    assert_frame_equal(expect, tracker_data)


def test_max_angle():
    apparent_zenith = pd.Series([60])
    apparent_azimuth = pd.Series([90])
    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=0, axis_azimuth=0,
                                       max_angle=45, backtrack=True, 
                                       gcr=2.0/7.0)
    
    expect = pd.DataFrame({'aoi': 15, 'surface_azimuth': 90,
                           'surface_tilt': 45, 'tracker_theta': 45},
                           index=[0], dtype=np.float64)
    
    assert_frame_equal(expect, tracker_data)


def test_backtrack():
    apparent_zenith = pd.Series([80])
    apparent_azimuth = pd.Series([90])

    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=0, axis_azimuth=0,
                                       max_angle=90, backtrack=False, 
                                       gcr=2.0/7.0)
    
    expect = pd.DataFrame({'aoi': 0, 'surface_azimuth': 90,
                           'surface_tilt': 80, 'tracker_theta': 80},
                           index=[0], dtype=np.float64)
    
    assert_frame_equal(expect, tracker_data)
    
    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=0, axis_azimuth=0,
                                       max_angle=90, backtrack=True, 
                                       gcr=2.0/7.0)
    
    expect = pd.DataFrame({'aoi': 52.5716, 'surface_azimuth': 90,
                           'surface_tilt': 27.42833, 'tracker_theta': 27.4283},
                           index=[0], dtype=np.float64)
    
    assert_frame_equal(expect, tracker_data)


def test_axis_tilt():
    apparent_zenith = pd.Series([30])
    apparent_azimuth = pd.Series([135])

    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=30, axis_azimuth=180,
                                       max_angle=90, backtrack=True, 
                                       gcr=2.0/7.0)
    
    expect = pd.DataFrame({'aoi': 7.286245, 'surface_azimuth': 142.65730,
                           'surface_tilt': 35.98741, 'tracker_theta': -20.88121},
                           index=[0], dtype=np.float64)
    
    assert_frame_equal(expect, tracker_data)
    
    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=30, axis_azimuth=0,
                                       max_angle=90, backtrack=True, 
                                       gcr=2.0/7.0)
    
    expect = pd.DataFrame({'aoi': 47.6632, 'surface_azimuth': 50.96969,
                           'surface_tilt': 42.5152, 'tracker_theta': 31.6655},
                           index=[0], dtype=np.float64)
    
    assert_frame_equal(expect, tracker_data)


def test_axis_azimuth():
    apparent_zenith = pd.Series([30])
    apparent_azimuth = pd.Series([90])

    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=0, axis_azimuth=90,
                                       max_angle=90, backtrack=True, 
                                       gcr=2.0/7.0)
    
    expect = pd.DataFrame({'aoi': 30, 'surface_azimuth': 180,
                           'surface_tilt': 0, 'tracker_theta': 0},
                           index=[0], dtype=np.float64)
    
    assert_frame_equal(expect, tracker_data)
    
    apparent_zenith = pd.Series([30])
    apparent_azimuth = pd.Series([180])
    
    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=0, axis_azimuth=90,
                                       max_angle=90, backtrack=True, 
                                       gcr=2.0/7.0)
    
    expect = pd.DataFrame({'aoi': 0, 'surface_azimuth': 180,
                           'surface_tilt': 30, 'tracker_theta': 30},
                           index=[0], dtype=np.float64)
    
    assert_frame_equal(expect, tracker_data)


@raises(ValueError)
def test_index_mismatch():
    apparent_zenith = pd.Series([30])
    apparent_azimuth = pd.Series([90,180])

    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=0, axis_azimuth=90,
                                       max_angle=90, backtrack=True, 
                                       gcr=2.0/7.0)


def test_SingleAxisTracker_creation():
    system = tracking.SingleAxisTracker(max_angle=45,
                                        gcr=.25,
                                        module='blah',
                                        inverter='blarg')

    assert system.max_angle == 45
    assert system.gcr == .25
    assert system.module == 'blah'
    assert system.inverter == 'blarg'


def test_SingleAxisTracker_tracking():
    system = tracking.SingleAxisTracker(max_angle=90, axis_tilt=30,
                                        axis_azimuth=180, gcr=2.0/7.0,
                                        backtrack=True)

    apparent_zenith = pd.Series([30])
    apparent_azimuth = pd.Series([135])

    tracker_data = system.singleaxis(apparent_zenith, apparent_azimuth)
    
    expect = pd.DataFrame({'aoi': 7.286245, 'surface_azimuth': 142.65730 ,
                           'surface_tilt': 35.98741, 'tracker_theta': -20.88121},
                           index=[0], dtype=np.float64)
    
    assert_frame_equal(expect, tracker_data)

    ### results calculated using PVsyst
    pvsyst_solar_azimuth = 7.1609
    pvsyst_solar_height = 27.315
    pvsyst_axis_tilt = 20.
    pvsyst_axis_azimuth = 20.
    pvsyst_system = tracking.SingleAxisTracker(max_angle=60.,
                                               axis_tilt=pvsyst_axis_tilt,
                                               axis_azimuth=180+pvsyst_axis_azimuth,
                                               backtrack=False)
    # the definition of azimuth is different from PYsyst
    apparent_azimuth = pd.Series([180+pvsyst_solar_azimuth])
    apparent_zenith = pd.Series([90-pvsyst_solar_height])
    tracker_data = pvsyst_system.singleaxis(apparent_zenith, apparent_azimuth)
    expect = pd.DataFrame({'aoi': 41.07852 , 'surface_azimuth': 180-18.432 ,
                           'surface_tilt': 24.92122 , 'tracker_theta': -15.18391},
                           index=[0], dtype=np.float64)

    assert_frame_equal(expect, tracker_data)



def test_LocalizedSingleAxisTracker_creation():
    localized_system = tracking.LocalizedSingleAxisTracker(latitude=32,
                                                           longitude=-111,
                                                           module='blah',
                                                           inverter='blarg')

    assert localized_system.module == 'blah'
    assert localized_system.inverter == 'blarg'
    assert localized_system.latitude == 32
    assert localized_system.longitude == -111


def test_SingleAxisTracker_localize():
    system = tracking.SingleAxisTracker(max_angle=45, gcr=.25,
                                        module='blah', inverter='blarg')

    localized_system = system.localize(latitude=32, longitude=-111)

    assert localized_system.module == 'blah'
    assert localized_system.inverter == 'blarg'
    assert localized_system.latitude == 32
    assert localized_system.longitude == -111


def test_SingleAxisTracker_localize_location():
    system = tracking.SingleAxisTracker(max_angle=45, gcr=.25,
                                        module='blah', inverter='blarg')
    location = Location(latitude=32, longitude=-111)
    localized_system = system.localize(location=location)

    assert localized_system.module == 'blah'
    assert localized_system.inverter == 'blarg'
    assert localized_system.latitude == 32
    assert localized_system.longitude == -111


def test_get_irradiance():
    system = tracking.SingleAxisTracker(max_angle=90, axis_tilt=30,
                                        axis_azimuth=180, gcr=2.0/7.0,
                                        backtrack=True)
    times = pd.DatetimeIndex(start='20160101 1200-0700',
                             end='20160101 1800-0700', freq='6H')
    location = Location(latitude=32, longitude=-111)
    solar_position = location.get_solarposition(times)
    irrads = pd.DataFrame({'dni':[900,0], 'ghi':[600,0], 'dhi':[100,0]},
                          index=times)
    solar_zenith = solar_position['apparent_zenith']
    solar_azimuth = solar_position['azimuth']
    tracker_data = system.singleaxis(solar_zenith, solar_azimuth)
                          
    irradiance = system.get_irradiance(irrads['dni'],
                                       irrads['ghi'],
                                       irrads['dhi'],
                                       solar_zenith=solar_zenith,
                                       solar_azimuth=solar_azimuth,
                                       surface_tilt=tracker_data['surface_tilt'],
                                       surface_azimuth=tracker_data['surface_azimuth'])
    
    expected = pd.DataFrame(data=np.array(
        [[ 961.80070,   815.94490,   145.85580,   135.32820,
          10.52757492],
       [          nan,           nan,           nan,           nan,
                  nan]]),
                            columns=['poa_global', 'poa_direct',
                                     'poa_diffuse', 'poa_sky_diffuse',
                                     'poa_ground_diffuse'],
                            index=times)

    irradiance = np.round(irradiance, 4)
    expected = np.round(expected, 4)
    assert_frame_equal(irradiance, expected)
