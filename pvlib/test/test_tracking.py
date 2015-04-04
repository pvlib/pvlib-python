import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import numpy as np
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
    
    expect = pd.DataFrame({'aoi': 7.286245, 'surface_azimuth': 37.3427,
                           'surface_tilt': 35.98741, 'tracker_theta': -20.88121},
                           index=[0], dtype=np.float64)
    
    assert_frame_equal(expect, tracker_data)
    
    tracker_data = tracking.singleaxis(apparent_zenith, apparent_azimuth,
                                       axis_tilt=30, axis_azimuth=0,
                                       max_angle=90, backtrack=True, 
                                       gcr=2.0/7.0)
    
    expect = pd.DataFrame({'aoi': 47.6632, 'surface_azimuth': 129.0303,
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