import logging
pvl_logger = logging.getLogger('pvlib')

import pytz
from nose.tools import raises
from pytz.exceptions import UnknownTimeZoneError

from ..location import Location

aztz = pytz.timezone('US/Arizona')

def test_location_required():
    Location(32.2, -111)
    
def test_location_all():
    Location(32.2, -111, 'US/Arizona', 700, 'Tucson')

@raises(UnknownTimeZoneError)    
def test_location_invalid_tz():
    Location(32.2, -111, 'invalid')
    
@raises(TypeError)
def test_location_invalid_tz_type():
    Location(32.2, -111, 5)
    
def test_location_pytz_tz():
    Location(32.2, -111, aztz)

def test_location_print_all():
    tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
    expected_str = 'Tucson: latitude=32.2, longitude=-111, tz=US/Arizona, altitude=700'
    assert tus.__str__() == expected_str
    
def test_location_print_pytz():
    tus = Location(32.2, -111, aztz, 700, 'Tucson')
    expected_str = 'Tucson: latitude=32.2, longitude=-111, tz=US/Arizona, altitude=700'
    assert tus.__str__() == expected_str
