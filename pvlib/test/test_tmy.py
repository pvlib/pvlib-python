import logging
pvl_logger = logging.getLogger('pvlib')

import inspect
import os

test_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

from pvlib import tmy


def test_readtmy3():
    tmy.readtmy3(os.path.join(test_dir, '../data/703165TY.csv'))
    
def test_readtmy2():
    tmy.readtmy2(os.path.join(test_dir, '../data/12839.tm2'))
    
