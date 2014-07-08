
from nose.tools import *
import numpy as np
import pandas as pd 
from .. import pvl_pres2alt

def test_proper():
	alt=pvl_pres2alt(pressure=222)
	assert(alt>0)

	alt=pvl_pres2alt(pressure=[222,4434,32453,212])
	assert(np.size(alt)>1)
	#include physical checks
@raises(Exception)
def test_fail():
	alt=pvl_pres2alt(doy=-600)
	assert(alt>0)

def main():
    unittest.main()

if __name__ == '__main__':
    main()