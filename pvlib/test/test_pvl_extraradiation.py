
from nose.tools import *
import numpy as np
import pandas as pd 
from ..irradiance import extraradiation

def test_proper():
	etr=extraradiation(doy=5)
	assert(etr>0)

	#include physical checks
@raises(Exception)
def test_fail():
	etr=extraradiation(doy=600)
	assert(etr>0)

def main():
    unittest.main()

if __name__ == '__main__':
    main()