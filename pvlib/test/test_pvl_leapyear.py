
from nose.tools import *
import numpy as np
import pandas as pd 
from .. import pvl_leapyear

def test_proper():
	LY=pvl_leapyear.pvl_leapyear(Year=2012)
	assert(LY==1)

	LY=pvl_leapyear.pvl_leapyear(Year=2013)
	assert(LY==0)

	LY=pvl_leapyear.pvl_leapyear(Year=2016)
	assert(LY==1)

def main():
    unittest.main()

if __name__ == '__main__':
    main()