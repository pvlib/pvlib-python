
from nose.tools import *
import numpy as np
import pandas as pd 
import sys
import os


from .. import pvl_ephemeris
from .. import tmy
from .. import pvl_makelocationstruct

def test_inputs():
	TMY,meta=tmy.readtmy3(filename='703165TY.csv')

	DFout=pvl_ephemeris(Time=TMY.index,Location=meta)
	assert(1)
def main():
    unittest.main()

if __name__ == '__main__':
    main()

    