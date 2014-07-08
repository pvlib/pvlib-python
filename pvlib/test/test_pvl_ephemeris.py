
from nose.tools import *
import numpy as np
import pandas as pd 

from .. import pvl_ephemeris
from .. import pvl_readtmy3
from .. import pvl_makelocationstruct

def test_inputs():
	TMY,meta=pvl_readtmy3(FileName='/Users/robandrews/Dropbox/My_Documents/Documents/Projects/Data/TMY/tmy3/700260TY.csv')

	DFout=pvl_ephemeris(Time=TMY.index,Location=meta)
	assert(1)
def main():
    unittest.main()

if __name__ == '__main__':
    main()

    