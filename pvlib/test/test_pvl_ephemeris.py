
from nose.tools import *
import numpy as np
import pandas as pd 
import sys
import os


from ..solarposition import ephemeris
from .. import tmy

def test_inputs():
	TMY,meta=tmy.readtmy3(filename='data/703165TY.csv')

	DFout=ephemeris(Time=TMY.index,Location=meta)
	assert(1)
def main():
    unittest.main()

if __name__ == '__main__':
    main()

    