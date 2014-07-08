
from nose.tools import *
import numpy as np
import pandas as pd 
from datetime import datetime

from .. import pvl_spa
from .. import pvl_readtmy3
from .. import pvl_makelocationstruct

def test_inputs():
	TMY,meta=pvl_readtmy3(FileName='/Users/robandrews/Dropbox/My_Documents/Documents/Projects/Data/TMY/tmy3/700260TY.csv')

	DFout=pvl_spa(Time=TMY.index,Location=meta)
	assert(1)

def test_physical():

	date = pd.date_range(datetime(2003,10,17,12,30,30), periods=1, freq='D')

	location={'latitude':39.742476,
              'longitude':-105.1786,
              'altitude':1830,
              'TZ':-7}
	
	g=pvl_spa(Time=date, Location=location)

	assert( (((g[0]>14.3) & (g[0]<14.4)) & ((g[1]>39) & (g[1]<40)) & ((g[2]>50.1) & (g[2]<50.2) )).any) #Physical constraints from spatester.c
	

def main():
    unittest.main()

if __name__ == '__main__':
    main()

    