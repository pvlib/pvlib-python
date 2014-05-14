
from nose.tools import *
import numpy as np
import pandas as pd 
from .. import pvl_readtmy2

def test_proper():
	fname='/Users/robandrews/Dropbox/My_Documents/Documents/Projects/Data/TMY/tm2/03945.tm2'
	TMY2,TMY2meta=pvl_readtmy2.pvl_readtmy2(FileName=fname)
	assert(max(TMY2.RHum<=100))

	fname='/Users/robandrews/Dropbox/My_Documents/Documents/Projects/Data/TMY/tm2/12839.tm2'
	TMY2,TMY2meta=pvl_readtmy2.pvl_readtmy2(FileName=fname)
	assert(max(TMY2.RHum<=100))
	assert(TMY2meta['Latitude']>0)
	assert(TMY2meta['Longitude']<0)
	assert(TMY2meta['City']=='MIAMI')
def main():
    unittest.main()

if __name__ == '__main__':
    main()