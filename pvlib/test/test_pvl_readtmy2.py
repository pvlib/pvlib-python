
from nose.tools import *
import numpy as np
import pandas as pd 

from .. import tmy

def test_proper():
	fname='12839.tm2'
	TMY2,TMY2meta=tmy.readtmy2(filename=fname)
	assert(max(TMY2.RHum<=100))

	fname='12839.tm2'
	TMY2,TMY2meta=tmy.readtmy2(filename=fname)
	assert(max(TMY2.RHum<=100))
	assert(TMY2meta['latitude']>0)
	assert(TMY2meta['longitude']<0)
	assert(TMY2meta['City']=='MIAMI')
def main():
    unittest.main()

if __name__ == '__main__':
    main()