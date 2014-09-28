
from nose.tools import *
import numpy as np
import pandas as pd 
from ..pvsystem import makelocationstruct

def test_proper():
	Loc=makelocationstruct(latitude=46, longitude=-100, TZ=-5)
	assert(Loc.latitude==46)

	Loc=makelocationstruct(latitude=46, longitude=-100,TZ=-5)
	assert(Loc.longitude==-100)

	Loc=makelocationstruct(latitude=46, longitude=-100, TZ=-5)
	assert(Loc.altitude==100)

	Loc=makelocationstruct(latitude=46, longitude=-100, altitude=14,TZ=-5)
	assert(Loc.altitude==14)

def main():
    unittest.main()

if __name__ == '__main__':
    main()