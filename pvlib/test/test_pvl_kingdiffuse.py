from nose.tools import *
import numpy as np
import pandas as pd 
from .. import pvl_kingdiffuse

def test_proper():
	king=pvl_kingdiffuse(40,pd.DataFrame(range(400)),pd.DataFrame(range(400)),pd.DataFrame(range(400))/40)
	assert(1)
	
def test_proper_salar():
	king=pvl_kingdiffuse(40,400,400,30)
	assert(1)


def main():
    unittest.main()

if __name__ == '__main__':
    main()