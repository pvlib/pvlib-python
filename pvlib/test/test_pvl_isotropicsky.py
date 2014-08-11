from nose.tools import *
import numpy as np
import pandas as pd 
from .. import pvl_isotropicsky

def test_proper():
	DHI=pvl_isotropicsky(70,pd.DataFrame(range(400)))	
	assert(1)
	
def test_proper_salar():
	DHI=pvl_isotropicsky(70,400)	
	assert(1)


def main():
    unittest.main()

if __name__ == '__main__':
    main()