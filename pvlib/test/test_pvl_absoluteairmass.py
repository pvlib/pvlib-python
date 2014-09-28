
import unittest
from ..clearsky import absoluteairmass 
from nose.tools import *
import numpy as np



def test_nan():
 	absoluteairmass(AMrelative=np.nan,Pressure=1)


def test_non_array():
	absoluteairmass(AMrelative=1,Pressure=1)

@raises(Exception)
def test_numeric():
	absoluteairmass(AMrelative='g',Pressure=1)

@raises(Exception)
def test_numeric_2():
	absoluteairmass(AMrelative=1,Pressure='g')

def test_logical_2():
	absoluteairmass(AMrelative=-1,Pressure=1)
  
    	

def main():
    unittest.main()

if __name__ == '__main__':
    main()