
import unittest
from .. import pvl_absoluteairmass 
from nose.tools import *
import numpy as np

reload(pvl_absoluteairmass)


def test_nan():
 	pvl_absoluteairmass.pvl_absoluteairmass(AMrelative=np.nan,Pressure=1)


def test_non_array():
	pvl_absoluteairmass.pvl_absoluteairmass(AMrelative=1,Pressure=1)

@raises(Exception)
def test_numeric():
	pvl_absoluteairmass.pvl_absoluteairmass(AMrelative='g',Pressure=1)

@raises(Exception)
def test_numeric_2():
	pvl_absoluteairmass.pvl_absoluteairmass(AMrelative=1,Pressure='g')

def test_logical_2():
	pvl_absoluteairmass.pvl_absoluteairmass(AMrelative=-1,Pressure=1)
  
    	

def main():
    unittest.main()

if __name__ == '__main__':
    main()