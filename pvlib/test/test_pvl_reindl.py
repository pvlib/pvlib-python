from nose.tools import *
import numpy as np
import pandas as pd 
from ..solarposition import ephemeris 
from ..irradiance import extraradiation,reindl1990 
from ..atmosphere import relativeairmass 
from .. import tmy
import os
def test():
	
	TMY, meta=tmy.readtmy3(filename='data/703165TY.csv')
	
	meta['SurfTilt']=30
	meta['SurfAz']=0
	meta['Albedo']=0.2 

	TMY['SunAz'], TMY['SunEl'], TMY['ApparentSunEl'], TMY['SolarTime'], TMY['SunZen']=ephemeris(Time=TMY.index,Location=meta)

	TMY['HExtra']=extraradiation(doy=TMY.index.dayofyear)

	TMY['AM']=relativeairmass(z=TMY.SunZen)

	TMY['In_Plane_SkyDiffuse']=reindl1990(SurfTilt=meta['SurfTilt'],
	                                        SurfAz=meta['SurfAz'],
	                                        DHI=TMY.DHI,
	                                        DNI=TMY.DNI,
	                                        GHI=TMY.GHI,
	                                        HExtra=TMY.HExtra,
	                                        SunZen=TMY.SunZen,
	                                        SunAz=TMY.SunAz)

	assert(np.size(TMY['In_Plane_SkyDiffuse'])==np.size(TMY['SunZen']))

def test_scalar():
	
	assert(False)

def main():
    unittest.main()

if __name__ == '__main__':
    main()