from nose.tools import *
import numpy as np
import pandas as pd 
from .. import pvl_tools
from ..solarposition import ephemeris 
from ..irradiance import extraradiation,haydavies1980  
from ..clearsky import relativeairmass 
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

	TMY['In_Plane_SkyDiffuse']=haydavies1980(SurfTilt=meta['SurfTilt'],
	                                        SurfAz=meta['SurfAz'],
	                                        DHI=TMY.DHI,
	                                        DNI=TMY.DNI,
	                                        HExtra=TMY.HExtra,
	                                        SunZen=TMY.SunZen,
	                                        SunAz=TMY.SunAz)

	assert(np.size(TMY['In_Plane_SkyDiffuse'])==np.size(TMY['SunZen']))

def test_scalar():
	
	SurfTilt=30
	SurfAz=0
	Albedo=0.2 



	diff=haydavies1980(SurfTilt=SurfTilt,
	                                        SurfAz=SurfAz,
	                                        DHI=500,
	                                        DNI=400,
	                                        HExtra=1340,
	                                        SunZen=30,
	                                        SunAz=0)

	assert(diff==diff)


def main():
    unittest.main()

if __name__ == '__main__':
    main()