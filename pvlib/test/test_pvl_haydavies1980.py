from nose.tools import *
import numpy as np
import pandas as pd 
from .. import pvl_tools
from .. import pvl_ephemeris 
from .. import pvl_extraradiation 
from .. import pvl_relativeairmass 
from .. import pvl_haydavies1980 
from .. import tmy
import os
def test():
	
	TMY, meta=tmy.readtmy3(filename='703165TY.csv')
	
	meta['SurfTilt']=30
	meta['SurfAz']=0
	meta['Albedo']=0.2 

	TMY['SunAz'], TMY['SunEl'], TMY['ApparentSunEl'], TMY['SolarTime'], TMY['SunZen']=pvl_ephemeris(Time=TMY.index,Location=meta)

	TMY['HExtra']=pvl_extraradiation(doy=TMY.index.dayofyear)

	TMY['AM']=pvl_relativeairmass(z=TMY.SunZen)

	TMY['In_Plane_SkyDiffuse']=pvl_haydavies1980(SurfTilt=meta['SurfTilt'],
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



	diff=pvl_haydavies1980(SurfTilt=SurfTilt,
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