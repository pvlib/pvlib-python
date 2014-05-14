from nose.tools import *
import numpy as np
import pandas as pd 
from .. import pvl_tools
from .. import pvl_readtmy3 as tmy3
from .. import pvl_ephemeris as eph
from .. import pvl_extraradiation as ext
from .. import pvl_relativeairmass as AM
from .. import pvl_haydavies1980 as hd
import os
def test():
	
	TMY, meta=tmy3.pvl_readtmy3(FileName=os.path.abspath('')+'/723650TY.csv')
	meta.SurfTilt=30

	meta.SurfAz=0
	meta.Albedo=0.2 

	TMY['SunAz'], TMY['SunEl'], TMY['ApparentSunEl'], TMY['SolarTime'], TMY['SunZen']=eph.pvl_ephemeris(Time=TMY.index,Location=meta)

	TMY['HExtra']=ext.pvl_extraradiation(doy=TMY.index.dayofyear)

	TMY['AM']=AM.pvl_relativeairmass(z=TMY.SunZen)

	TMY['In_Plane_SkyDiffuse']=hd.pvl_haydavies1980(SurfTilt=meta.SurfTilt,
	                                        SurfAz=meta.SurfAz,
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



	diff=hd.pvl_haydavies1980(SurfTilt=SurfTilt,
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