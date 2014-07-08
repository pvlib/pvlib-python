from nose.tools import *
import numpy as np
import pandas as pd 
from .. import pvl_tools
from .. import pvl_readtmy3 
from .. import pvl_ephemeris 
from .. import pvl_extraradiation 
from .. import pvl_relativeairmass 
from .. import pvl_reindl1990 
import os
def test():
	
	TMY, meta=pvl_readtmy3(FileName=os.path.abspath('')+'/723650TY.csv')
	
	meta['SurfTilt']=30
	meta['SurfAz']=0
	meta['Albedo']=0.2 

	TMY['SunAz'], TMY['SunEl'], TMY['ApparentSunEl'], TMY['SolarTime'], TMY['SunZen']=pvl_ephemeris(Time=TMY.index,Location=meta)

	TMY['HExtra']=pvl_extraradiation(doy=TMY.index.dayofyear)

	TMY['AM']=pvl_relativeairmass(z=TMY.SunZen)

	TMY['In_Plane_SkyDiffuse']=pvl_reindl1990(SurfTilt=meta['SurfTilt'],
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