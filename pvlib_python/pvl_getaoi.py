'''
 PVL_GETAOI Determine angle of incidence from surface tilt/azimuth and apparent sun zenith/azimuth 

 Syntax
   AOI = pvl_getaoi(surftilt, surfaz, sunzen, sunaz)
   
 Description
   Determine the angle of incidence in degrees between a surface and the
   beam of the sun. The surface is defined by its tilt angle from
   horizontal and its azimuth pointing angle. The sun position is defined
   by the apparent (refraction corrected)sun zenith angle and the sun 
   azimuth angle.

 Inputs:
   SurfTilt - a scalar or vector of surface tilt angles in decimal degrees.
     If SurfTilt is a vector it must be of the same size as all other vector
     inputs. SurfTilt must be >=0 and <=180. The tilt angle is defined as
     degrees from horizontal (e.g. surface facing up = 0, surface facing
     horizon = 90)
   SurfAz - A scalar or vector of the surface azimuth angles in decimal degrees.
     If SurfAz is a vector it must be of the same size as all other vector
     inputs. SurfAz must be >=0 and <=360. The Azimuth convention is defined
     as degrees east of north (e.g. North = 0, East = 90, West = 270).
   SunZen - a scalar or vector of apparent (refraction-corrected) zenith
     angles in decimal degrees. If SunZen is a vector it must be of the
     same size as all other vector inputs. SunZen must be >=0 and <=180.
   SunAz - a scalar or vector of sun azimuth angles in decimal degrees.
     If SunAz is a vector it must be of the same size as all other vector
     inputs. SunAz must be >=0 and <=360. The Azimuth convention is defined
     as degrees east of north (e.g. North = 0, East = 90, West = 270).

 Output:
   AOI - A column vector with the same number of elements as any input
     vectors, which contains the angle, in decimal degrees, between the 
     surface normal vector and the sun beam vector. 

 References
   [1] D.L. King, J.A. Kratochvil, W.E. Boyson. "Spectral and
   Angle-of-Incidence Effects on Photovoltaic Modules and Solar Irradiance
   Sensors". 26th IEEE Photovoltaic Specialists Conference. Sept. 1997.

 See also  PVL_EPHEMERIS
'''
import pandas as pd
import numpy as np
import pvl_tools
def pvl_getaoi(SurfTilt,SurfAz,SunZen,SunAz):
	Vars=locals()
	Expect={'SurfTilt':('num','x>=0'),
			'SurfAz':('num','x>=-180','x<=180'),
			'SunZen':('x>=0'),
			'SunAz':('x>=0')
	}
	
	var=pvl_tools.Parse(Vars,Expect)
	
	AOI=np.degrees(np.arccos(np.cos(np.radians(var.SunZen))*(np.cos(np.radians(var.SurfTilt))) + np.sin(np.radians(var.SurfTilt))*(np.sin(np.radians(var.SunZen)))*(np.cos(np.radians(var.SunAz) - np.radians(var.SurfAz))))) #Duffie and Beckmann 1.6.3
	

	return pd.DataFrame({'AOI':AOI})
