import numpy as np
import pvl_tools

def pvl_kingdiffuse(SurfTilt,DHI,GHI,SunZen):
	'''
	Determine diffuse irradiance from the sky on a tilted surface using the King model

	King's model determines the diffuse irradiance from the sky
	(ground reflected irradiance is not included in this algorithm) on a
	tilted surface using the surface tilt angle, diffuse horizontal
	irradiance, global horizontal irradiance, and sun zenith angle. Note
	that this model is not well documented and has not been published in
	any fashion (as of January 2012).

	Parameters
	----------

	SurfTilt : float or DataFrame
	      Surface tilt angles in decimal degrees.
	      SurfTilt must be >=0 and <=180. The tilt angle is defined as
	      degrees from horizontal (e.g. surface facing up = 0, surface facing
	      horizon = 90)
	DHI : float or DataFrame
	      diffuse horizontal irradiance in W/m^2. 
	      DHI must be >=0.
	GHI : float or DataFrame
	      global horizontal irradiance in W/m^2. 
	      DHI must be >=0.

	SunZen : float or DataFrame
	      apparent (refraction-corrected) zenith
	      angles in decimal degrees. 
	      SunZen must be >=0 and <=180.

	Returns
	--------

	SkyDiffuse : float or DataFrame

			the diffuse component of the solar radiation  on an
			arbitrarily tilted surface as given by a model developed by David L.
			King at Sandia National Laboratories. 


	See Also
	--------

	pvl_ephemeris   
	pvl_extraradiation   
	pvl_isotropicsky
	pvl_haydavies1980   
	pvl_perez 
	pvl_klucher1979   
	pvl_reindl1990

	'''
	Vars=locals()
	Expect={'SurfTilt':('num','x>=0'),
	      'SunZen':('x>=-180'),
	      'DHI':('x>=0'),
	      'GHI':('x>=0')
	      }

	var=pvl_tools.Parse(Vars,Expect)

	SkyDiffuse=DHI*((1 + pvl_tools.cosd(SurfTilt))) / 2 + GHI*((0.012 * SunZen - 0.04))*((1 - pvl_tools.cosd(SurfTilt))) / 2

	return SkyDiffuse
