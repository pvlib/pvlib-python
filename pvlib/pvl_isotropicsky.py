
 

import pvl_tools

def pvl_isotropicsky(SurfTilt,DHI):
	'''
	Determine diffuse irradiance from the sky on a tilted surface using isotropic sky model

	Hottel and Woertz's model treats the sky as a uniform source of diffuse
	irradiance. Thus the diffuse irradiance from the sky (ground reflected
	irradiance is not included in this algorithm) on a tilted surface can
	be found from the diffuse horizontal irradiance and the tilt angle of
	the surface.

	Parameters
	----------

	SurfTilt : float or DataFrame
			Surface tilt angles in decimal degrees. 
			SurfTilt must be >=0 and <=180. The tilt angle is defined as
			degrees from horizontal (e.g. surface facing up = 0, surface facing
			horizon = 90)
	
	DHI : float or DataFrame
			Diffuse horizontal irradiance in W/m^2.
			DHI must be >=0.


	Returns
	-------   

	SkyDiffuse : float of DataFrame

			The diffuse component of the solar radiation  on an
			arbitrarily tilted surface defined by the isotropic sky model as
			given in Loutzenhiser et. al (2007) equation 3.
			SkyDiffuse is the diffuse component ONLY and does not include the ground
			reflected irradiance or the irradiance due to the beam.
			SkyDiffuse is a column vector vector with a number of elements equal to
			the input vector(s).


	References
	----------

	[1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
	solar irradiance on inclined surfaces for building energy simulation"
	2007, Solar Energy vol. 81. pp. 254-267
	
	[2] Hottel, H.C., Woertz, B.B., 1942. Evaluation of flat-plate solar heat
	collector. Trans. ASME 64, 91.

	See also    
	--------

	pvl_reindl1990  
	pvl_haydavies1980  
	pvl_perez  
	pvl_klucher1979
	pvl_kingdiffuse
	'''

	Vars=locals()
	Expect={'SurfTilt':'x <= 180 & x >= 0 ',
			'DHI':'x>=0'
			}
	var=pvl_tools.Parse(Vars,Expect)

	SkyDiffuse=DHI * (1 + pvl_tools.cosd(SurfTilt)) * 0.5

	return SkyDiffuse
