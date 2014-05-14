


import numpy as np
import pvl_tools as pvt

def pvl_absoluteairmass(AMrelative,Pressure):
	'''
	Determine absolute (pressure corrected) airmass from relative airmass and pressure

	Gives the airmass for locations not at sea-level (i.e. not at standard
	pressure). The input argument "AMrelative" is the relative airmass. The
	input argument "pressure" is the pressure (in Pascals) at the location
	of interest and must be greater than 0. The calculation for
	absolute airmass is:
	absolute airmass = (relative airmass)*pressure/101325

	Parameters
	----------

	AMrelative : float or DataFrame
	
				The airmass at sea-level which can be calculated using the 
				PV_LIB function pvl_relativeairmass. 
	
	pressure : float or DataFrame

				a scalar or vector of values providing the site pressure in
				Pascal. If pressure is a vector it must be of the same size as all
				other vector inputs. pressure must be >=0. Pressure may be measured
				or an average pressure may be calculated from site altitude.

	Returns
	-------

	AMa : float or DataFrame

				Absolute (pressure corrected) airmass

	References
	----------

	[1] C. Gueymard, "Critical analysis and performance assessment of 
	clear sky solar irradiance models using theoretical and measured data,"
	Solar Energy, vol. 51, pp. 121-138, 1993.

	See also 
	---------
	pvl_relativeairmass 

	'''
	
	Vars=locals()
	Expect={'AMrelative': ('array','num'),
			'Pressure': ('array', 'num', 'x>0')}

	var=pvt.Parse(Vars,Expect)
	
	AMa=var.AMrelative.dot(var.Pressure) / 101325
	
	return AMa
