import numpy as np
import pandas as pd
import pvl_tools

def pvl_systemdef(TMYmeta,SurfTilt, SurfAz,Albedo,SeriesModules,ParallelModules):

	'''
    Generates a dict of system paramters used throughout a simulation

    Parameters
    ----------

    TMYmeta : struct or dict
                meta file generated from a TMY file using pvl_readtmy2 or pvl_readtmy3.
                It should contain at least the following fields: 

					===============   ======  ====================  
					meta field        format  description
					===============   ======  ====================  
					meta.altitude     Float   site elevation
					meta.latitude     Float   site latitude
					meta.longitude    Float   site longitude
					meta.Name         String  site name
					meta.State        String  state
					meta.TZ           Float   timezone
					===============   ======  ====================  

	SurfTilt : float or DataFrame
	          Surface tilt angles in decimal degrees.
	          SurfTilt must be >=0 and <=180. The tilt angle is defined as
	          degrees from horizontal (e.g. surface facing up = 0, surface facing
	          horizon = 90)

	SurfAz : float or DataFrame
			Surface azimuth angles in decimal degrees.
			SurfAz must be >=0 and <=360. The Azimuth convention is defined
			as degrees east of north (e.g. North = 0, South=180 East = 90, West = 270).

	Albedo : float or DataFrame 
			Ground reflectance, typically 0.1-0.4 for
			surfaces on Earth (land), may increase over snow, ice, etc. May also 
			be known as the reflection coefficient. Must be >=0 and <=1.

	SeriesModules : float
			Number of modules connected in series in a string. 

	ParallelModules : int
			Number of strings connected in parallel.
    
    

    Returns
    -------

    Result : dict

                A dict with the following fields.
      
					* 'SurfTilt'
					* 'SurfAz'
					* 'Albedo'
					* 'SeriesModules'
					* 'ParallelModules'
					* 'Lat'
					* 'Long'
					* 'TZ'
					* 'name'
					* 'altitude'


    See also
    --------
    pvl_readtmy3
    pvl_readtmy2


    '''


	Vars=locals()
	Expect={'TMYmeta':'',
			'SurfTilt':('num','x>=0'),
			'SurfAz':('num'),
			'Albedo':('num','x>=0'),
			'SeriesModules':('default','default=1','num','x>=0'),
			'ParallelModules':('default','default=1','num','x>=0')}

	var=pvl_tools.Parse(Vars,Expect)

	system={'SurfTilt':var.SurfTilt,
			'SurfAz':var.SurfAz,
			'Albedo':var.Albedo,
			'SeriesModules':var.SeriesModules,
			'ParallelModules':var.ParallelModules,
			'Lat':var.TMYmeta.latitude,
			'Long':var.TMYmeta.longitude,
			'TZ':var.TMYmeta.TZ,
			'name':var.TMYmeta.Name,
			'altitude':var.TMYmeta.altitude}

	return pvl_tools.repack(system)

