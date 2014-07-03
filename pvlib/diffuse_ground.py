"""
Estimate diffuse irradiance from ground reflections.
"""

import logging
pvl_logger = logging.getLogger('pvlib')

import numpy as np
import pandas as pd


SURFACE_ALBEDOS = {'urban':0.18,
                   'grass':0.20,
                   'fresh grass':0.26,
                   'soil':0.17,
                   'sand':0.40,
                   'snow':0.65,
                   'fresh snow':0.75,
                   'asphalt':0.12,
                   'concrete':0.30,
                   'aluminum':0.85,
                   'copper':0.74,
                   'fresh steel':0.35,
                   'dirty steel':0.08,
                    }


def get_diffuse_ground(surf_tilt, ghi, albedo=.25, surface_type=None):
    ''' 
    Estimate diffuse irradiance from ground reflections given 
    irradiance, albedo, and surface tilt 

    Function to determine the portion of irradiance on a tilted surface due
    to ground reflections. Any of the inputs may be DataFrames or scalars.

    Parameters
    ----------
    surf_tilt : float or DataFrame 
            Surface tilt angles in decimal degrees. 
           SurfTilt must be >=0 and <=180. The tilt angle is defined as
           degrees from horizontal (e.g. surface facing up = 0, surface facing
           horizon = 90).

    ghi : float or DataFrame 
          Global horizontal irradiance in W/m^2.  

    albedo : float or DataFrame 
          Ground reflectance, typically 0.1-0.4 for
          surfaces on Earth (land), may increase over snow, ice, etc. May also 
          be known as the reflection coefficient. Must be >=0 and <=1.
          Will be overridden if surface_type is supplied.
          
    surface_type: None or string in 
                  'urban', 'grass', 'fresh grass', 'snow', 'fresh snow',
                  'asphalt', 'concrete', 'aluminum', 'copper', 
                  'fresh steel', 'dirty steel'.
                  Overrides albedo.

    Returns
    -------

    float or DataFrame  
          Ground reflected irradiances in W/m^2. 


    References
    ----------

    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267. 
    
    The calculation is the last term of equations 3, 4, 7, 8, 10, 11, and 12.
    
    [2] albedos from: 
    http://pvpmc.org/modeling-steps/incident-irradiance/plane-of-array-poa-irradiance/calculating-poa-irradiance/poa-ground-reflected/albedo/
    
    and
    
    http://en.wikipedia.org/wiki/Albedo
    
    See Also
    --------

    pvlib.diffuse_sky

    '''
    
    if surface_type is not None:
        albedo = SURFACE_ALBEDOS[surface_type]
        pvl_logger.info('surface_type={} mapped to albedo={}'
                        .format(surface_type, albedo))

    diffuse_irrad = ghi * albedo * (1 - np.cos(np.radians(surf_tilt))) * 0.5
    
    try:
        diffuse_irrad.name = 'diffuse_ground'
    except AttributeError:
        pass
    
    return diffuse_irrad
    
    
