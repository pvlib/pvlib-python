''' 
 PVL_GROUNDDIFFUSE Estimate diffuse irradiance from ground reflections given irradiance, albedo, and surface tilt 

 Syntax
   GR = pvl_grounddiffuse(SurfTilt, GHI, Albedo)

 Description
   Function to determine the portion of irradiance on a tilted surface due
   to ground reflections. Any of the inputs may be DataFrames or scalars.

 Inputs
   SurfTilt - a scalar or DataFrame of surface tilt angles in decimal degrees. 
     SurfTilt must be >=0 and <=180. The tilt angle is defined as
     degrees from horizontal (e.g. surface facing up = 0, surface facing
     horizon = 90).
   GHI - a scalar or DataFrame of global horizontal irradiance in W/m^2.  
     GHI must be >=0.
   Albedo - a scalar or DataFrame for ground reflectance, typically 0.1-0.4 for
     surfaces on Earth (land), may increase over snow, ice, etc. May also 
     be known as the reflection coefficient. Must be >=0 and <=1.

 Outputs
   GR is a scalar or DataFrame of ground reflected irradiances in W/m^2. 
   The DataFrame has the same number of elements as the input DataFrame(s). 

 References
   [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
   solar irradiance on inclined surfaces for building energy simulation"
   2007, Solar Energy vol. 81. pp. 254-267

 See also PVL_DISC    PVL_PEREZ    PVL_REINDL1990    PVL_KLUCHER1979
 PVL_HAYDAVIES1980    PVL_ISOTROPICSKY    PVL_KINGDIFFUSE

'''
import numpy as np
import pvl_tools 
import pandas as pd

def pvl_grounddiffuse(SurfTilt,GHI,Albedo):

    Expect={'SurfTilt':('num'),
            'GHI':('x>=0'),
            'Albedo':('num','array','x>=0','x<=1'),
            }

    var=pvl_tools.Parse(kwargs,Expect)

    GR=var.GHI*(var.Albedo)*((1 - np.cos(np.radians(var.SurfTilt)))*(0.5))

    
    return pd.DataFrame({'GR':GR})
