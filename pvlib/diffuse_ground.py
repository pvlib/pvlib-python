
import numpy as np
import pvl_tools 
import pandas as pd

def pvl_grounddiffuse(SurfTilt,GHI,Albedo):
  ''' 
  Estimate diffuse irradiance from ground reflections given irradiance, albedo, and surface tilt 

  Function to determine the portion of irradiance on a tilted surface due
  to ground reflections. Any of the inputs may be DataFrames or scalars.

  Parameters
  ----------
  SurfTilt : float or DataFrame 
            Surface tilt angles in decimal degrees. 
           SurfTilt must be >=0 and <=180. The tilt angle is defined as
           degrees from horizontal (e.g. surface facing up = 0, surface facing
           horizon = 90).

  GHI : float or DataFrame 
          Global horizontal irradiance in W/m^2.  
          GHI must be >=0.

  Albedo : float or DataFrame 
          Ground reflectance, typically 0.1-0.4 for
          surfaces on Earth (land), may increase over snow, ice, etc. May also 
          be known as the reflection coefficient. Must be >=0 and <=1.

  Returns
  -------

  GR : float or DataFrame  
          Ground reflected irradiances in W/m^2. 
  

  References
  ----------

  [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
  solar irradiance on inclined surfaces for building energy simulation"
  2007, Solar Energy vol. 81. pp. 254-267

  See Also
  --------

  pvl_disc
  pvl_perez
  pvl_reindl1990
  pvl_klucher1979
  pvl_haydavies1980
  pvl_isotropicsky
  pvl_kingdiffuse

  '''

  Vars=locals()
  Expect={'SurfTilt':('num'),
          'GHI':('x>=0'),
          'Albedo':('num','array','x>=0','x<=1'),
          }

  var=pvl_tools.Parse(Vars,Expect)

  GR=var.GHI*(var.Albedo)*((1 - np.cos(np.radians(var.SurfTilt)))*(0.5))


  return pd.DataFrame({'GR':GR})
