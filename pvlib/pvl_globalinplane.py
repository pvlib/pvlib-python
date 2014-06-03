
import numpy as np
import pandas as pd
import pvl_tools

def pvl_globalinplane(SurfTilt,SurfAz,AOI,DNI,In_Plane_SkyDiffuse, GR):
  '''
  Determine the three components on in-plane irradiance

  Combines in-plane irradaince compoents from the chosen diffuse translation, ground 
  reflection and beam irradiance algorithms into the total in-plane irradiance.    

  Parameters
  ----------

  SurfTilt : float or DataFrame
          surface tilt angles in decimal degrees.
          SurfTilt must be >=0 and <=180. The tilt angle is defined as
          degrees from horizontal (e.g. surface facing up = 0, surface facing
          horizon = 90)

  SurfAz : float or DataFrame
          Surface azimuth angles in decimal degrees.
          SurfAz must be >=0 and <=360. The Azimuth convention is defined
          as degrees east of north (e.g. North = 0, south=180, East = 90, West = 270).

  AOI : float or DataFrame
          Angle of incedence of solar rays with respect
          to the module surface, from :py:mod:`pvl_getaoi`. AOI must be >=0 and <=180.

  DNI : float or DataFrame
          Direct normal irradiance (W/m^2), as measured 
          from a TMY file or calculated with a clearsky model. 
  
  In_Plane_SkyDiffuse :  float or DataFrame
          Diffuse irradiance (W/m^2) in the plane of the modules, as
          calculated by a diffuse irradiance translation function

  GR : float or DataFrame
          a scalar or DataFrame of ground reflected irradiance (W/m^2), as calculated
          by a albedo model (eg. :py:mod:`pvl_grounddiffuse`)

  Returns
  -------

  E : float or DataFrame
          Total in-plane irradiance (W/m^2)
  Eb : float or DataFrame 
          Total in-plane beam irradiance (W/m^2)
  Ediff : float or DataFrame
          Total in-plane diffuse irradiance (W/m^2)

  See also
  --------

  pvl_grounddiffuse
  pvl_getaoi
  pvl_perez
  pvl_reindl1990
  pvl_klucher1979
  pvl_haydavies1980
  pvl_isotropicsky
  pvl_kingdiffuse

  '''
  Vars=locals()
  Expect={'SurfTilt':('num','x>=0'),
      'SurfAz':('num','x>=-180','x<=180'),
      'AOI':('x>=0'),
      'DNI':('x>=0'),
      'In_Plane_SkyDiffuse':('x>=0'),
      'GR':('x>=0'),
      }

  var=pvl_tools.Parse(Vars,Expect)

  Eb = var.DNI*np.cos(np.radians(var.AOI)) 
  E = Eb + var.In_Plane_SkyDiffuse + var.GR
  Ediff = var.In_Plane_SkyDiffuse + var.GR



  return E, Eb, Ediff

