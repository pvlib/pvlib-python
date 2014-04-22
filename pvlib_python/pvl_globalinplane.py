'''
 PVL_GLOBALINPLANE Determine the three components on in-plane irradiance

 Syntax
   DataFrame['E','Eb','Ediff'] = pvl_globalinplane(SurfTilt, SurfAz, AOI, DNI, In_Plane_SkyDiffuse,GR)
   
 Description
    Combines in-plane irradaince compoents from the chosen diffuse translation, ground 
    reflection and beam irradiance algorithms into the total in-plane irradiance.    

 Inputs:
   SurfTilt - a scalar or DataFrame of surface tilt angles in decimal degrees.
     SurfTilt must be >=0 and <=180. The tilt angle is defined as
     degrees from horizontal (e.g. surface facing up = 0, surface facing
     horizon = 90)
   SurfAz - A scalar or DataFrame of the surface azimuth angles in decimal degrees.
     SurfAz must be >=0 and <=360. The Azimuth convention is defined
     as degrees east of north (e.g. North = 0, south=180, East = 90, West = 270).
   AOI - a scalar or DataFrame of angle of incedence of solar rays with respect
      to the module surface, from PVL_GETAOI. AOI must be >=0 and <=180.
   DNI - a scalar or DataFrame of direct normal irradiance (W/m^2), as measured 
     from a TMY file or calculated with a clearsky model. 
    In_Plane_SkyDiffuse- a scalar or DataFrame of diffuse irradiance (W/m^2), as
      calculated by a diffuse irradiance translation function
    GR- a scalar or DataFrame of ground reflected irradiance (W/m^2), as calculated
      by a albedo model (eg. PVL_GROUNDDIFFUSE)

 Output:
    E - Scalar or DataFrame of total in-plane irradiance (W/m^2)
    Eb - Scalar or DataFrame of total in-plane beam irradiance (W/m^2)
    Ediff - Scalar or DataFrame of total in-plane diffuse irradiance (W/m^2)

 References


 See also  PVL_GROUNDDIFFUSE, PVL_GETAOI, PVL_PEREZ,    PVL_REINDL1990    PVL_KLUCHER1979
 PVL_HAYDAVIES1980    PVL_ISOTROPICSKY    PVL_KINGDIFFUSE

'''
import numpy as np
import pandas as pd
import pvl_tools

def pvl_globalinplane(**kwargs):
    Expect={'SurfTilt':('num','x>=0'),
        'SurfAz':('num','x>=-180','x<=180'),
        'AOI':('x>=0'),
        'DNI':('x>=0'),
        'In_Plane_SkyDiffuse':('x>=0'),
        'GR':('x>=0'),
        }

    var=pvl_tools.Parse(kwargs,Expect)

    Eb = var.DNI*np.cos(np.radians(var.AOI)) #Implies that AOI is relative to normal CHECK

    E = Eb + var.In_Plane_SkyDiffuse + var.GR
    Ediff = var.In_Plane_SkyDiffuse + var.GR


    
    return E, Eb, Ediff

