
import numpy as np
import pvl_tools

def pvl_reindl1990(SurfTilt,SurfAz,DHI,DNI,GHI,HExtra,SunZen,SunAz):
  '''
  Determine diffuse irradiance from the sky on a tilted surface using Reindl's 1990 model


  Reindl's 1990 model determines the diffuse irradiance from the sky
  (ground reflected irradiance is not included in this algorithm) on a
  tilted surface using the surface tilt angle, surface azimuth angle,
  diffuse horizontal irradiance, direct normal irradiance, global
  horizontal irradiance, extraterrestrial irradiance, sun zenith angle,
  and sun azimuth angle.

  Parameters
  ----------
    
  SurfTilt : DataFrame
          Surface tilt angles in decimal degrees.
          SurfTilt must be >=0 and <=180. The tilt angle is defined as
          degrees from horizontal (e.g. surface facing up = 0, surface facing
          horizon = 90)

  SurfAz : DataFrame
          Surface azimuth angles in decimal degrees.
          SurfAz must be >=0 and <=360. The Azimuth convention is defined
          as degrees east of north (e.g. North = 0, South=180 East = 90, West = 270).

  DHI : DataFrame
          diffuse horizontal irradiance in W/m^2. 
          DHI must be >=0.

  DNI : DataFrame
          direct normal irradiance in W/m^2. 
          DNI must be >=0.

  GHI: DataFrame
          Global irradiance in W/m^2. 
          GHI must be >=0.

  HExtra : DataFrame
          extraterrestrial normal irradiance in W/m^2. 
           HExtra must be >=0.

  SunZen : DataFrame
          apparent (refraction-corrected) zenith
          angles in decimal degrees. 
          SunZen must be >=0 and <=180.

  SunAz : DataFrame
          Sun azimuth angles in decimal degrees.
          SunAz must be >=0 and <=360. The Azimuth convention is defined
          as degrees east of north (e.g. North = 0, East = 90, West = 270).

  Returns
  -------

  SkyDiffuse : DataFrame

           the diffuse component of the solar radiation  on an
           arbitrarily tilted surface defined by the Reindl model as given in
           Loutzenhiser et. al (2007) equation 8.
           SkyDiffuse is the diffuse component ONLY and does not include the ground
           reflected irradiance or the irradiance due to the beam.
           SkyDiffuse is a column vector vector with a number of elements equal to
           the input vector(s).


  Notes
  -----
  
     The POAskydiffuse calculation is generated from the Loutzenhiser et al.
     (2007) paper, equation 8. Note that I have removed the beam and ground
     reflectance portion of the equation and this generates ONLY the diffuse
     radiation from the sky and circumsolar, so the form of the equation
     varies slightly from equation 8.
  
  References
  ----------

  [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
  solar irradiance on inclined surfaces for building energy simulation"
  2007, Solar Energy vol. 81. pp. 254-267

  [2] Reindl, D.T., Beckmann, W.A., Duffie, J.A., 1990a. Diffuse fraction
  correlations. Solar Energy 45(1), 1-7.

  [3] Reindl, D.T., Beckmann, W.A., Duffie, J.A., 1990b. Evaluation of hourly
  tilted surface radiation models. Solar Energy 45(1), 9-17.

  See Also 
  ---------
  pvl_ephemeris   
  pvl_extraradiation   
  pvl_isotropicsky
  pvl_haydavies1980   
  pvl_perez 
  pvl_klucher1979   
  pvl_kingdiffuse

  '''          
  Vars=locals()
  Expect={'SurfTilt':('num','x>=0'),
      'SurfAz':('num','x>=-180'),
      'DHI':('num','x>=0'),
      'DNI':('num','x>=0'),
      'GHI':('num','x>=0'),
      'HExtra':('num','x>=0'),
      'SunZen':('num','x>=0'),
      'SunAz':('num','x>=-180'),
        }

  var=pvl_tools.Parse(Vars,Expect)


  small=1e-06

  COSTT=pvl_tools.cosd(SurfTilt)*pvl_tools.cosd(SunZen) + pvl_tools.sind(SurfTilt)*pvl_tools.sind(SunZen)*pvl_tools.cosd(SunAz - SurfAz)
  RB=np.max(COSTT,0) / np.max(pvl_tools.cosd(SunZen),0.01745)
  AI=DNI / HExtra
  GHI[GHI < small]=small
  HB=DNI*(pvl_tools.cosd(SunZen))
  HB[HB < 0]=0
  GHI[GHI < 0]=0
  F=np.sqrt(HB / GHI)
  SCUBE=(pvl_tools.sind(SurfTilt*(0.5))) ** 3


  SkyDiffuse=DHI*((AI*(RB) + (1 - AI)*(0.5)*((1 + pvl_tools.cosd(SurfTilt)))*((1 + F*(SCUBE)))))

  return SkyDiffuse
