

import numpy as np
import pvl_tools

def pvl_haydavies1980(SurfTilt,SurfAz,DHI,DNI,HExtra,SunZen,SunAz):

    '''
    Determine diffuse irradiance from the sky on a tilted surface using Hay & Davies' 1980 model

    
    Hay and Davies' 1980 model determines the diffuse irradiance from the sky
    (ground reflected irradiance is not included in this algorithm) on a
    tilted surface using the surface tilt angle, surface azimuth angle,
    diffuse horizontal irradiance, direct normal irradiance, 
    extraterrestrial irradiance, sun zenith angle, and sun azimuth angle.


    Parameters
    ----------

    SurfTilt : float or DataFrame
          Surface tilt angles in decimal degrees.
          SurfTilt must be >=0 and <=180. The tilt angle is defined as
          degrees from horizontal (e.g. surface facing up = 0, surface facing
          horizon = 90)

    SurfAz : float or DataFrame
          Surface azimuth angles in decimal degrees.
          SurfAz must be >=0 and <=360. The Azimuth convention is defined
          as degrees east of north (e.g. North = 0, South=180 East = 90, West = 270).

    DHI : float or DataFrame
          diffuse horizontal irradiance in W/m^2. 
          DHI must be >=0.

    DNI : float or DataFrame
          direct normal irradiance in W/m^2. 
          DNI must be >=0.

    HExtra : float or DataFrame
          extraterrestrial normal irradiance in W/m^2. 
           HExtra must be >=0.

    SunZen : float or DataFrame
          apparent (refraction-corrected) zenith
          angles in decimal degrees. 
          SunZen must be >=0 and <=180.

    SunAz : float or DataFrame
          Sun azimuth angles in decimal degrees.
          SunAz must be >=0 and <=360. The Azimuth convention is defined
          as degrees east of north (e.g. North = 0, East = 90, West = 270).

    Returns
    --------

    SkyDiffuse : float or DataFrame

          the diffuse component of the solar radiation  on an
          arbitrarily tilted surface defined by the Perez model as given in
          reference [3].
          SkyDiffuse is the diffuse component ONLY and does not include the ground
          reflected irradiance or the irradiance due to the beam.

    References
    -----------
    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267
    
    [2] Hay, J.E., Davies, J.A., 1980. Calculations of the solar radiation incident
    on an inclined surface. In: Hay, J.E., Won, T.K. (Eds.), Proc. of First
    Canadian Solar Radiation Data Workshop, 59. Ministry of Supply
    and Services, Canada.

    See Also
    --------
    pvl_ephemeris   
    pvl_extraradiation   
    pvl_isotropicsky
    pvl_reindl1990   
    pvl_perez 
    pvl_klucher1979   
    pvl_kingdiffuse
    pvl_spa

    '''

    Vars=locals()
    Expect={'SurfTilt':('num','x>=0'),
              'SurfAz':('x>=-180'),
              'DHI':('x>=0'),
              'DNI':('x>=0'),
              'HExtra':('x>=0'),
              'SunZen':('x>=0'),
              'SunAz':('x>=-180'),
              }
    var=pvl_tools.Parse(Vars,Expect)

    COSTT=pvl_tools.cosd(SurfTilt)*pvl_tools.cosd(SunZen) + pvl_tools.sind(SurfTilt)*pvl_tools.sind(SunZen)*pvl_tools.cosd(SunAz - SurfAz)

    RB=np.max(COSTT,0) / np.max(pvl_tools.cosd(SunZen),0.01745)

    AI=DNI / HExtra

    SkyDiffuse=DHI*((AI*(RB) + (1 - AI)*(0.5)*((1 + pvl_tools.cosd(SurfTilt)))))


    return SkyDiffuse
