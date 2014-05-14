import numpy as np
import pvl_tools

def pvl_klucher1979(SurfTilt,SurfAz,DHI,GHI,SunZen,SunAz):
    '''
    Determine diffuse irradiance from the sky on a tilted surface using Klucher's 1979 model


    Klucher's 1979 model determines the diffuse irradiance from the sky
    (ground reflected irradiance is not included in this algorithm) on a
    tilted surface using the surface tilt angle, surface azimuth angle,
    diffuse horizontal irradiance, direct normal irradiance, global
    horizontal irradiance, extraterrestrial irradiance, sun zenith angle,
    and sun azimuth angle.

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

    GHI : float or DataFrame
            Global  irradiance in W/m^2. 
            DNI must be >=0.

    SunZen : float or DataFrame
            apparent (refraction-corrected) zenith
            angles in decimal degrees. 
            SunZen must be >=0 and <=180.

    SunAz : float or DataFrame
            Sun azimuth angles in decimal degrees.
            SunAz must be >=0 and <=360. The Azimuth convention is defined
            as degrees east of north (e.g. North = 0, East = 90, West = 270).

    Returns
    -------
    SkyDiffuse : float or DataFrame

                the diffuse component of the solar radiation  on an
                arbitrarily tilted surface defined by the Klucher model as given in
                Loutzenhiser et. al (2007) equation 4.
                SkyDiffuse is the diffuse component ONLY and does not include the ground
                reflected irradiance or the irradiance due to the beam.
                SkyDiffuse is a column vector vector with a number of elements equal to
                the input vector(s).

    References
    ----------
    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267

    [2] Klucher, T.M., 1979. Evaluation of models to predict insolation on tilted
    surfaces. Solar Energy 23 (2), 111-114.

    See also
    --------
    pvl_ephemeris   
    pvl_extraradiation   
    pvl_isotropicsky
    pvl_haydavies1980   
    pvl_perez 
    pvl_reindl1990  
    pvl_kingdiffuse

    '''
    Vars=locals()
    Expect={'SurfTilt':('num','x>=0'),
            'SurfAz':('x>=-180'),
            'DHI':('x>=0'),
            'GHI':('x>=0'),
            'SunZen':('x>=0'),
            'SunAz':('x>=-180')
            }

    var=pvl_tools.Parse(Vars,Expect)

    GHI[GHI < DHI]=DHI
    GHI[GHI < 1e-06]=1e-06

    COSTT=pvl_tools.cosd(SurfTilt)*pvl_tools.cosd(SunZen) + pvl_tools.sind(SurfTilt)*pvl_tools.sind(SunZen)*pvl_tools.cosd(SunAz - SurfAz)

    F=1 - ((DHI / GHI) ** 2)

    SkyDiffuse=DHI*((0.5*((1 + pvl_tools.cosd(SurfTilt)))))*((1 + F*(((pvl_tools.sind(SurfTilt / 2)) ** 3))))*((1 + F*(((COSTT) ** 2))*(((pvl_tools.sind(SunZen)) ** 3))))

    return SkyDiffuse
