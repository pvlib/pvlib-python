
import numpy as np
import pvl_tools


def pvl_clearsky_haurwitz(ApparentZenith):
    '''
    Determine clear sky GHI from Haurwitz model


    Implements the Haurwitz clear sky model for global horizontal
    irradiance (GHI) as presented in [1, 2]. A report on clear
    sky models found the Haurwitz model to have the best performance of
    models which require only zenith angle [3].

    Parameters
    ----------
    ApparentZenith : DataFrame

      The apparent (refraction corrected) sun zenith angle in degrees.

    Returns
    -------
    ClearSkyGHI : DataFrame

      the modeled global horizonal irradiance in W/m^2 provided
      by the Haurwitz clear-sky model.

      Initial implementation of this algorithm by Matthew Reno.

    References
    ----------

    [1] B. Haurwitz, "Insolation in Relation to Cloudiness and Cloud
     Density," Journal of Meteorology, vol. 2, pp. 154-166, 1945.

    [2] B. Haurwitz, "Insolation in Relation to Cloud Type," Journal of
     Meteorology, vol. 3, pp. 123-124, 1946.

    [3] M. Reno, C. Hansen, and J. Stein, "Global Horizontal Irradiance Clear
     Sky Models: Implementation and Analysis", Sandia National
     Laboratories, SAND2012-2389, 2012.

    See Also
    ---------
    pvl_maketimestruct
    pvl_makelocationstruct
    pvl_ephemeris
    pvl_spa
    pvl_ineichen
    '''

    Vars=locals()

    Expect={'ApparentZenith':('x<=180','x>=0')}
    var=pvl_tools.Parse(Vars,Expect)

    ClearSkyGHI=1098.0 * pvl_tools.cosd(ApparentZenith)*(np.exp(- 0.059 / pvl_tools.cosd(ApparentZenith)))

    ClearSkyGHI[ClearSkyGHI < 0]=0

    return ClearSkyGHI
