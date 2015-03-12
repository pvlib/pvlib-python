"""
Contains several methods to calculate clear sky GHI, DNI, and DHI.
"""

from __future__ import division

import logging
logger = logging.getLogger('pvlib')

import os

import numpy as np
import pandas as pd
import scipy.io

from pvlib import tools
from pvlib import irradiance
from pvlib import atmosphere
from pvlib import solarposition



def ineichen(time, location, linke_turbidity=None, 
             solarposition_method='pyephem', zenith_data=None,
             airmass_model='young1994', airmass_data=None,
             interp_turbidity=True):
    '''
    Determine clear sky GHI, DNI, and DHI from Ineichen/Perez model

    Implements the Ineichen and Perez clear sky model for global horizontal
    irradiance (GHI), direct normal irradiance (DNI), and calculates
    the clear-sky diffuse horizontal (DHI) component as the difference
    between GHI and DNI*cos(zenith) as presented in [1, 2]. A report on clear
    sky models found the Ineichen/Perez model to have excellent performance
    with a minimal input data set [3]. Default values for Linke turbidity
    provided by SoDa [4, 5].

    Parameters
    -----------
    time : pandas.DatetimeIndex
    
    location : pvlib.Location
    
    linke_turbidity : None or float
    
    solarposition_method : string
        See pvlib.solarposition.get_solarposition()
    
    zenith_data : None or pandas.Series
        If None, ephemeris data will be calculated using ``solarposition_method``.
    
    airmass_model : string
        See pvlib.airmass.relativeairmass().
    
    airmass_data : None or pandas.Series
        If None, absolute air mass data will be calculated using 
        ``airmass_model`` and location.alitude.

    Returns
    --------

    ClearSkyGHI : Dataframe.
         the modeled global horizonal irradiance in W/m^2 provided
          by the Ineichen clear-sky model.

    ClearSkyDNI : Dataframe.
        the modeled direct normal irradiance in W/m^2 provided
        by the Ineichen clear-sky model.

    ClearSkyDHI : Dataframe.
        the calculated diffuse horizonal irradiance in W/m^2 
        provided by the Ineichen clear-sky model.

    Notes
    -----
    If you are using this function
    in a loop, it may be faster to load LinkeTurbidities.mat outside of
    the loop and feed it in as a variable, rather than
    having the function open the file each time it is called.


    References
    ----------

    [1] P. Ineichen and R. Perez, "A New airmass independent formulation for
        the Linke turbidity coefficient", Solar Energy, vol 73, pp. 151-157, 2002.

    [2] R. Perez et. al., "A New Operational Model for Satellite-Derived
        Irradiances: Description and Validation", Solar Energy, vol 73, pp.
        307-317, 2002.

    [3] M. Reno, C. Hansen, and J. Stein, "Global Horizontal Irradiance Clear
        Sky Models: Implementation and Analysis", Sandia National
        Laboratories, SAND2012-2389, 2012.

    [4] http://www.soda-is.com/eng/services/climat_free_eng.php#c5 (obtained
        July 17, 2012).

    [5] J. Remund, et. al., "Worldwide Linke Turbidity Information", Proc.
        ISES Solar World Congress, June 2003. Goteborg, Sweden.
    '''
    # Initial implementation of this algorithm by Matthew Reno.
    # Ported to python by Rob Andrews
    # Added functionality by Will Holmgren
    
    I0 = irradiance.extraradiation(time.dayofyear)
    
    if zenith_data is None:
        ephem_data = solarposition.get_solarposition(time, location, 
                                                     method=solarposition_method)
        time = ephem_data.index # fixes issue with time possibly not being tz-aware    
        try:
            ApparentZenith = ephem_data['apparent_zenith']
        except KeyError:
            ApparentZenith = ephem_data['zenith']
            logger.warning('could not find apparent_zenith. using zenith')
    else:
        ApparentZenith = zenith_data
    #ApparentZenith[ApparentZenith >= 90] = 90 # can cause problems in edge cases

    
    if linke_turbidity is None:
        # The .mat file 'LinkeTurbidities.mat' contains a single 2160 x 4320 x 12
        # matrix of type uint8 called 'LinkeTurbidity'. The rows represent global
        # latitudes from 90 to -90 degrees; the columns represent global longitudes
        # from -180 to 180; and the depth (third dimension) represents months of
        # the year from January (1) to December (12). To determine the Linke
        # turbidity for a position on the Earth's surface for a given month do the
        # following: LT = LinkeTurbidity(LatitudeIndex, LongitudeIndex, month).  
        # Note that the numbers within the matrix are 20 * Linke Turbidity, 
        # so divide the number from the file by 20 to get the
        # turbidity.
        
        # consider putting this code at module level
        this_path = os.path.dirname(os.path.abspath(__file__))
        logger.debug('this_path={}'.format(this_path))
        mat = scipy.io.loadmat(os.path.join(this_path, 'data', 'LinkeTurbidities.mat'))
        linke_turbidity = mat['LinkeTurbidity']
        LatitudeIndex = np.round_(_linearly_scale(location.latitude,90,- 90,1,2160))
        LongitudeIndex = np.round_(_linearly_scale(location.longitude,- 180,180,1,4320))
        g = linke_turbidity[LatitudeIndex][LongitudeIndex]
        if interp_turbidity:
            logger.info('interpolating turbidity to the day')
            g2 = np.concatenate([[g[-1]], g, [g[0]]]) # wrap ends around
            days = np.linspace(-15, 380, num=14) # map day of year onto month (approximate)
            LT = pd.Series(np.interp(time.dayofyear, days, g2), index=time)
        else:
            logger.info('using monthly turbidity')
            ApplyMonth = lambda x:g[x[0]-1]
            LT = pd.DataFrame(time.month, index=time)
            LT = LT.apply(ApplyMonth, axis=1)
        TL = LT / 20.
        logger.info('using TL=\n{}'.format(TL))
    else:
        TL = linke_turbidity

    # Get the absolute airmass assuming standard local pressure (per
    # alt2pres) using Kasten and Young's 1989 formula for airmass.
    
    if airmass_data is None:
        AMabsolute = atmosphere.absoluteairmass(AMrelative=atmosphere.relativeairmass(ApparentZenith, airmass_model),
                                                pressure=atmosphere.alt2pres(location.altitude))
    else:
        AMabsolute = airmass_data
        
    fh1 = np.exp(-location.altitude/8000.)
    fh2 = np.exp(-location.altitude/1250.)
    cg1 = 5.09e-05 * location.altitude + 0.868
    cg2 = 3.92e-05 * location.altitude + 0.0387
    logger.debug('fh1={}, fh2={}, cg1={}, cg2={}'.format(fh1, fh2, cg1, cg2))

    #  Dan's note on the TL correction: By my reading of the publication on
    #  pages 151-157, Ineichen and Perez introduce (among other things) three
    #  things. 1) Beam model in eqn. 8, 2) new turbidity factor in eqn 9 and
    #  appendix A, and 3) Global horizontal model in eqn. 11. They do NOT appear
    #  to use the new turbidity factor (item 2 above) in either the beam or GHI
    #  models. The phrasing of appendix A seems as if there are two separate
    #  corrections, the first correction is used to correct the beam/GHI models,
    #  and the second correction is used to correct the revised turibidity
    #  factor. In my estimation, there is no need to correct the turbidity
    #  factor used in the beam/GHI models.

    #  Create the corrected TL for TL < 2
    #  TLcorr = TL;
    #  TLcorr(TL < 2) = TLcorr(TL < 2) - 0.25 .* (2-TLcorr(TL < 2)) .^ (0.5);

    #  This equation is found in Solar Energy 73, pg 311. 
    #  Full ref: Perez et. al., Vol. 73, pp. 307-317 (2002).
    #  It is slightly different than the equation given in Solar Energy 73, pg 156. 
    #  We used the equation from pg 311 because of the existence of known typos 
    #  in the pg 156 publication (notably the fh2-(TL-1) should be fh2 * (TL-1)). 
    
    cos_zenith = tools.cosd(ApparentZenith)
    
    clearsky_GHI = cg1 * I0 * cos_zenith * np.exp(-cg2*AMabsolute*(fh1 + fh2*(TL - 1))) * np.exp(0.01*AMabsolute**1.8)
    clearsky_GHI[clearsky_GHI < 0] = 0
    
    # BncI == "normal beam clear sky radiation"
    b = 0.664 + 0.163/fh1
    BncI = b * I0 * np.exp( -0.09 * AMabsolute * (TL - 1) )
    logger.debug('b={}'.format(b))
    
    # "empirical correction" SE 73, 157 & SE 73, 312.
    BncI_2 = clearsky_GHI * ( 1 - (0.1 - 0.2*np.exp(-TL))/(0.1 + 0.882/fh1) ) / cos_zenith
    #return BncI, BncI_2
    clearsky_DNI = np.minimum(BncI, BncI_2) # Will H: use np.minimum explicitly
    
    clearsky_DHI = clearsky_GHI - clearsky_DNI*cos_zenith
    
    df_out = pd.DataFrame({'GHI':clearsky_GHI, 'DNI':clearsky_DNI, 
                           'DHI':clearsky_DHI})
    df_out.fillna(0, inplace=True)
    #df_out['BncI'] = BncI
    #df_out['BncI_2'] = BncI
    
    return df_out
    
    
    
def haurwitz(ApparentZenith):
    '''
    Determine clear sky GHI from Haurwitz model
   
    Implements the Haurwitz clear sky model for global horizontal
    irradiance (GHI) as presented in [1, 2]. A report on clear
    sky models found the Haurwitz model to have the best performance of
    models which require only zenith angle [3]. Extreme care should
    be taken in the interpretation of this result! 

    Parameters
    ----------
    ApparentZenith : DataFrame
        The apparent (refraction corrected) sun zenith angle
        in degrees.

    Returns
    -------        
    pd.Series. The modeled global horizonal irradiance in W/m^2 provided
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
    maketimestruct    
    makelocationstruct   
    ephemeris   
    spa
    ineichen
    '''

    cos_zenith = tools.cosd(ApparentZenith)

    clearsky_GHI = 1098.0 * cos_zenith * np.exp(-0.059/cos_zenith)

    clearsky_GHI[clearsky_GHI < 0] = 0
    
    df_out = pd.DataFrame({'GHI':clearsky_GHI})
    
    return df_out
	    
    

def _linearly_scale(inputmatrix, inputmin, inputmax, outputmin, outputmax):
    """ used by linke turbidity lookup function """
    
    inputrange = inputmax - inputmin
    outputrange = outputmax - outputmin
    OutputMatrix = (inputmatrix - inputmin) * outputrange / inputrange + outputmin
    return OutputMatrix



def disc(GHI, zenith, times, pressure=101325):
    '''
    Estimate Direct Normal Irradiance from Global Horizontal Irradiance 
    using the DISC model.

    The DISC algorithm converts global horizontal irradiance to direct
    normal irradiance through empirical relationships between the global
    and direct clearness indices. 

    Parameters
    ----------

    GHI : Series
        Global horizontal irradiance in W/m^2.

    zenith : Series
        True (not refraction - corrected) solar zenith 
        angles in decimal degrees. 

    times : Series or DatetimeIndex

    pressure : float or Series
        Site pressure in Pascal.

    Returns   
    -------
    DataFrame with the following keys:
        * ``DNI_gen_DISC``: The modeled direct normal irradiance 
          in W/m^2 provided by the
          Direct Insolation Simulation Code (DISC) model. 
        * ``Kt_gen_DISC``: Ratio of global to extraterrestrial 
          irradiance on a horizontal plane.
        * ``AM``: Airmass
        * ``Ztemp``: Zenith

    References
    ----------

    [1] Maxwell, E. L., "A Quasi-Physical Model for Converting Hourly 
    Global Horizontal to Direct Normal Insolation", Technical 
    Report No. SERI/TR-215-3087, Golden, CO: Solar Energy Research 
    Institute, 1987.

    [2] J.W. "Fourier series representation of the position of the sun". 
    Found at:
    http://www.mail-archive.com/sundial@uni-koeln.de/msg01050.html on
    January 12, 2012

    See Also 
    --------
    ephemeris 
    alt2pres 
    dirint
    '''

    logger.debug('clearsky.disc')
    
    temp = pd.DataFrame(index=times, columns=['A','B','C'])

    doy = times.dayofyear
    
    DayAngle = 2. * np.pi*(doy - 1) / 365
    
    re = (1.00011 + 0.034221*np.cos(DayAngle) + 0.00128*np.sin(DayAngle)
          + 0.000719*np.cos(2.*DayAngle) + 7.7e-05*np.sin(2.*DayAngle) )
          
    I0 = re * 1370.
    I0h = I0 * np.cos(np.radians(zenith))
    
    Ztemp = zenith.copy()
    Ztemp[zenith > 87] = np.NaN
    
    AM = 1.0 / ( np.cos(np.radians(Ztemp)) + 0.15*( (93.885 - Ztemp)**(-1.253) ) ) * (pressure / 101325)
    
    Kt = GHI / I0h
    Kt[Kt < 0] = 0
    Kt[Kt > 2] = np.NaN
    
    temp.A[Kt > 0.6] = -5.743 + 21.77*(Kt[Kt > 0.6]) - 27.49*(Kt[Kt > 0.6] ** 2) + 11.56*(Kt[Kt > 0.6] ** 3)
    temp.B[Kt > 0.6] = 41.4 - 118.5*(Kt[Kt > 0.6]) + 66.05*(Kt[Kt > 0.6] ** 2) + 31.9*(Kt[Kt > 0.6] ** 3)
    temp.C[Kt > 0.6] = -47.01 + 184.2*(Kt[Kt > 0.6]) - 222.0 * Kt[Kt > 0.6] ** 2 + 73.81*(Kt[Kt > 0.6] ** 3)
    temp.A[Kt <= 0.6] = 0.512 - 1.56*(Kt[Kt <= 0.6]) + 2.286*(Kt[Kt <= 0.6] ** 2) - 2.222*(Kt[Kt <= 0.6] ** 3)
    temp.B[Kt <= 0.6] = 0.37 + 0.962*(Kt[Kt <= 0.6])
    temp.C[Kt <= 0.6] = -0.28 + 0.932*(Kt[Kt <= 0.6]) - 2.048*(Kt[Kt <= 0.6] ** 2)
    
    #return to numeric after masking operations 
    temp = temp.astype(float)

    delKn = temp.A + temp.B * np.exp(temp.C*AM)
   
    Knc = 0.866 - 0.122*(AM) + 0.0121*(AM ** 2) - 0.000653*(AM ** 3) + 1.4e-05*(AM ** 4)
    Kn = Knc - delKn
    
    DNI = Kn * I0

    DNI[zenith > 87] = np.NaN
    DNI[GHI < 1] = np.NaN
    DNI[DNI < 0] = np.NaN

    DFOut = pd.DataFrame({'DNI_gen_DISC':DNI})

    DFOut['Kt_gen_DISC'] = Kt
    DFOut['AM'] = AM
    DFOut['Ztemp'] = Ztemp

    return DFOut
