"""
Contains several methods to calculate clear sky GHI, DNI, and DHI.
"""

import logging
pvl_logger = logging.getLogger('pvlib')

import os
import pdb

import numpy as np
import scipy.io
import pandas as pd

import pvl_tools
import pvl_extraradiation
import pvl_alt2pres
import airmass
import pvl_ephemeris
import solarposition


def ineichen(time, location, linke_turbidity=None, 
             solarposition_method='pyephem',
             airmass_model='kastenyoung1989'):
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
    time : Dataframe.index
            A timezone aware pandas dataframe index. 

    location : pvlib Location object

    Other Parameters 
    ----------------

    linke_turbidity : Optional, float or DataFrame

                    An optional input to provide your own Linke
                    turbidity. If this input is omitted, the default Linke turbidity
                    maps will be used. linke_turbidity may be a float or 
                    dataframe of Linke turbidities. If dataframe is provided, the same
                    turbidity will be used for all time/location sets. If a dataframe is
                    provided, it must be of the same size as any time/location dataframes
                    and each element of the dataframe corresponds to any time and location
                    elements.

    Returns
    --------

    ClearSkyGHI : Dataframe

         the modeled global horizonal irradiance in W/m^2 provided
          by the Ineichen clear-sky model.

    ClearSkyDNI : Dataframe

        the modeled direct normal irradiance in W/m^2 provided
        by the Ineichen clear-sky model.

    ClearSkyDHI : Dataframe

        the calculated diffuse horizonal irradiance in W/m^2 
        provided by the Ineichen clear-sky model.

    Notes
    -----

        This implementation of the Ineichen model requires a number of other
        PV_LIB functions including pvl_ephemeris, pvl_date2doy,
        pvl_extraradiation, pvl_absoluteairmass, pvl_relativeairmass, and
        pvl_alt2pres. It also requires the file "LinkeTurbidities.mat" to be
        in the working directory. If you are using pvl_ineichen
        in a loop, it may be faster to load LinkeTurbidities.mat outside of
        the loop and feed it into pvl_ineichen as a variable, rather than
        having pvl_ineichen open the file each time it is called (or utilize
        column vectors of time/location instead of a loop).

        Initial implementation of this algorithm by Matthew Reno.

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


    See Also
    --------

    pvl_maketimestruct    
    pvl_makelocationstruct   
    pvl_ephemeris
    pvl_haurwitz

    '''
    
    I0 = pvl_extraradiation.pvl_extraradiation(time.dayofyear)
    
    ephem_data = solarposition.get_solarposition(time, location, 
                                                 method=solarposition_method)
    
    time = ephem_data.index # fixes issue with time possibly not being tz-aware
    
    try:
        ApparentZenith = ephem_data['apparent_zenith']
    except KeyError:
        ApparentZenith = ephem_data['zenith']
        pvl_logger.warning('could not find apparent_zenith. using zenith')
    #ApparentZenith[ApparentZenith >= 90] = 90 # can cause problems in edge cases

    
    if linke_turbidity is None:
        # The .mat file 'LinkeTurbidities.mat' contains a single 2160 x 4320 x 12
        # matrix of type uint8 called 'LinkeTurbidity'. The rows represent global
        # latitudes from 90 to -90 degrees; the columns represent global longitudes
        # from -180 to 180; and the depth (third dimension) represents months of
        # the year from January (1) to December (12). To determine the Linke
        # turbidity for a position on the Earth's surface for a given month do the
        # following: LT = LinkeTurbidity(LatitudeIndex, LongitudeIndex, month).  Note that the numbers within the matrix are 20 * Linke
        # Turbidity, so divide the number from the file by 20 to get the
        # turbidity.
        
        # consider putting this code at module level
        this_path = os.path.dirname(os.path.abspath(__file__))
        pvl_logger.debug('this_path={}'.format(this_path))
        mat = scipy.io.loadmat(this_path+'/LinkeTurbidities.mat')
        linke_turbidity = mat['LinkeTurbidity']
        LatitudeIndex = np.round_(LinearlyScale(location.latitude,90,- 90,1,2160))
        LongitudeIndex = np.round_(LinearlyScale(location.longitude,- 180,180,1,4320))
        g = linke_turbidity[LatitudeIndex][LongitudeIndex]
        ApplyMonth = lambda x:g[x[0]-1]
        LT = pd.DataFrame(time.month)
        LT.index = time
        LT = LT.apply(ApplyMonth,axis=1)
        TL = LT / float(20)
        pvl_logger.info('using TL={}'.format(TL))
    else:
        TL = linke_turbidity

    # Get the absolute airmass assuming standard local pressure (per
    # pvl_alt2pres) using Kasten and Young's 1989 formula for airmass.

    AMabsolute = airmass.absoluteairmass(AMrelative=airmass.relativeairmass(ApparentZenith, airmass_model),
                                         pressure=pvl_alt2pres.pvl_alt2pres(location.altitude))
    
    fh1 = np.exp(-location.altitude/8000.) # Will H: added . for float division!
    fh2 = np.exp(-location.altitude/1250.)
    cg1 = 5.09e-05 * location.altitude + 0.868
    cg2 = 3.92e-05 * location.altitude + 0.0387
    pvl_logger.debug('fh1={}, fh2={}, cg1={}, cg2={}'.format(fh1, fh2, cg1, cg2))

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
    
    cos_zenith = pvl_tools.cosd(ApparentZenith)
    
    clearsky_GHI = cg1 * I0 * cos_zenith * np.exp(-cg2*AMabsolute*(fh1 + fh2*(TL - 1))) * np.exp(0.01*AMabsolute**1.8)
    clearsky_GHI[clearsky_GHI < 0] = 0
    
    # BncI == "normal beam clear sky radiation"
    b = 0.664 + 0.163/fh1
    BncI = b * I0 * np.exp( -0.09 * AMabsolute * (TL - 1) )
    pvl_logger.debug('b={}'.format(b))
    
    # "empirical correction" SE 73, 157 & SE 73, 312.
    BncI_2 = clearsky_GHI * ( 1 - (0.1 - 0.2*np.exp(-TL))/(0.1 + 0.882/fh1) ) / cos_zenith
    #return BncI, BncI_2
    clearsky_DNI = np.minimum(BncI, BncI_2) # Will H: use np.minimum explicitly
    
    clearsky_DHI = clearsky_GHI - clearsky_DNI*cos_zenith
    
    df_out = pd.DataFrame({'GHI':clearsky_GHI, 'DNI':clearsky_DNI, 
                           'DHI':clearsky_DHI})
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

    cos_zenith = pvl_tools.cosd(ApparentZenith)

    clearsky_GHI = 1098.0 * cos_zenith * np.exp(-0.059/cos_zenith)

    clearsky_GHI[clearsky_GHI < 0] = 0
    
    df_out = pd.DataFrame({'GHI':clearsky_GHI})
    
    return df_out
	    
    

def LinearlyScale(inputmatrix, inputmin, inputmax, outputmin, outputmax):
    """ used by linke turbidity lookup function """
    
    inputrange = inputmax - inputmin
    outputrange = outputmax - outputmin
    OutputMatrix = (inputmatrix - inputmin) * outputrange / inputrange + outputmin
    return OutputMatrix
