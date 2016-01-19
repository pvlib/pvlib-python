"""
The ``clearsky`` module contains several methods 
to calculate clear sky GHI, DNI, and DHI.
"""

from __future__ import division

import logging
logger = logging.getLogger('pvlib')

import os

import numpy as np
import pandas as pd

from pvlib import tools
from pvlib import irradiance
from pvlib import atmosphere
from pvlib import solarposition


def ineichen(time, latitude, longitude, altitude=0, linke_turbidity=None, 
             solarposition_method='nrel_numpy', zenith_data=None,
             airmass_model='young1994', airmass_data=None,
             interp_turbidity=True):
    '''
    Determine clear sky GHI, DNI, and DHI from Ineichen/Perez model

    Implements the Ineichen and Perez clear sky model for global horizontal
    irradiance (GHI), direct normal irradiance (DNI), and calculates
    the clear-sky diffuse horizontal (DHI) component as the difference
    between GHI and DNI*cos(zenith) as presented in [1, 2]. A report on clear
    sky models found the Ineichen/Perez model to have excellent performance
    with a minimal input data set [3]. 
    
    Default values for montly Linke turbidity provided by SoDa [4, 5].

    Parameters
    -----------
    time : pandas.DatetimeIndex
    
    latitude : float
    
    longitude : float

    altitude : float
    
    linke_turbidity : None or float
        If None, uses ``LinkeTurbidities.mat`` lookup table.
    
    solarposition_method : string
        Sets the solar position algorithm. 
        See solarposition.get_solarposition()
    
    zenith_data : None or Series
        If None, ephemeris data will be calculated using ``solarposition_method``.
    
    airmass_model : string
        See pvlib.airmass.relativeairmass().
    
    airmass_data : None or Series
        If None, absolute air mass data will be calculated using 
        ``airmass_model`` and location.alitude.
    
    interp_turbidity : bool
        If ``True``, interpolates the monthly Linke turbidity values
        found in ``LinkeTurbidities.mat`` to daily values.

    Returns
    --------
    DataFrame with the following columns: ``ghi, dni, dhi``.

    Notes
    -----
    If you are using this function
    in a loop, it may be faster to load LinkeTurbidities.mat outside of
    the loop and feed it in as a keyword argument, rather than
    having the function open and process the file each time it is called.

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
    # Added functionality by Will Holmgren (@wholmgren)
    
    I0 = irradiance.extraradiation(time.dayofyear)
    
    if zenith_data is None:
        ephem_data = solarposition.get_solarposition(time,
                                                     latitude=latitude,
                                                     longitude=longitude,
                                                     altitude=altitude,
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
        TL = lookup_linke_turbidity(time, latitude, longitude,
                                    interp_turbidity=interp_turbidity)
    else:
        TL = linke_turbidity

    # Get the absolute airmass assuming standard local pressure (per
    # alt2pres) using Kasten and Young's 1989 formula for airmass.
    
    if airmass_data is None:
        AMabsolute = atmosphere.absoluteairmass(airmass_relative=atmosphere.relativeairmass(ApparentZenith, airmass_model),
                                                pressure=atmosphere.alt2pres(altitude))
    else:
        AMabsolute = airmass_data
        
    fh1 = np.exp(-altitude/8000.)
    fh2 = np.exp(-altitude/1250.)
    cg1 = 5.09e-05 * altitude + 0.868
    cg2 = 3.92e-05 * altitude + 0.0387
    logger.debug('fh1=%s, fh2=%s, cg1=%s, cg2=%s', fh1, fh2, cg1, cg2)

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
    
    clearsky_GHI = ( cg1 * I0 * cos_zenith *
                     np.exp(-cg2*AMabsolute*(fh1 + fh2*(TL - 1))) *
                     np.exp(0.01*AMabsolute**1.8) )
    clearsky_GHI[clearsky_GHI < 0] = 0
    
    # BncI == "normal beam clear sky radiation"
    b = 0.664 + 0.163/fh1
    BncI = b * I0 * np.exp( -0.09 * AMabsolute * (TL - 1) )
    logger.debug('b=%s', b)
    
    # "empirical correction" SE 73, 157 & SE 73, 312.
    BncI_2 = ( clearsky_GHI *
               ( 1 - (0.1 - 0.2*np.exp(-TL))/(0.1 + 0.882/fh1) ) /
               cos_zenith )

    clearsky_DNI = np.minimum(BncI, BncI_2)
    
    clearsky_DHI = clearsky_GHI - clearsky_DNI*cos_zenith
    
    df_out = pd.DataFrame({'ghi':clearsky_GHI, 'dni':clearsky_DNI, 
                           'dhi':clearsky_DHI})
    df_out.fillna(0, inplace=True)
    
    return df_out


def lookup_linke_turbidity(time, latitude, longitude, filepath=None,
                           interp_turbidity=True):
    """
    Look up the Linke Turibidity from the ``LinkeTurbidities.mat``
    data file supplied with pvlib.

    Parameters
    ----------
    time : pandas.DatetimeIndex

    latitude : float

    longitude : float

    filepath : string
        The path to the ``.mat`` file.

    interp_turbidity : bool
        If ``True``, interpolates the monthly Linke turbidity values
        found in ``LinkeTurbidities.mat`` to daily values.

    Returns
    -------
    turbidity : Series
    """

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
    
    try:
        import scipy.io
    except ImportError:
        raise ImportError('The Linke turbidity lookup table requires scipy. ' +
                          'You can still use clearsky.ineichen if you ' +
                          'supply your own turbidities.')

    if filepath is None:
        pvlib_path = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(pvlib_path, 'data', 'LinkeTurbidities.mat')

    mat = scipy.io.loadmat(filepath)
    linke_turbidity_table = mat['LinkeTurbidity']

    latitude_index = np.around(_linearly_scale(latitude, 90, -90, 1, 2160))
    longitude_index = np.around(_linearly_scale(longitude, -180, 180, 1, 4320))

    g = linke_turbidity_table[latitude_index][longitude_index]

    if interp_turbidity:
        logger.info('interpolating turbidity to the day')
        # Cata covers 1 year.
        # Assume that data corresponds to the value at
        # the middle of each month.
        # This means that we need to add previous Dec and next Jan
        # to the array so that the interpolation will work for 
        # Jan 1 - Jan 15 and Dec 16 - Dec 31.
        # Then we map the month value to the day of year value.
        # This is approximate and could be made more accurate.
        g2 = np.concatenate([[g[-1]], g, [g[0]]])
        days = np.linspace(-15, 380, num=14)
        linke_turbidity = pd.Series(np.interp(time.dayofyear, days, g2),
                                    index=time)
    else:
        logger.info('using monthly turbidity')
        apply_month = lambda x: g[x[0]-1]
        linke_turbidity = pd.DataFrame(time.month, index=time)
        linke_turbidity = linke_turbidity.apply(apply_month, axis=1)

    linke_turbidity /= 20.

    return linke_turbidity


def haurwitz(apparent_zenith):
    '''
    Determine clear sky GHI from Haurwitz model.
   
    Implements the Haurwitz clear sky model for global horizontal
    irradiance (GHI) as presented in [1, 2]. A report on clear
    sky models found the Haurwitz model to have the best performance of
    models which require only zenith angle [3]. Extreme care should
    be taken in the interpretation of this result! 

    Parameters
    ----------
    apparent_zenith : Series
        The apparent (refraction corrected) sun zenith angle
        in degrees.

    Returns
    -------        
    pd.Series
    The modeled global horizonal irradiance in W/m^2 provided
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
    '''

    cos_zenith = tools.cosd(apparent_zenith)

    clearsky_GHI = 1098.0 * cos_zenith * np.exp(-0.059/cos_zenith)

    clearsky_GHI[clearsky_GHI < 0] = 0
    
    df_out = pd.DataFrame({'ghi':clearsky_GHI})
    
    return df_out


def _linearly_scale(inputmatrix, inputmin, inputmax, outputmin, outputmax):
    """ used by linke turbidity lookup function """
    
    inputrange = inputmax - inputmin
    outputrange = outputmax - outputmin
    OutputMatrix = (inputmatrix-inputmin) * outputrange/inputrange + outputmin
    return OutputMatrix
