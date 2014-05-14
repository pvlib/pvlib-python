import numpy as np
import scipy
import os
import pvl_tools
import pvl_extraradiation
import pvl_alt2pres
import pvl_relativeairmass
import pvl_absoluteairmass
import pvl_ephemeris
import pandas as pd
import pdb

def pvl_clearsky_ineichen(Time,Location,LinkeTurbidity=-999):
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
    Time : Dataframe.index
            A timezone aware pandas dataframe index. 


    Location : struct

            *Location.latitude* - vector or scalar latitude in decimal degrees (positive is
                                  northern hemisphere)
            *Location.longitude* - vector or scalar longitude in decimal degrees (positive is 
                                  east of prime meridian)
            *Location.altitude* - an optional component of the Location struct, not
                                  used in the ephemeris code directly, but it may be used to calculate
                                  standard site pressure (see pvl_alt2pres function)
            *location.TZ*     - Time Zone offset from UTC 

    Other Parameters 
    ----------------

    LinkeTurbidityInput : Optional, float or DataFrame

                    An optional input to provide your own Linke
                    turbidity. If this input is omitted, the default Linke turbidity
                    maps will be used. LinkeTurbidityInput may be a float or 
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

    Vars=locals()
    Expect={'Time':(''),
            'Location':(''),
            'LinkeTurbidity':('optional')}
    var=pvl_tools.Parse(Vars,Expect)
            
    I0=pvl_extraradiation.pvl_extraradiation(var.Time.dayofyear)
    
    __,__,ApparentSunElevation,__,__=pvl_ephemeris.pvl_ephemeris(var.Time,var.Location,pvl_alt2pres.pvl_alt2pres(var.Location.altitude)) # nargout=4
    
    ApparentZenith=90 - ApparentSunElevation
    ApparentZenith[ApparentZenith>=90]=90
    
    

    if LinkeTurbidity==-999:

        # The .mat file 'LinkeTurbidities.mat' contains a single 2160 x 4320 x 12
        # matrix of type uint8 called 'LinkeTurbidity'. The rows represent global
        # latitudes from 90 to -90 degrees; the columns represent global longitudes
        # from -180 to 180; and the depth (third dimension) represents months of
        # the year from January (1) to December (12). To determine the Linke
        # turbidity for a position on the Earth's surface for a given month do the
        # following: LT = LinkeTurbidity(LatitudeIndex, LongitudeIndex, month).  Note that the numbers within the matrix are 20 * Linke
        # Turbidity, so divide the number from the file by 20 to get the
        # turbidity.
        mat = scipy.io.loadmat('LinkeTurbidities.mat')
        LinkeTurbidity=mat['LinkeTurbidity']
        LatitudeIndex=np.round_(LinearlyScale(Location.latitude,90,- 90,1,2160))
        LongitudeIndex=np.round_(LinearlyScale(Location.longitude,- 180,180,1,4320))
        g=LinkeTurbidity[LatitudeIndex][LongitudeIndex]
        ApplyMonth=lambda x:g[x[0]-1]
        LT=pd.DataFrame(Time.month)
        LT.index=Time
        LT=LT.apply(ApplyMonth,axis=1)
        TL=LT / float(20)
    else:
    
        TL=var.LinkeTurbidity

    # Get the absolute airmass assuming standard local pressure (per
    # pvl_alt2pres) using Kasten and Young's 1989 formula for airmass.

    AMabsolute=pvl_absoluteairmass.pvl_absoluteairmass(AMrelative=pvl_relativeairmass.pvl_relativeairmass(ApparentZenith,model='kastenyoung1989'),Pressure=pvl_alt2pres.pvl_alt2pres(var.Location.altitude))

    fh1=np.exp(var.Location.altitude*((- 1 / 8000)))
    fh2=np.exp(var.Location.altitude*((- 1 / 1250)))
    cg1=(5.09e-05*(var.Location.altitude) + 0.868)
    cg2=3.92e-05*(var.Location.altitude) + 0.0387

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

    #  This equation is found in Solar Energy 73, pg 311. It is slightly
    #  different than the equation given in Solar Energy 73, pg 156. We used the
    #  equation from pg 311 because of the existence of known typos in the pg 156
    #  publication (notably the fh2-(TL-1) should be fh2 * (TL-1)). 

    ClearSkyGHI=cg1*(I0)*(pvl_tools.cosd(ApparentZenith))*(np.exp(- cg2*(AMabsolute)*((fh1 + fh2*((TL - 1))))))*(np.exp(0.01*((AMabsolute) ** (1.8))))
    ClearSkyGHI[ClearSkyGHI < 0]=0

    b=0.664 + 0.163 / fh1
    BncI=b*(I0)*(np.exp(- 0.09*(AMabsolute)*((TL - 1))))

    ClearSkyDNI=np.min(BncI,ClearSkyGHI*((1 - (0.1 - 0.2*(np.exp(- TL))) / (0.1 + 0.882 / fh1))) / pvl_tools.cosd(ApparentZenith))
    
    #ClearSkyDNI=ClearSkyGHI*((1 - (0.1 - 0.2*(np.exp(- TL))) / (0.1 + 0.882 / fh1))) / pvl_tools.cosd(ApparentZenith)
    
    ClearSkyDHI=ClearSkyGHI - ClearSkyDNI*(pvl_tools.cosd(ApparentZenith))

    return ClearSkyGHI,ClearSkyDNI,ClearSkyDHI,BncI

def LinearlyScale(inputmatrix,inputmin,inputmax,outputmin,outputmax):
    inputrange=inputmax - inputmin
    outputrange=outputmax - outputmin
    OutputMatrix=(inputmatrix - inputmin) * outputrange / inputrange + outputmin
    return OutputMatrix
