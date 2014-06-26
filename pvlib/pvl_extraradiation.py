

import numpy as np
import pvl_tools as pvt

def pvl_extraradiation(doy, solar_constant=1366.1):
    '''
    Determine extraterrestrial radiation from day of year.
    
    The calculation method appears to be based on Spencer 1971.
    The primary references here cannot be easily obtained.
    
    If you really care about this parameter, consider using NREL's SOLPOS or 
    pyephem's earth_distance() function.

    Parameters
    ----------

    doy : int or pandas.index.dayofyear
        Day of the year

    Returns
    -------

    Ea : float or DataFrame

        the extraterrestrial radiation present in watts per square meter
        on a surface which is normal to the sun. Ea is of the same size as the
        input doy.


    References
    ----------
    <http://solardat.uoregon.edu/SolarRadiationBasics.html>, Eqs. SR1 and SR2

    SR1       Partridge, G. W. and Platt, C. M. R. 1976. Radiative Processes in Meteorology and Climatology.

    SR2       Duffie, J. A. and Beckman, W. A. 1991. Solar Engineering of Thermal Processes, 2nd edn. J. Wiley and Sons, New York.

    There is a minus sign discrepancy with equation 12 of 
    M. Reno, C. Hansen, and J. Stein, "Global Horizontal Irradiance Clear
    Sky Models: Implementation and Analysis", Sandia National
    Laboratories, SAND2012-2389, 2012. 
    
    See Also 
    --------
    pvl_disc

    '''
    
    B = 2 * np.pi * doy / 365
    RoverR0sqrd = 1.00011 + 0.034221*np.cos(B) + 0.00128*np.sin(B) + 0.000719*np.cos(2*B) + 7.7e-05*np.sin(2*B)
    Ea = solar_constant * RoverR0sqrd
    
    return Ea
    