

import numpy as np
import pvl_tools as pvt

def pvl_extraradiation(doy):
  '''
  Determine extraterrestrial radiation from day of year

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

  See Also 
  --------
  pvl_disc

  '''
  Vars=locals()
  Expect={'doy': ('array','num','x>=1','x<367')}

  var=pvt.Parse(Vars,Expect)

  B=2 * np.pi * var.doy / 365
  Rfact2=1.00011 + 0.034221*(np.cos(B)) + 0.00128*(np.sin(B)) + 0.000719*(np.cos(2 * B)) + 7.7e-05*(np.sin(2 * B))
  Ea=1367 * Rfact2
  return Ea
