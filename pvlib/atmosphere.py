
import numpy as np
import pvl_tools as pvt

def pvl_pres2alt(pressure):
  '''
  Determine altitude from site pressure


  Parameters
  ----------
  Pressure : scalar, vector or DataFrame
            Atomspheric pressure (Pascals)

  Returns
  -------
  altitude: scalar, vector or DataFrame
          Altitude in meters above sea level 

  Notes
  ------
  The following assumptions are made

    ============================   ================
    Parameter                      Value
    ============================   ================
    Base pressure                  101325 Pa
    Temperature at zero altitude   288.15 K
    Gravitational acceleration     9.80665 m/s^2
    Lapse rate                     -6.5E-3 K/m
    Gas constant for air           287.053 J/(kgK)
    Relative Humidity               0%
    ============================   ================

  References
  -----------

  "A Quick Derivation relating altitude to air pressure" from Portland
  State Aerospace Society, Version 1.03, 12/22/2004.

  See also
  --------
  pvl_alt2pres ,pvl_makelocationstruct
  
  '''


  Vars=locals()
  Expect={'pressure': ('array', 'num', 'x>0')}

  var=pvt.Parse(Vars,Expect)
  Alt=44331.5 - 4946.62 * var.pressure ** (0.190263)
  return Alt
