
import numpy as np
import pvl_tools as pvt
import pdb
def pvl_alt2pres(altitude):
  '''
  Determine site pressure from altitude

  Parameters
  ----------
  Altitude: scalar, vector or DataFrame
          Altitude in meters above sea level 

  Returns
  -------
  Pressure : scalar, vector or DataFrame
            Atomspheric pressure (Pascals)

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
  Expect={'altitude': 'num'}
  var=pvt.Parse(Vars,Expect)

  Press=100 * ((44331.514 - var.altitude) / 11880.516) ** (1 / 0.1902632)

  return Press
