 '''
 PVL_ALT2PRES Determine site pressure from altitude

 Syntax
   Press = pvl_alt2pres(altitude)
 
 Description
   PVL_ALT2PRES determines the atmospheric pressure (in Pascals) of a 
   site on Earth's surface given its altitude (in meters above sea level). 
   Output "Press" is given in Pascals. Press is of the same size 
   as altitude.

 Assumptions include:
   Base pressure = 101325 Pa
   Temperature at zero altitude = 288.15 K
   Gravitational acceleration = 9.80665 m/s^2
   Lapse rate = -6.5E-3 K/m
   Gas constant for air = 287.053 J/(kg*K)
   Relative Humidity = 0
 
 Inputs:

   altitude - altitude (in meters above sea level)

 Outputs:
   pressure - atmospheric pressure (in Pascals) of a 
              site on Earth's surface given its altitude

 References:
   "A Quick Derivation relating altitude to air pressure" from Portland
   State Aerospace Society, Version 1.03, 12/22/2004.

 See also PVL_PRES2ALT PVL_MAKELOCATIONSTRUCT
'''
import numpy as np
import pvl_tools as pvt

def pvl_alt2pres(altitude):
	Vars=locals()
    Expect={'altitude', 'num'}
    var=pvl_tools.Parse(Vars,Expect)

    Press=100 * ((44331.514 - var.altitude) / 11880.516) ** (1 / 0.1902632)
    
    return Press
