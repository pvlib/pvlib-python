
import numpy as np
import pvl_tools as pvt

def pres2alt(pressure):
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

def alt2pres(altitude):
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

def absoluteairmass(AMrelative,Pressure):
  '''
  Determine absolute (pressure corrected) airmass from relative airmass and pressure

  Gives the airmass for locations not at sea-level (i.e. not at standard
  pressure). The input argument "AMrelative" is the relative airmass. The
  input argument "pressure" is the pressure (in Pascals) at the location
  of interest and must be greater than 0. The calculation for
  absolute airmass is:
  absolute airmass = (relative airmass)*pressure/101325

  Parameters
  ----------

  AMrelative : float or DataFrame
  
        The airmass at sea-level which can be calculated using the 
        PV_LIB function pvl_relativeairmass. 
  
  pressure : float or DataFrame

        a scalar or vector of values providing the site pressure in
        Pascal. If pressure is a vector it must be of the same size as all
        other vector inputs. pressure must be >=0. Pressure may be measured
        or an average pressure may be calculated from site altitude.

  Returns
  -------

  AMa : float or DataFrame

        Absolute (pressure corrected) airmass

  References
  ----------

  [1] C. Gueymard, "Critical analysis and performance assessment of 
  clear sky solar irradiance models using theoretical and measured data,"
  Solar Energy, vol. 51, pp. 121-138, 1993.

  See also 
  ---------
  pvl_relativeairmass 

  '''
  
  Vars=locals()
  Expect={'AMrelative': ('array','num'),
      'Pressure': ('array', 'num', 'x>0')}

  var=pvt.Parse(Vars,Expect)
  
  AMa=var.AMrelative.dot(var.Pressure) / 101325
  
  return AMa



def relativeairmass(z,model='kastenyoung1989'):
  '''
  Gives the relative (not pressure-corrected) airmass

    Gives the airmass at sea-level when given a sun zenith angle, z (in 
    degrees). 
    The "model" variable allows selection of different airmass models
    (described below). "model" must be a valid string. If "model" is not 
    included or is not valid, the default model is 'kastenyoung1989'.

  Parameters
  ----------

  z : float or DataFrame 

      Zenith angle of the sun.  Note that some models use the apparent (refraction corrected)
      zenith angle, and some models use the true (not refraction-corrected)
      zenith angle. See model descriptions to determine which type of zenith
      angle is required.
      
  model : String 
      Avaiable models include the following:

         * 'simple' - secant(apparent zenith angle) - Note that this gives -inf at zenith=90
         * 'kasten1966' - See reference [1] - requires apparent sun zenith
         * 'youngirvine1967' - See reference [2] - requires true sun zenith
         * 'kastenyoung1989' - See reference [3] - requires apparent sun zenith
         * 'gueymard1993' - See reference [4] - requires apparent sun zenith
         * 'young1994' - See reference [5] - requries true sun zenith
         * 'pickering2002' - See reference [6] - requires apparent sun zenith

  Returns
  -------
    AM : float or DataFrame 
            Relative airmass at sea level.  Will return NaN values for all zenith 
            angles greater than 90 degrees.

  References
  ----------

  [1] Fritz Kasten. "A New Table and Approximation Formula for the
  Relative Optical Air Mass". Technical Report 136, Hanover, N.H.: U.S.
  Army Material Command, CRREL.

  [2] A. T. Young and W. M. Irvine, "Multicolor Photoelectric Photometry
  of the Brighter Planets," The Astronomical Journal, vol. 72, 
  pp. 945-950, 1967.

  [3] Fritz Kasten and Andrew Young. "Revised optical air mass tables and
  approximation formula". Applied Optics 28:4735-4738

  [4] C. Gueymard, "Critical analysis and performance assessment of 
  clear sky solar irradiance models using theoretical and measured data,"
  Solar Energy, vol. 51, pp. 121-138, 1993.

  [5] A. T. Young, "AIR-MASS AND REFRACTION," Applied Optics, vol. 33, 
  pp. 1108-1110, Feb 1994.

  [6] Keith A. Pickering. "The Ancient Star Catalog". DIO 12:1, 20,

  See Also
  --------
  pvl_absoluteairmass
  pvl_ephemeris

  '''
  Vars=locals()
  Expect={'z': ('array','num','x<=90','x>=0'),
      'model': ('default','default=kastenyoung1989')}
  var=pvt.Parse(Vars,Expect)

  if ('kastenyoung1989') == var.model.lower():
      AM=1.0 / (np.cos(np.radians(var.z)) + 0.50572*(((6.07995 + (90 - var.z)) ** - 1.6364)))
  else:
      if ('kasten1966') == var.model.lower():
          AM=1.0 / (np.cos(np.radians(var.z)) + 0.15*((93.885 - var.z) ** - 1.253))
      else:
          if ('simple') == var.model.lower():
              AM=np.sec(np.radians(var.z))
          else:
              if ('pickering2002') == var.model.lower():
                  AM=1.0 / (np.sin(np.radians(90 - var.z + 244.0 / (165 + 47.0 * (90 - var.z) ** 1.1))))
              else:
                  if ('youngirvine1967') == var.model.lower():
                      AM=1.0 / np.cos(np.radians(var.z))*((1 - 0.0012*((((np.sec(np.radians(var.z)) ** 2) - 1)))))
                  else:
                      if ('young1994') == var.model.lower():
                          AM=(1.002432*((np.cos(np.radians(var.z))) ** 2) + 0.148386*(np.cos(np.radians(var.z))) + 0.0096467) / (np.cos(np.radians(var.z)) ** 3 + 0.149864*(np.cos(np.radians(var.z)) ** 2) + 0.0102963*(np.cos(np.radians(var.z))) + 0.000303978)
                      else:
                          if ('gueymard1993') == var.model.lower():
                              AM=1.0 / (np.cos(np.radians(var.z)) + 0.00176759*(var.z)*((94.37515 - var.z) ** - 1.21563))
                          else:
                              print(var.model + " is not a valid model type for relative airmass. The 'kastenyoung1989' model was used.")
                              AM=1.0 / (np.cos(np.radians(var.z)) + 0.50572*(((6.07995 + (90 - var.z)) ** - 1.6364)))
  return AM




