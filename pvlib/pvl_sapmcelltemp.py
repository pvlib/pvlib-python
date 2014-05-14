

import numpy as np
import pandas as pd
import pvl_tools

def pvl_sapmcelltemp(E, Wspd, Tamb,modelt='Open_rack_cell_glassback',**kwargs):
  '''
  Estimate cell temperature from irradiance, windspeed, ambient temperature, and module parameters (SAPM)

  Estimate cell and module temperatures per the Sandia PV Array
  Performance model (SAPM, SAND2004-3535), when given the incident
  irradiance, wind speed, ambient temperature, and SAPM module
  parameters.

  Parameters
  ----------

  E : float or DataFrame
          Total incident irradiance in W/m^2. Must be >=0.


  windspeed : float or DataFrame
          Wind speed in m/s at a height of 10 meters. Must be >=0

  Tamb : float or DataFrame
          Ambient dry bulb temperature in degrees C. Must be >= -273.15.


  Other Parameters
  ----------------

  modelt :  string

  Model to be used for parameters, can be:

          * 'Open_rack_cell_glassback' (DEFAULT)
          * 'Roof_mount_cell_glassback'
          * 'Open_rack_cell_polymerback'
          * 'Insulated_back_polumerback'
          * 'Open_rack_Polymer_thinfilm_steel'
          * '22X_Concentrator_tracker'

  a : float (optional)
          SAPM module parameter for establishing the upper limit for module 
          temperature at low wind speeds and high solar irradiance (see SAPM
          eqn. 11). Must be a scalar.If not input, this value will be taken from the chosen
          model
  b : float (optional)

          SAPM module parameter for establishing the rate at which the module
          temperature drops as wind speed increases (see SAPM eqn. 11). Must be
          a scalar.If not input, this value will be taken from the chosen
          model

  deltaT : float (optional) 

          SAPM module parameter giving the temperature difference
          between the cell and module back surface at the reference irradiance,
          E0. Must be a numeric scalar >=0. If not input, this value will be taken from the chosen
          model

  Returns
  --------
  Tcell : float or DataFrame
          Cell temperatures in degrees C.
  
  Tmodule : float or DataFrame
          Module back temperature in degrees C.

  References
  ----------

  [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance Model", SAND Report
  3535, Sandia National Laboratories, Albuquerque, NM

  See Also 
  --------

  pvl_sapm
  '''
  Vars=locals()
  Expect={'a':('optional','num'),
          'b':('optional','num'),
          'deltaT':('optional','num'), 
          'E':('x>=0'),
          'Wspd':('x>=0'),
          'Tamb':('x>=0'),
          'modelt': ('default','default=Open_rack_cell_glassback')
          }

  var=pvl_tools.Parse(Vars,Expect)

  TempModel={'Open_rack_cell_glassback':[-3.47, -.0594, 3],
              'Roof_mount_cell_glassback':[-2.98, -.0471, 1],
              'Open_rack_cell_polymerback': [-3.56, -.0750, 3],
              'Insulated_back_polumerback': [-2.81, -.0455, 0 ],
              'Open_rack_Polymer_thinfilm_steel':[-3.58, -.113, 3],
              '22X_Concentrator_tracker':[-3.23, -.130, 13]
          }
  try: 
      a=var.a
      b=var.b
      deltaT=var.deltaT
  except:
      a=TempModel[var.modelt][0]
      b=TempModel[var.modelt][1]
      deltaT=TempModel[var.modelt][2]

  E0=1000 # Reference irradiance

  Tmodule=var.E*((np.exp(a + b*var.Wspd))) + var.Tamb

  Tcell=Tmodule + var.E / E0*(deltaT)

  return Tcell, Tmodule
