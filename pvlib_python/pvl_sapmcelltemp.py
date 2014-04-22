 '''
 PVL_SAPMCELLTEMP Estimate cell temperature from irradiance, windspeed, ambient temperature, and module parameters (SAPM)

 Syntax
   [Tcell Tmodule] = pvl_sapmcelltemp(E, E0, windspeed, Tamb, model, a, b, deltaT)

 Description
   Estimate cell and module temperatures per the Sandia PV Array
   Performance model (SAPM, SAND2004-3535), when given the incident
   irradiance, wind speed, ambient temperature, and SAPM module
   parameters.

 Inputs
   E - Total incident irradiance in W/m^2. E must be a scalar or a vector
     of the same size as windspeed, and Tamb. Must be >=0.
   E0 - Reference irradiance used when determining delta T, in W/m^2. E0
     must be a scalar. Must be >=0;

   windspeed - Wind speed in m/s at a height of 10 meters. windspeed must
     be a scalar or a vector of the same size as E and Tamb. Must be >=0;
   Tamb - Ambient dry bulb temperature in degrees C. Tamb must be a scalar
     or a vector of the same size as windspeed and E. Must be >= -273.15.
   modelt- Model to be used for parameters: 
                'Open_rack_cell_glassback':
                'Roof_mount_cell_glassback'
                'Open_rack_cell_polymerback'
                'Insulated_back_polumerback'
                'Open_rack_Polymer_thinfilm_steel'
                '22X_Concentrator_tracker'

   a - (optional)SAPM module parameter for establishing the upper limit for module 
     temperature at low wind speeds and high solar irradiance (see SAPM
     eqn. 11). Must be a scalar.If not input, this value will be taken from the chosen
     model
   b - (optional)SAPM module parameter for establishing the rate at which the module
     temperature drops as wind speed increases (see SAPM eqn. 11). Must be
     a scalar.If not input, this value will be taken from the chosen
     model
   deltaT - (optional) SAPM module parameter giving the temperature difference
     between the cell and module back surface at the reference irradiance,
     E0. Must be a numeric scalar >=0. If not input, this value will be taken from the chosen
     model
  
 Outputs
   Tcell - A column vector of cell temperatures in degrees C.
   Tmodule - A column vector of module back temperature in degrees C.

 References
   [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance Model", SAND Report
   3535, Sandia National Laboratories, Albuquerque, NM

 See also PVL_SAPM
'''

import numpy as np
import pandas as pd
import pvl_tools

def pvl_sapmcelltemp(E, E0, windspeed, Tamb,modelt='Open_rack_cell_glassback',**kwargs):
    Vars=locals()
    Expect={'a':('optional','num'),
            'b':('optional','num'),
            'deltaT':('optional','num'), 
            'E':('x>=0'),
            'Wspd':('x>=0'),
            'DryBulb':('x>=0'),
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

    Tmodule=var.E*((np.exp(a + b*var.Wspd))) + var.DryBulb

    Tcell=Tmodule + var.E / E0*(deltaT)
    
    return Tcell, Tmodule
