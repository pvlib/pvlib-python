
import numpy as np
import pandas as pd 
import pdb
import pvl_tools

def pvl_snlinverter(Inverter,Vmp,Pmp):
  '''
  Converts DC power and voltage to AC power using Sandia's Grid-Connected PV Inverter model

  Determine the AC power output of an inverter given the DC voltage, DC
  power, and appropriate Sandia Grid-Connected Photovoltaic Inverter
  Model parameters. The output, ACPower, is clipped at the maximum power
  output, and gives a negative power during low-input power conditions,
  but does NOT account for maximum power point tracking voltage windows
  nor maximum current or voltage limits on the inverter. 

  Parameters
  ----------

  Inverter : DataFrame

           A DataFrame defining the inverter to be used, giving the
           inverter performance parameters according to the Sandia
           Grid-Connected Photovoltaic Inverter Model (SAND 2007-5036) [1]. A set of
           inverter performance parameters are provided with PV_LIB, or may be
           generated from a System Advisor Model (SAM) [2] library using pvl_retreivesam. 
           
            Required DataFrame components are:

           =============   ==============================================================================================================================================================================================
           Field            DataFrame
           =============   ==============================================================================================================================================================================================
           Inverter.Pac0   AC-power output from inverter based on input power and voltage, (W) 
           Inverter.Pdc0   DC-power input to inverter, typically assumed to be equal to the PV array maximum power, (W)
           Inverter.Vdc0   DC-voltage level at which the AC-power rating is achieved at the reference operating condition, (V)
           Inverter.Ps0    DC-power required to start the inversion process, or self-consumption by inverter, strongly influences inverter efficiency at low power levels, (W)
           Inverter.C0     Parameter defining the curvature (parabolic) of the relationship between ac-power and dc-power at the reference operating condition, default value of zero gives a linear relationship, (1/W)
           Inverter.C1     Empirical coefficient allowing Pdco to vary linearly with dc-voltage input, default value is zero, (1/V)
           Inverter.C2     empirical coefficient allowing Pso to vary linearly with dc-voltage input, default value is zero, (1/V)
           Inverter.C3     empirical coefficient allowing Co to vary linearly with dc-voltage input, default value is zero, (1/V)
           Inverter.Pnt    ac-power consumed by inverter at night (night tare) to maintain circuitry required to sense PV array voltage, (W)
           =============   ==============================================================================================================================================================================================
  
  Vdc : float or DataFrame
          DC voltages, in volts, which are provided as input to the inverter. Vdc must be >= 0.
  Pdc : float or DataFrame

          A scalar or DataFrame of DC powers, in watts, which are provided
           as input to the inverter. Pdc must be >= 0.

  Returns
  -------

  ACPower : float or DataFrame

           Mdeled AC power output given the input 
           DC voltage, Vdc, and input DC power, Pdc. When ACPower would be 
           greater than Pac0, it is set to Pac0 to represent inverter 
           "clipping". When ACPower would be less than Ps0 (startup power
           required), then ACPower is set to -1*abs(Pnt) to represent nightly 
           power losses. ACPower is not adjusted for maximum power point
           tracking (MPPT) voltage windows or maximum current limits of the
           inverter.

  References
  ----------

  [1] (SAND2007-5036, "Performance Model for Grid-Connected Photovoltaic 
  Inverters by D. King, S. Gonzalez, G. Galbraith, W. Boyson)

  [2] System Advisor Model web page. https://sam.nrel.gov.

  See also
  --------

  pvl_sapm
  pvl_samlibrary
  pvl_singlediode

  '''

  Vars=locals()
  Expect={'Inverter':(''),
      'Vmp':'',
      'Pmp':''}

  var=pvl_tools.Parse(Vars,Expect)

  Paco=var.Inverter['Paco']
  Pdco=var.Inverter['Pdco']
  Vdco=var.Inverter['Vdco']
  Pso=var.Inverter['Pso']
  C0=var.Inverter['C0']
  C1=var.Inverter['C1']
  C2=var.Inverter['C2']
  C3=var.Inverter['C3']
  Pnt=var.Inverter['Pnt']


  A=Pdco*((1 + C1*((var.Vmp - Vdco))))
  B=Pso*((1 + C2*((var.Vmp - Vdco))))
  C=C0*((1 + C3*((var.Vmp - Vdco))))
  ACPower=((Paco / (A - B)) - C*((A - B)))*((var.Pmp - B)) + C*((var.Pmp - B) ** 2)
  ACPower[ACPower > Paco]=Paco
  ACPower[ACPower < Pso]=- 1.0 * abs(Pnt)

  return ACPower
