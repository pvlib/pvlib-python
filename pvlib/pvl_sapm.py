

import numpy as Np
import pvl_tools
import pandas as pd

def pvl_sapm(Module,Eb,Ediff,Tcell,AM,AOI):
  '''
  Performs Sandia PV Array Performance Model to get 5 points on IV curve given SAPM module parameters, Ee, and cell temperature

  The Sandia PV Array Performance Model (SAPM) generates 5 points on a PV
  module's I-V curve (Voc, Isc, Ix, Ixx, Vmp/Imp) according to
  SAND2004-3535. Assumes a reference cell temperature of 25 C.
  
  parameters
  ----------

  Module : DataFrame

          A DataFrame defining the SAPM performance parameters (see
          pvl_retreivesam)

  Eb : float of DataFrame

          The effective irradiance incident upon the module (suns). Any Ee<0
          are set to 0.
  
  celltemp : float of DataFrame
          
          The cell temperature (degrees C)

  Returns
  -------
  Result - DataFrame

          A DataFrame with:

          * Result.Isc
          * Result.Imp
          * Result.Ix
          * Result.Ixx
          * Result.Voc
          * Result.Vmp
          * Result.Pmp

  Notes
  -----

  The particular coefficients from SAPM which are required in Module
  are:
  
  ================   ======================================================================================================================
  Module field        Description
  ================   ======================================================================================================================
  Module.c           1x8 vector with the C coefficients Module.c(1) = C0
  Module.Isc0        Short circuit current at reference condition (amps)
  Module.Imp0        Maximum power current at reference condition (amps)
  Module.AlphaIsc    Short circuit current temperature coefficient at reference condition (1/C)
  Module.AlphaImp    Maximum power current temperature coefficient at reference condition (1/C)
  Module.BetaVoc     Open circuit voltage temperature coefficient at reference condition (V/C)
  Module.mBetaVoc    Coefficient providing the irradiance dependence for the BetaVoc temperature coefficient at reference irradiance (V/C)
  Module.BetaVmp     Maximum power voltage temperature coefficient at reference condition
  Module.mBetaVmp    Coefficient providing the irradiance dependence for the BetaVmp temperature coefficient at reference irradiance (V/C)
  Module.n           Empirically determined "diode factor" (dimensionless)
  Module.Ns          Number of cells in series in a module's cell string(s)
  ================   ======================================================================================================================
  
  References
  ----------

  [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance Model", SAND Report
  3535, Sandia National Laboratories, Albuquerque, NM

  See Also
  --------

  pvl_retreivesam
  pvl_sapmcelltemp 
  
  '''
  Vars=locals()
  Expect={'Module':(''),
          'Eb':('x>0'),
          'Ediff':('x>0'),
          'Tcell':('x>0'),
          'AM':('x>0'),
          'AOI':('x>0')
  }
  var=pvl_tools.Parse(Vars,Expect)

  T0=25
  q=1.60218e-19
  k=1.38066e-23
  E0=1000

  AMcoeff=[var.Module['A4'],var.Module['A3'],var.Module['A2'],var.Module['A1'],var.Module['A0']]
  AOIcoeff=[var.Module['B5'],var.Module['B4'],var.Module['B3'],var.Module['B2'],var.Module['B1'],var.Module['B0']]

  F1 = Np.polyval(AMcoeff,var.AM)
  F2 = Np.polyval(AOIcoeff,var.AOI)
  var.Ee= F1*((var.Eb*F2+var.Module['FD']*var.Ediff)/E0)
  #var['Ee']=F1*((var.Eb+var.Ediff)/E0)
  #print "Ee modifed, revert for main function"
  var.Ee.fillna(0)
  var.Ee[var.Ee < 0]=0

  Filt=var.Ee[var.Ee >= 0.001]

  Isc=var.Module.ix['Isco']*(var.Ee)*((1 + var.Module.ix['Aisc']*((var.Tcell - T0))))

  DFOut=pd.DataFrame({'Isc':Isc})

  DFOut['Imp']=var.Module.ix['Impo']*((var.Module.ix['C0']*(var.Ee) + var.Module.ix['C1'] * (var.Ee ** 2)))*((1 + var.Module.ix['Aimp']*((var.Tcell - T0))))
  Bvoco=var.Module.ix['Bvoco'] + var.Module.ix['Mbvoc']*((1 - var.Ee))
  delta=var.Module.ix['N']*(k)*((var.Tcell + 273.15)) / q
  DFOut['Voc']=(var.Module.ix['Voco'] + var.Module.ix['#Series']*(delta)*(Np.log(var.Ee)) + Bvoco*((var.Tcell - T0)))
  Bvmpo=var.Module.ix['Bvmpo'] + var.Module.ix['Mbvmp']*((1 - var.Ee))
  DFOut['Vmp']=(var.Module.ix['Vmpo'] + var.Module.ix['C2']*(var.Module.ix['#Series'])*(delta)*(Np.log(var.Ee)) + var.Module.ix['C3']*(var.Module.ix['#Series'])*((delta*(Np.log(var.Ee))) ** 2) + Bvmpo*((var.Tcell - T0)))
  DFOut['Vmp'][DFOut['Vmp']<0]=0
  DFOut['Pmp']=DFOut.Imp*DFOut.Vmp
  DFOut['Ix']=var.Module.ix['IXO'] * (var.Module.ix['C4']*(var.Ee) + var.Module.ix['C5']*((var.Ee) ** 2))*((1 + var.Module.ix['Aisc']*((var.Tcell - T0))))
  DFOut['Ixx']=var.Module.ix['IXXO'] * (var.Module.ix['C6']*(var.Ee) + var.Module.ix['C7']*((var.Ee) ** 2))*((1 + var.Module.ix['Aisc']*((var.Tcell - T0))))

  return  DFOut
