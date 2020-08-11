# -*- coding: utf-8 -*-
"""
Created on Wed Aug 5 14:51:04 2020

@author: Saurabh

The following functions are from or based on NREL paper and existing equations 
already in PVLib

(poa_global, temp_air, wind_speed, a, b, deltaT,
              irrad_ref=1000)
"""

import pandas as pd
import numpy as np
from pvlib.temperature import sapm_module, sapm_cell
from pvlib.pvsystem import pvwatts_dc


#get dataframe prepared for testing with Portage data
df = pd.read_csv('Portage Solar_model.csv', index_col=0, parse_dates=True)

#inputs assumed will be in System object 
a_coeff = -3.56 #only needed to create Tmod_syn
b_coeff = -0.075 #only needed to create Tmod_syn
deltaT = 3
dc_cap = 1983.96
gamma_pdc = -0.00433
##################################################################################################
##################################################################################################



#Get weighted Tcell average using NREL paper, equation 5
def get_Tref(poa_global, cell_temperature):
    
    r'''
    Calculate weighted PV cell temperature .

    See equation [5] in Weather-Corrected Performance Ratio for details on 
    the weighted method.

    Parameters
    ----------
    poa_global : numeric
        Total incident irradiance [W/m^2].
        
    cell_temperature : numeric
        Temperature of back of module surface [C].


    Returns
    -------
    numeric, values in degrees C.


    References
    ----------
    .. [1] Dierauf..... (2013). "Weather-Corrected Performance Ratio" 
    '''
    
    Tcell_poa_global = poa_global * cell_temperature
    Tref = Tcell_poa_global.sum() / poa_global.sum()

    return Tref



#GET TMOD AND TCELL FROM EXISTING PVLIB TEMPERATURE FUNCTION
cell_temperature = sapm_cell(df['poa_global'], 
                             df['temp_air'], 
                             df['wind'], 
                             a_coeff, 
                             b_coeff, 
                             deltaT
                             )

#GET TMOD AND TCELL FROM EXISTING PVLIB TEMPERATURE FUNCTION
Tref = get_Tref(df['poa_global'],
                cell_temperature
                )

#GET TEMPERATURE-CORRECTED DC ENERGY USING WEIGHTED TREF WITH PVLIB PVWATTS FUNCTION
pdc = pvwatts_dc(df['poa_global'], 
                 cell_temperature, 
                 dc_cap, 
                 gamma_pdc, 
                 temp_ref=Tref
                 )


