# -*- coding: utf-8 -*-
"""
Created on Wed Aug 5 14:51:04 2020

@author: Saurabh

"""

from pvlib.temperature import sapm_cell
from pvlib.pvsystem import pvwatts_dc



def get_performance_ratio(poa_global, temp_air, wind_speed, pac, pdc0, a=-3.56, 
                          b=-0.075, deltaT=3, gamma_pdc=-0.00433):
    
    r'''
    Calculate NREL Performance Ratio.

    See equation [5] in Weather-Corrected Performance Ratio for details on 
    the weighted method for Tref.

    Parameters
    ----------
    poa_global : numeric
        Total incident irradiance [W/m^2].
        
    temp_air : numeric
        Ambient dry bulb temperature [C].
        
    wind_speed : numeric
        Wind speed at a height of 10 meters [m/s].

    pac : numeric
        AC power [kW].

    pdc0 : numeric
        Power of the modules at 1000 W/m2 and cell reference temperature.
        
    a : float
        Parameter :math:`a` in :eq:`sapm1mod`.

    b : float
        Parameter :math:`b` in :eq:`sapm1mod`.
        
    deltaT : float
        Parameter :math:`\Delta T` in :eq:`sapm2` [C].

    gamma_pdc : numeric
                The temperature coefficient in units of 1/C. Typically -0.002 
                to -0.005 per degree C.



    Returns
    -------
    performance_ratio: numeric
        Performance Ratio of data.


    References
    ----------
    .. [1] "Weather-Corrected Performance Ratio". NREL, 2013. 
    '''
    
    #GET TMOD/TCELL FROM EXISTING PVLIB TEMPERATURE FUNCTION
    cell_temperature = sapm_cell(poa_global, 
                                 temp_air, 
                                 wind_speed, 
                                 a, 
                                 b, 
                                 deltaT
                                 )
    
    #Get weighted Tcell average
    Tcell_poa_global = poa_global * cell_temperature
    Tref = Tcell_poa_global.sum() / poa_global.sum()
    
    
    #GET TEMPERATURE-CORRECTED DC ENERGY USING WEIGHTED TREF WITH PVLIB PVWATTS FUNCTION
    pdc = pvwatts_dc(poa_global, 
                     cell_temperature, 
                     pdc0, 
                     gamma_pdc, 
                     temp_ref=Tref
                     )

    performance_ratio = pac.sum() / pdc.sum()
    
    return performance_ratio