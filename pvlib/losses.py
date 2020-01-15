# -*- coding: utf-8 -*-
"""
This module contains functions for losses of various types: soiling, mismatch,
snow cover, etc.
"""

import numpy as np
import pandas as pd
from pvlib.tools import cosd


def soiling_hsu(rainfall, cleaning_threshold, tilt, pm2_5, pm10,
                depo_veloc={'2_5': 0.004, '10': 0.0009},
                rain_accum_period=pd.Timedelta('1h')):
    """
    Calculates soiling ratio given particulate and rain data using the model
    from Humboldt State University [1]_.

    Parameters
    ----------

    rainfall : Series
        Rain accumulated in each time period. [mm]

    cleaning_threshold : float
        Amount of rain in an accumulation period needed to clean the PV
        modules. [mm]

    tilt : float
        Tilt of the PV panels from horizontal. [degree]

    pm2_5 : numeric
        Concentration of airborne particulate matter (PM) with
        aerodynamic diameter less than 2.5 microns. [g/m^3]

    pm10 : numeric
        Concentration of airborne particulate matter (PM) with
        aerodynamicdiameter less than 10 microns. [g/m^3]

    depo_veloc : dict, default {'2_5': 0.4, '10': 0.09}
        Deposition or settling velocity of particulates. [m/s]

    rain_accum_period : Timedelta, default 1 hour
        Period for accumulating rainfall to check against `cleaning_threshold`
        It is recommended that `rain_accum_period` be between 1 hour and
        24 hours.

    Returns
    -------
    soiling_ratio : Series
        Values between 0 and 1. Equal to 1 - transmission loss.

    References
    -----------
    .. [1] M. Coello and L. Boyle, "Simple Model For Predicting Time Series
       Soiling of Photovoltaic Panels," in IEEE Journal of Photovoltaics.
       doi: 10.1109/JPHOTOV.2019.2919628
    .. [2] Atmospheric Chemistry and Physics: From Air Pollution to Climate
       Change. J. Seinfeld and S. Pandis. Wiley and Sons 2001.

    """

    # accumulate rainfall into periods for comparison with threshold
    accum_rain = rainfall.rolling(rain_accum_period, closed='right').sum()
    # cleaning is True for intervals with rainfall greater than threshold
    cleaning_times = accum_rain.index[accum_rain >= cleaning_threshold]

    horiz_mass_rate = pm2_5 * depo_veloc['2_5']\
        + np.maximum(pm10 - pm2_5, 0.) * depo_veloc['10']
    tilted_mass_rate = horiz_mass_rate * cosd(tilt)  # assuming no rain

    # tms -> tilt_mass_rate
    tms_cumsum = np.cumsum(tilted_mass_rate * np.ones(rainfall.shape))

    mass_no_cleaning = pd.Series(index=rainfall.index, data=tms_cumsum)
    mass_removed = pd.Series(index=rainfall.index)
    mass_removed[0] = 0.
    mass_removed[cleaning_times] = mass_no_cleaning[cleaning_times]
    accum_mass = mass_no_cleaning - mass_removed.ffill()

    soiling_ratio = 1 - 0.3437 * np.erf(0.17 * accum_mass**0.8473)

    return soiling_ratio


def depo_velocity(T, WindSpeed, LUC):

    # convert temperature into Kelvin
    T = T + 273.15

    # save wind data
    if(np.isscalar(WindSpeed)):
        u = np.array([WindSpeed])
    else:
        u = WindSpeed

    g = 9.81         # gravity in m/s^2
    # Na = 6.022 * 10**23  # avagadros number
    R = 8.314        # Universal gas consant in m3Pa/Kmol
    k = 1.38 * 10**-23  # Boltzmann's constant in m^2kg/sK
    P = 101300       # pressure in Pa
    rhoair = 1.2041  # density of air in kg/m3
    z0 = 1
    rhop = 1500      # Assume density of particle in kg/m^3

    switcher = {
        1: 0.56,
        4: 0.56,
        6: 0.54,
        8: 0.54,
        10: 0.54,
    }

    try:
        gamma = switcher[LUC]
    except Exception as e:
        warnings.warn("Unknown Land Use Category, assuming LUC 8. "+str(e))
        LUC = 8
        gamma = switcher[LUC]

    # Diameter of particle in um
    Dpum = np.array([2.5, 10])
    Dpm = Dpum*10**-6   # Diameter of particle in m

    # Calculations
    mu = 1.8*10**-5*(T/298)**0.85      # viscosity of air in kg/m s
    nu = mu/rhoair
    lambda1 = 2*mu/(P*(8.*0.0288/(np.pi*R*T))**(0.5))   # mean free path
    ll = np.array([lambda1, lambda1])
    Cc = 1+2*ll/Dpm*(1.257+0.4*np.exp(-1.1*Dpm/(ll*2)))
    # slip correction coefficient

    # Calculate vs
    vs = rhop*Dpm**2*(g*Cc/(mu*18))  # particle settling velocity

    # Calculate rb
    ustar = np.zeros_like(u, dtype=float)  # pre-allocate ustar
    # Equation 11.66 in Ramaswami (and 16.67 and Sienfeld &Pandis)
    ustar[u > 0] = 0.4 * u[u > 0]/np.log(10/z0)
    ustar[u <= 0] = 0.001

    D = k*T*(Cc/(3*np.pi*mu*Dpm))

    Sc = nu/D
    # gamma=0.56      # for urban
    # alpha=1.5     # for urban
    EB = Sc**(-1 * gamma)
    St = vs*(ustar**2)/(g*nu)

    EIM = 10.0**(-3.0/St)   # For smooth surfaces
    # EIM =((St)./(0.82+St)).^2

    R1 = np.exp(-St**(0.5))  # percentage of particles that stick

    rb = 1/(3*(EB+EIM)*ustar*R1)

    # Calculate ra
    a = np.array([-0.096, -0.037, -0.002, 0, 0.004, 0.035])
    b = np.array([0.029, 0.029, 0.018, 0, -0.018, -0.036])

    # For wind speeds <= 3, use a = -0.037 and b = 0.029
    # For wind speeds >3 and <=5, use a = -.002, b = 0.018
    # For wind speeds > 5, use a = 0, b = 0
    avals = a[1]*np.ones_like(u, dtype=float)
    avals[u > 3] = a[2]
    avals[u > 5] = a[3]

    bvals = b[1]*np.ones_like(u, dtype=float)
    bvals[u > 3] = b[2]
    bvals[u > 5] = b[3]

    L = 1/(avals + bvals*np.log(z0))

    zeta0 = z0/L
    zeta = 10.0/L
    eta = ((1-15*zeta)**(0.25))
    eta0 = ((1-15*zeta0)**(0.25))

    ra = np.zeros_like(zeta, dtype=float)  # Preallocate memory
    ra[zeta == 0] = (1 / (0.4 * ustar[zeta == 0])) * np.log(10.0 / z0)

    sub_a = 1 / (0.4 * ustar[zeta > 0])
    sub_b = np.log(10.0 / z0)
    sub_c = zeta[zeta > 0] - zeta0[zeta > 0]

    ra[zeta > 0] = sub_a * (sub_b + 4.7 * sub_c)

    sub_a = (1 / (0.4 * ustar[zeta < 0]))
    sub_b = np.log(10.0 / z0)
    sub_ba = eta0[zeta < 0]**2 + 1
    sub_bb = (eta0[zeta < 0]+1)**2
    sub_bc = eta[zeta < 0]**2 + 1
    sub_bd = (eta[zeta < 0]+1)**2
    sub_d = np.arctan(eta[zeta < 0])
    sub_e = np.arctan(eta0[zeta < 0])

    sub_c = np.log(sub_ba * sub_bb / (sub_bc * sub_bd))
    ra[zeta < 0] = sub_a * (sub_b + sub_c + 2*(sub_d-sub_e))

    # Calculate vd and mass flux

    vd = 1/(ra+rb)+vs


    return vd
