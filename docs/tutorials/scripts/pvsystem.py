# coding: utf-8

# # pvsystem tutorial

# This tutorial explores the ``pvlib.pvsystem`` module. The module has functions for importing PV module and inverter data and functions for modeling module and inverter performance.
# 
# 1. [systemdef](#systemdef)
# 2. [Angle of Incidence Modifiers](#Angle-of-Incidence-Modifiers)
# 2. [Sandia Cell Temp correction](#Sandia-Cell-Temp-correction)
# 2. [Sandia Inverter Model](#snlinverter)
# 2. [Sandia Array Performance Model](#SAPM)
#     1. [SAPM IV curves](#SAPM-IV-curves)
# 2. [DeSoto Model](#desoto)
# 2. [Single Diode Model](#Single-diode-model)
# 
# This tutorial has been tested against the following package versions:
# * pvlib 0.2.0
# * Python 2.7.10
# * IPython 3.2
# * Pandas 0.16.2
# 
# It should work with other Python and Pandas versions. It requires pvlib >= 0.2.0 and IPython >= 3.0.
# 
# Authors:
# * Will Holmgren (@wholmgren), University of Arizona. 2015.
# built-in python modules
import os
import inspect
import datetime

# scientific python add-ons
import numpy as np
import pandas as pd

# plotting stuff
# first line makes the plots appear in the notebook
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
# seaborn makes your plots look better
try:
    import seaborn as sns
    sns.set(rc={"figure.figsize": (12, 6)})
except ImportError:
    print('We suggest you install seaborn using conda or pip and rerun this cell')

# finally, we import the pvlib library
import pvlib

import pvlib
from pvlib import pvsystem


# ### systemdef

# ``pvlib`` can import TMY2 and TMY3 data. Here, we import the example files.
pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))

tmy3_data, tmy3_metadata = pvlib.tmy.readtmy3(os.path.join(pvlib_abspath, 'data', '703165TY.csv'))
tmy2_data, tmy2_metadata = pvlib.tmy.readtmy2(os.path.join(pvlib_abspath, 'data', '12839.tm2'))

pvlib.pvsystem.systemdef(tmy3_metadata, 0, 0, .1, 5, 5)

pvlib.pvsystem.systemdef(tmy2_metadata, 0, 0, .1, 5, 5)


# ### Angle of Incidence Modifiers
angles = np.linspace(-180,180,3601)
ashraeiam = pd.Series(pvsystem.ashraeiam(.05, angles), index=angles)

ashraeiam.plot()
plt.ylabel('ASHRAE modifier')
plt.xlabel('input angle (deg)')

angles = np.linspace(-180,180,3601)
physicaliam = pd.Series(pvsystem.physicaliam(4, 0.002, 1.526, angles), index=angles)

physicaliam.plot()
plt.ylabel('physical modifier')
plt.xlabel('input index')

plt.figure()
ashraeiam.plot(label='ASHRAE')
physicaliam.plot(label='physical')
plt.ylabel('modifier')
plt.xlabel('input angle (deg)')
plt.legend()


# ### Sandia Cell Temp correction

# PV system efficiency can vary by up to 0.5% per degree C, so it's important to accurately model cell and module temperature. The ``sapm_celltemp`` function uses plane of array irradiance, ambient temperature, wind speed, and module and racking type to calculate cell and module temperatures. From King et. al. (2004):
# 
# $$T_m = E e^{a+b*WS} + T_a$$
# 
# $$T_c = T_m + \frac{E}{E_0} \Delta T$$
# 
# The $a$, $b$, and $\Delta T$ parameters depend on the module and racking type. The default parameter set is ``open_rack_cell_glassback``.

# ``sapm_celltemp`` works with either scalar or vector inputs, but always returns a pandas DataFrame.
# scalar inputs
pvsystem.sapm_celltemp(900, 5, 20) # irrad, wind, temp

# vector inputs
times = pd.DatetimeIndex(start='2015-01-01', end='2015-01-02', freq='12H')
temps = pd.Series([0, 10, 5], index=times)
irrads = pd.Series([0, 500, 0], index=times)
winds = pd.Series([10, 5, 0], index=times)

pvtemps = pvsystem.sapm_celltemp(irrads, winds, temps)
pvtemps.plot()


# Cell and module temperature as a function of wind speed.
wind = np.linspace(0,20,21)
temps = pd.DataFrame(pvsystem.sapm_celltemp(900, wind, 20), index=wind)

temps.plot()
plt.legend()
plt.xlabel('wind speed (m/s)')
plt.ylabel('temperature (deg C)')


# Cell and module temperature as a function of ambient temperature.
atemp = np.linspace(-20,50,71)
temps = pvsystem.sapm_celltemp(900, 2, atemp).set_index(atemp)

temps.plot()
plt.legend()
plt.xlabel('ambient temperature (deg C)')
plt.ylabel('temperature (deg C)')


# Cell and module temperature as a function of incident irradiance.
irrad = np.linspace(0,1000,101)
temps = pvsystem.sapm_celltemp(irrad, 2, 20).set_index(irrad)

temps.plot()
plt.legend()
plt.xlabel('incident irradiance (W/m**2)')
plt.ylabel('temperature (deg C)')


# Cell and module temperature for different module and racking types.
models = ['open_rack_cell_glassback',
          'roof_mount_cell_glassback',
          'open_rack_cell_polymerback',
          'insulated_back_polymerback',
          'open_rack_polymer_thinfilm_steel',
          '22x_concentrator_tracker']

temps = pd.DataFrame(index=['temp_cell','temp_module'])

for model in models:
    temps[model] = pd.Series(pvsystem.sapm_celltemp(1000, 5, 20, model=model).ix[0])

temps.T.plot(kind='bar') # try removing the transpose operation and replotting
plt.legend()
plt.ylabel('temperature (deg C)')


# ### snlinverter
inverters = pvsystem.retrieve_sam('sandiainverter')
inverters

vdcs = pd.Series(np.linspace(0,50,51))
idcs = pd.Series(np.linspace(0,11,110))
pdcs = idcs * vdcs

pacs = pvsystem.snlinverter(inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'], vdcs, pdcs)
#pacs.plot()
plt.plot(pacs, pdcs)
plt.ylabel('ac power')
plt.xlabel('dc power')


# Need to put more effort into describing this function.

# ### SAPM

# The CEC module database.
cec_modules = pvsystem.retrieve_sam('cecmod')
cec_modules

cecmodule = cec_modules.Example_Module 
cecmodule


# The Sandia module database.
sandia_modules = pvsystem.retrieve_sam(name='SandiaMod')
sandia_modules

sandia_module = sandia_modules.Canadian_Solar_CS5P_220M___2009_
sandia_module


# Generate some irradiance data for modeling.
from pvlib import clearsky
from pvlib import irradiance
from pvlib import atmosphere
from pvlib.location import Location

tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
times = pd.date_range(start=datetime.datetime(2014,4,1), end=datetime.datetime(2014,4,2), freq='30s')
ephem_data = pvlib.solarposition.get_solarposition(times, tus)
irrad_data = clearsky.ineichen(times, tus)
#irrad_data.plot()

aoi = irradiance.aoi(0, 0, ephem_data['apparent_zenith'], ephem_data['azimuth'])
#plt.figure()
#aoi.plot()

am = atmosphere.relativeairmass(ephem_data['apparent_zenith'])

# a hot, sunny spring day in the desert.
temps = pvsystem.sapm_celltemp(irrad_data['ghi'], 0, 30)


# Now we can run the module parameters and the irradiance data through the SAPM function.
sapm_1 = pvsystem.sapm(sandia_module, irrad_data['dni']*np.cos(np.radians(aoi)),
                     irrad_data['ghi'], temps['temp_cell'], am, aoi)
sapm_1.head()

def plot_sapm(sapm_data):
    """
    Makes a nice figure with the SAPM data.
    
    Parameters
    ----------
    sapm_data : DataFrame
        The output of ``pvsystem.sapm``
    """
    fig, axes = plt.subplots(2, 3, figsize=(16,10), sharex=False, sharey=False, squeeze=False)
    plt.subplots_adjust(wspace=.2, hspace=.3)

    ax = axes[0,0]
    sapm_data.filter(like='i_').plot(ax=ax)
    ax.set_ylabel('Current (A)')

    ax = axes[0,1]
    sapm_data.filter(like='v_').plot(ax=ax)
    ax.set_ylabel('Voltage (V)')

    ax = axes[0,2]
    sapm_data.filter(like='p_').plot(ax=ax)
    ax.set_ylabel('Power (W)')

    ax = axes[1,0]
    [ax.plot(sapm_data['effective_irradiance'], current, label=name) for name, current in
     sapm_data.filter(like='i_').iteritems()]
    ax.set_ylabel('Current (A)')
    ax.set_xlabel('Effective Irradiance')
    ax.legend(loc=2)

    ax = axes[1,1]
    [ax.plot(sapm_data['effective_irradiance'], voltage, label=name) for name, voltage in
     sapm_data.filter(like='v_').iteritems()]
    ax.set_ylabel('Voltage (V)')
    ax.set_xlabel('Effective Irradiance')
    ax.legend(loc=4)

    ax = axes[1,2]
    ax.plot(sapm_data['effective_irradiance'], sapm_data['p_mp'], label='p_mp')
    ax.set_ylabel('Power (W)')
    ax.set_xlabel('Effective Irradiance')
    ax.legend(loc=2)

    # needed to show the time ticks
    for ax in axes.flatten():
        for tk in ax.get_xticklabels():
            tk.set_visible(True)

plot_sapm(sapm_1)


# For comparison, here's the SAPM for a sunny, windy, cold version of the same day.
temps = pvsystem.sapm_celltemp(irrad_data['ghi'], 10, 5)

sapm_2 = pvsystem.sapm(sandia_module, irrad_data['dni']*np.cos(np.radians(aoi)),
                     irrad_data['dhi'], temps['temp_cell'], am, aoi)

plot_sapm(sapm_2)

sapm_1['p_mp'].plot(label='30 C,  0 m/s')
sapm_2['p_mp'].plot(label=' 5 C, 10 m/s')
plt.legend()
plt.ylabel('Pmp')
plt.title('Comparison of a hot, calm day and a cold, windy day')


# #### SAPM IV curves

# The IV curve function only calculates the 5 points of the SAPM. We will add arbitrary points in a future release, but for now we just interpolate between the 5 SAPM points.
import warnings
warnings.simplefilter('ignore', np.RankWarning)

def sapm_to_ivframe(sapm_row):
    pnt = sapm_row.T.ix[:,0]

    ivframe = {'Isc': (pnt['i_sc'], 0),
              'Pmp': (pnt['i_mp'], pnt['v_mp']),
              'Ix': (pnt['i_x'], 0.5*pnt['v_oc']),
              'Ixx': (pnt['i_xx'], 0.5*(pnt['v_oc']+pnt['v_mp'])),
              'Voc': (0, pnt['v_oc'])}
    ivframe = pd.DataFrame(ivframe, index=['current', 'voltage']).T
    ivframe = ivframe.sort('voltage')
    
    return ivframe

def ivframe_to_ivcurve(ivframe, points=100):
    ivfit_coefs = np.polyfit(ivframe['voltage'], ivframe['current'], 30)
    fit_voltages = np.linspace(0, ivframe.ix['Voc', 'voltage'], points)
    fit_currents = np.polyval(ivfit_coefs, fit_voltages)
    
    return fit_voltages, fit_currents

sapm_to_ivframe(sapm_1['2014-04-01 10:00:00'])

times = ['2014-04-01 07:00:00', '2014-04-01 08:00:00', '2014-04-01 09:00:00', 
         '2014-04-01 10:00:00', '2014-04-01 11:00:00', '2014-04-01 12:00:00']
times.reverse()

fig, ax = plt.subplots(1, 1, figsize=(12,8))

for time in times:
    ivframe = sapm_to_ivframe(sapm_1[time])

    fit_voltages, fit_currents = ivframe_to_ivcurve(ivframe)

    ax.plot(fit_voltages, fit_currents, label=time)
    ax.plot(ivframe['voltage'], ivframe['current'], 'ko')
    
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Current (A)')
ax.set_ylim(0, None)
ax.set_title('IV curves at multiple times')
ax.legend()


# ### desoto

# The same data run through the desoto model.
photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = (
    pvsystem.calcparams_desoto(irrad_data.ghi,
                                 temp_cell=temps['temp_cell'],
                                 alpha_isc=cecmodule['alpha_sc'],
                                 module_parameters=cecmodule,
                                 EgRef=1.121,
                                 dEgdT=-0.0002677) )

photocurrent.plot()
plt.ylabel('Light current I_L (A)')

saturation_current.plot()
plt.ylabel('Saturation current I_0 (A)')

resistance_series

resistance_shunt.plot()
plt.ylabel('Shunt resistance (ohms)')
plt.ylim(0,100)

nNsVth.plot()
plt.ylabel('nNsVth')


# ### Single diode model
single_diode_out = pvsystem.singlediode(cecmodule, photocurrent, saturation_current,
                                        resistance_series, resistance_shunt, nNsVth)
single_diode_out

single_diode_out['i_sc'].plot()

single_diode_out['v_oc'].plot()

single_diode_out['p_mp'].plot()



