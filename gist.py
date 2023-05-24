import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt


modules = pvlib.pvsystem.retrieve_sam('CECMod')
cec_model = modules['Canadian_Solar_Inc__CS6P_235P']

STC = pd.DataFrame(
        {'effective_irradiance':
         np.array([1000.]),
         'temp_cell':
         np.array([25.])
         })

IEC61853 = pd.DataFrame(
        {'effective_irradiance':
         np.array([100., 100., 200., 200., 400., 400., 400.,
                   600., 600., 600., 600., 800., 800., 800., 800.,
                   1000., 1000., 1000., 1000., 1100., 1100., 1100.]),
         'temp_cell':
         np.array([15., 25., 15., 25., 15., 25., 50.,
                   15., 25., 50., 75., 15., 25., 50., 75.,
                   15., 25., 50., 75., 25., 50., 75.])
         })


# Constant irradiance and temperature
cec_stc_params_const = pvlib.pvsystem.calcparams_cec(
        1000., 25.,
        alpha_sc=cec_model['alpha_sc'], a_ref=cec_model['a_ref'],
        I_L_ref=cec_model['I_L_ref'], I_o_ref=cec_model['I_o_ref'],
        R_sh_ref=cec_model['R_sh_ref'], R_s=cec_model['R_s'],
        Adjust=cec_model['Adjust'])

# Single irradiance and temperature value but use Series
cec_stc_params_series = pvlib.pvsystem.calcparams_cec(
        STC['effective_irradiance'], STC['temp_cell'],
        alpha_sc=cec_model['alpha_sc'], a_ref=cec_model['a_ref'],
        I_L_ref=cec_model['I_L_ref'], I_o_ref=cec_model['I_o_ref'],
        R_sh_ref=cec_model['R_sh_ref'], R_s=cec_model['R_s'],
        Adjust=cec_model['Adjust'])

print('When Series are input, output from calcparams inherits names from input Series')
print(cec_stc_params_series)
print()

cec_iec_params = pvlib.pvsystem.calcparams_cec(
        IEC61853['effective_irradiance'], IEC61853['temp_cell'],
        alpha_sc=cec_model['alpha_sc'], a_ref=cec_model['a_ref'],
        I_L_ref=cec_model['I_L_ref'], I_o_ref=cec_model['I_o_ref'],
        R_sh_ref=cec_model['R_sh_ref'], R_s=cec_model['R_s'],
        Adjust=cec_model['Adjust'])

iv_stc = pvlib.pvsystem.singlediode(*cec_stc_params_const)
print('Irradiance and temperature are floats')
print('Returned items have different types')
for k, v in iv_stc.items():
    print(k, type(v))
print()

iv_stc_series = pvlib.pvsystem.singlediode(*cec_stc_params_series, ivcurve_pnts=100)
print('Irradiance and temperature are Series of length 1')
print('All returned items are numpy arrays, but v and i are 1x100 so the v, i '
      'plot is 100 Line2D objects')
for k, v in iv_stc_series.items():
    print(k, v.shape)
plt.figure()
plt.plot(iv_stc_series['v'], iv_stc_series['i'], '.')
print(' Have to transpose to get an IV curve plot')
plt.figure()
plt.plot(iv_stc_series['v'].T, iv_stc_series['i'].T, '.')
print()

iv_iec_series = pvlib.pvsystem.singlediode(*cec_iec_params)
print('Irradiance and temperature are Series with length > 1')
print('Returned items are columns in a DataFrame')
print(iv_iec_series.head())
print()
iv_iec_series_with_pnts = pvlib.pvsystem.singlediode(*cec_iec_params,
                                                     ivcurve_pnts=100)
print('Returns an OrderedDict')
print('IV curve data are by rows so must be transposted to plot')
for k, v in iv_iec_series_with_pnts.items():
    print(k, v.shape)
