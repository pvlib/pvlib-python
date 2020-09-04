# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:53:11 2020

@author: cliff
"""

import pandas as pd
import numpy as np
import itertools

from scipy.optimize import minimize

from pvlib.pvsystem import snlinverter

params = {'Paco': 1000., 'Pdco': 1050., 'Vdco': 240., 'Pso': 10., 'C0': 1e-6,
          'C1': 1e-4, 'C2': 1e-2, 'C3': 1e-3, 'Pnt': 1}

power_levels = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
v_levels = {'Vmin': 220., 'Vnom': 240., 'Vmax': 260.}

num = len(power_levels) * len(v_levels)
res = pd.DataFrame(index=range(num), columns=['fraction_of_rated_power',
                   'efficiency', 'dc_voltage_level', 'pdc', 'pac'])


def obj(pdc, pac, vdc, inverter):
    Paco = inverter['Paco']
    Pdco = inverter['Pdco']
    Vdco = inverter['Vdco']
    Pso = inverter['Pso']
    C0 = inverter['C0']
    C1 = inverter['C1']
    C2 = inverter['C2']
    C3 = inverter['C3']

    A = Pdco * (1 + C1*(vdc - Vdco))
    B = Pso * (1 + C2*(vdc - Vdco))
    C = C0 * (1 + C3*(vdc - Vdco))

    ac_power = (Paco/(A-B) - C*(A-B)) * (pdc-B) + C*((pdc-B)**2)

    return (pac - ac_power)**2.


for idx, pair in enumerate(itertools.product(power_levels, v_levels.keys())):
    plev, vlev = pair
    p_ac = plev * params['Paco']
    opt = minimize(obj, plev*params['Pdco'],
                   args=(p_ac, v_levels[vlev], params))
    p_dc = opt.x[0]
#    p_dc = plev * params['Pdco']
#    v_input = v_levels[vlev]
#    p_ac = snlinverter(v_input, p_dc, params)
    res['fraction_of_rated_power'][idx] = plev
    res['dc_voltage_level'][idx] = vlev
    res['efficiency'][idx] = p_ac / p_dc
    res['pac'][idx] = p_ac
    res['pdc'][idx] = p_dc

#with open('inverter_fit_snl_datasheet.csv', 'w') as outfile:
#    res.to_csv(outfile, index=False, line_terminator='\n')

def _calc_c0(curves, p_ac0, p_dc0, p_s0):
    x = curves['pdc'] - p_s0
    return (curves['pac'] - x / (p_dc0 - p_s0) * p_ac0) / (x**2. - x * (p_dc0 - p_s0))

c_M = params['C0'] * (1. + params['C3'] * (v_levels['Vmax'] - v_levels['Vnom']))
c_m = params['C0'] * (1. + params['C3'] * (v_levels['Vmin'] - v_levels['Vnom']))
