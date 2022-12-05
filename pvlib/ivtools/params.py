"""
The ``params`` module contains classes and functions to extract parameters
(e.g. Isc and Voc) from current-voltage curves.
"""
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial as Poly


def astm_e1036(v, i, imax_limits=(0.75, 1.15), vmax_limits=(0.75, 1.15),
               voc_points=3, isc_points=3):
    '''
    Extract photovoltaic IV parameters according to ASTM E1036. Assumes the
    curve is in the first quadrant

    Parameters
    ----------
    v : array-like
        Voltage points
    i : array-like
        Current points
    imax_limits : tuple, default (0.75, 1.15)
        Two-element tuple (low, high) specifying the fraction of estimated
        Imp within which to fit a polynomial for max power calculation
    vmax_limits : tuple, default (0.75, 1.15)
        Two-element tuple (low, high) specifying the fraction of estimated
        Vmp within which to fit a polynomial for max power calculation
    voc_points : int, default 3
        the number of points near open circuit to use for linear fit
        and Voc calculation
    isc_points : int, default 3
        the number of points near short circuit to use for linear fit and
        Isc calculation

    Returns
    -------
    dict
        Calculated IV parameters

    Adapted from https://github.com/NREL/iv_params
    Copyright (c) 2022, Alliance for Sustainable Energy, LLC
    All rights reserved.
    '''

    df = pd.DataFrame()
    df['v'] = v
    df['i'] = i
    df['p'] = df['v'] * df['i']

    # first calculate estimates of voc and isc
    voc = np.nan
    isc = np.nan

    # determine if we can use voc and isc estimates
    i_min_ind = df['i'].abs().idxmin()
    v_min_ind = df['v'].abs().idxmin()
    voc_est = df['v'][i_min_ind]
    isc_est = df['i'][v_min_ind]

    # accept the estimates if they are close enough
    if abs(df['i'][i_min_ind]) <= isc_est * 0.001:
        voc = voc_est
    if abs(df['v'][v_min_ind]) <= voc_est * 0.005:
        isc = isc_est

    # perform a linear fit if estimates rejected
    if np.isnan(voc):
        df['i_abs'] = df['i'].abs()
        voc_df = df.nsmallest(voc_points, 'i_abs')
        voc_fit = Poly.fit(voc_df['i'], voc_df['v'], 1)
        voc = voc_fit(0)

    if np.isnan(isc):
        df['v_abs'] = df['v'].abs()
        isc_df = df.nsmallest(isc_points, 'v_abs')
        isc_fit = Poly.fit(isc_df['v'], isc_df['i'], 1)
        isc = isc_fit(0)

    # estimate max power point
    max_index = df['p'].idxmax()
    mp_est = df.loc[max_index]

    # filter around max power
    mask = (
        (df['i'] >= imax_limits[0] * mp_est['i']) &
        (df['i'] <= imax_limits[1] * mp_est['i']) &
        (df['v'] >= vmax_limits[0] * mp_est['v']) &
        (df['v'] <= vmax_limits[1] * mp_est['v'])
    )
    filtered = df[mask]

    # fit polynomial and find max
    mp_fit = Poly.fit(filtered['v'], filtered['p'], 4)
    roots = mp_fit.deriv().roots()
    # only consider real roots
    roots = roots.real[abs(roots.imag) < 1e-5]
    # only consider roots in the relevant part of the domain
    roots = roots[(roots < filtered['v'].max()) &
                  (roots > filtered['v'].min())]
    vmp = roots[np.argmax(mp_fit(roots))]
    pmp = mp_fit(vmp)
    imp = pmp / vmp

    ff = pmp / (voc * isc)

    result = {}
    result['voc'] = voc
    result['isc'] = isc
    result['vmp'] = vmp
    result['imp'] = imp
    result['pmp'] = pmp
    result['ff'] = ff

    return result
