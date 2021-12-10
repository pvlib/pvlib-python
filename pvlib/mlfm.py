'''
This ``mlfm code`` module contains functions to analyse and predict        79-|
performance of PV modules using the mechanistic performance (MPM) and
loss factors models (LFM)

Authors : Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)

https://pvlib-python.readthedocs.io/en/stable/variables_style_rules.html#variables-style-rules

https://github.com/python/peps/blob/master/pep-0008.txt

'''

import numpy as np
import pandas as pd

from scipy import optimize


''' DEFINE REFERENCE MEASUREMENT CONDITIONS '''
# or use existing definitions in pvlib

# NAME  value  comment         unit     PV_LIB name
#
T_STC = 25.0  # STC temperature [C]      temperature_ref
T_HTC = 75.0  # HTC temperature [C]
G_STC = 1.0   # STC irradiance  [kW/m^2]
G_LIC = 0.2   # LIC irradiance  [kW/m^2]


'''  Define standardised MLFM graph colours as a dict ``clr``  '''

clr = {
    # parameter_clr colour            R   G   B
    'irradiance':   'darkgreen',   # 000 064 000
    'temp_module':  'red',         # 255 000 000
    'temp_air':     'yellow',      # 245 245 220
    'wind_speed':   'grey',        # 127 127 127

    'i_sc':         'purple',      # 128 000 128
    'r_sc':         'orange',      # 255 165 000
    'i_ff':         'lightgreen',  # 144 238 144
    'i_mp':         'green',       # 000 255 000
    'i_v':          'black',       # 000 000 000 between i and v losses
    'v_ff':         'cyan',        # 000 255 255
    'v_mp':         'blue',        # 000 000 255
    'r_oc':         'pink',        # 255 192 203
    'v_oc':         'sienna',      # 160 082 045

    'pr_dc':        'black',       # 000 000 000
}


def mlfm_meas_to_norm(dmeas, ref, qty_mlfm_vars):
    '''
    Convert measured values e.g. meas(i_sc, ... v_oc)
    to normalised loss values e.g. norm(i_sc, ... v_oc)
    normalising by ref stc values and irradiance.

    Parameters
    ----------
    dmeas : dataframe
        measured weather data
        'poa_global', 'temp_module', 'wind_speed'
        and measured electrical/thermal values
        'i_sc' .. 'v_oc', temp_module.

    ref : dict
        reference stc values e.g. 'v_oc' and
        temperature coeffs e.g. 'beta_v_oc' .

    qty_mlfm_vars : int
        number of mlfm_values present in data usually
        2 = (imp, vmp) from mpp tracker
        4 = (i_sc, i_mp, v_mp, v_oc) from matrix
        6 = (i_sc, i_mp, v_mp, v_oc, r_sc, r_oc) from iv curve.

    Returns
    -------
    dnorm : dataframe
        normalised multiplicative lfm loss values 'i_sc' .. 'v_oc'
        where pr_dc = 1/ff * product('i_sc', ... 'v_oc').
    '''
    dnorm = pd.DataFrame()

    # calculate normalised mlfm values depending on number of qty_mlfm_vars

    if qty_mlfm_vars >= 1:  # do for all measurements
        dnorm['pr_dc'] = (
            dmeas['p_mp'] /
            (ref['p_mp'] * dmeas['poa_global_kwm2']))

        # temperature corrected
        dnorm['pr_dc_temp_corr'] = (
            dnorm['pr_dc'] *
            (1 - ref['gamma_p_mp']*(dmeas['temp_module']-T_STC)))

    if qty_mlfm_vars >= 2:  #
        dnorm['i_mp'] = dmeas['i_mp'] / dmeas['i_sc']
        dnorm['v_mp'] = dmeas['v_mp'] / dmeas['v_oc']

    if qty_mlfm_vars >= 4:  #
        dnorm['i_sc'] = (
            dmeas['i_sc'] /
            (dmeas['poa_global_kwm2'] * ref['i_sc']))

        dnorm['v_oc'] = dmeas['v_oc'] / ref['v_oc']

        # temperature corrected
        dnorm['v_oc_temp_corr'] = (
            dnorm['v_oc'] *
            (1 - ref['beta_v_oc']*(dmeas['temp_module']-T_STC)))

    if qty_mlfm_vars >= 6:  # 6,8 IV data

        '''
        create temporary variables (ir, vr) from
        intercept of r_sc (at i_sc) with r_oc (at v_oc)
        to make maths easier
        '''

        ir = (
            (dmeas['i_sc'] * dmeas['r_sc'] - dmeas['v_oc']) /
            (dmeas['r_sc'] - dmeas['r_oc']))

        vr = (
            dmeas['r_sc'] * (dmeas['v_oc'] - dmeas['i_sc'] *
            dmeas['r_oc']) / (dmeas['r_sc'] - dmeas['r_oc']))

        # calculate normalised resistances r_sc and r_oc
        dnorm['r_sc'] = ir / dmeas['i_sc']  # norm_r @ isc
        dnorm['r_oc'] = vr / dmeas['v_oc']  # norm_r @ roc

        # calculate remaining fill factor losses partitioned to i_ff, v_ff
        dnorm['i_ff'] = dmeas['i_mp'] / ir
        dnorm['v_ff'] = dmeas['v_mp'] / vr

    return dnorm


def mlfm_6(dmeas, c_1, c_2, c_3, c_4, c_5, c_6):
    '''
    Predict normalised lfm values e.g. pr_dc, norm(i_sc, ... v_oc)
    from poa_global, temp_module, wind_speed and mlfm(c_1 .. c_6).

    Parameters
    ----------

    dmeas : dataframe
        measured weather data
        'poa_global', 'temp_module', 'wind_speed'
        and measured electrical/thermal values
        'i_sc' .. 'v_oc', temp_module.

    c_1 to c_6 : float
        fitted mlfm coefficients (dependencies)
            c_1 - constant
            c_2 - temperature coefficient (1/K)
            c_3 - low light log irradiance drop (~v_oc)
            c_4 - high light linear irradiance drop (~r_s)
            c_5 - wind speed dependence (=0 if indoor)
            c_6 - inverse irradiance (<= 0).

    Returns
    -------
    mlfm_6 : float
        predicted performance values for pr_dc, norm(i_sc, .. v_oc) .
    '''

    mlfm_6 = (
        c_1 +                                       # 'constant' lossless
        c_2 * (dmeas['temp_module'] - T_STC) +      # temperature coefficient
        c_3 * np.log10(dmeas['poa_global_kwm2']) +  # low light drop, 'v_oc'
        c_4 * dmeas['poa_global_kwm2'] +            # high light drop 'rs'
        c_5 * dmeas['wind_speed'] +                 # wind_speed (optional or 0)
        c_6 / dmeas['poa_global_kwm2']              # rsh (optional but < 0)
    )

    return mlfm_6


def mlfm_norm_to_stack(dmeas, dnorm, ref, qty_mlfm_vars):
    '''
    Converts MLFM normalised multiplicative losses norm(i_sc ... v_oc)
    to stacked subtractive losses stack(i_sc ... v_oc).

    Ref:
    http://www.steveransome.com/PUBS/1909_5CV4_35_PVSEC36_Marseille_Ransome_PPT.pdf

    current losses :
        meas(imp) / ref(i_sc) =
        poa_global_kwm2 * (norm(i_sc) * norm(r_sc) * norm(i_ff))

    voltage losses :
        meas(vmp) / ref(v_oc) =
        (norm(v_ff) * norm(r_oc) * norm(v_oc))

    1/ff_ref = (ref(isc) / ref(imp)) * (ref(voc) / ref(vmp))


    Multiplicative losses:
        - are easier to use on a scatter plot vs. irradiance or temperature.

    Stacked losses:
        - are better to use to see relative loss proportions
        (for underperformance limitations)
        or vs. time e.g. for degradation.

        - Stacked losses are scaled so they give the correct pr_dc
        and are in the right proportion to each other.

    Parameters
    ----------

    dmeas : dataframe
        measured weather data
        'poa_global', 'temp_module', 'wind_speed'
        and measured electrical/thermal values
        'i_sc' .. 'v_oc', temp_module..

    dnorm : dataframe
        normalised multiplicative lfm loss values 'i_sc' .. 'v_oc'
        where pr_dc = 1/ff * product('i_sc', ... 'v_oc').

    ref : dict
        reference stc values e.g. 'v_oc' and
        temperature coeffs e.g. 'beta_v_oc' .

    qty_mlfm_vars : int
        number of mlfm_values present in data usually
        2 = (imp, vmp) from mpp tracker
        4 = (i_sc, i_mp, v_mp, v_oc) from matrix
        6 = (i_sc, i_mp, v_mp, v_oc, r_sc, r_oc) from iv curve.

    Returns
    -------
    dstack : dataframe
        normalised subtractive lfm loss values 'i_sc' .. 'v_oc'
        where pr_dc = 1/ff - sum('i_sc', ... 'v_oc').

    '''

    # create an empty pands to put stack results
    dstack = pd.DataFrame()

    # create a gap to differentiate i and v losses : gap width~0.01
    gap = 0.01

    # calculate reference fill factor (usually between 0.5 and 0.8)
    ff_ref = ref['ff']

    '''
    calculate inverse fill factor ~ 1.25 - 2 as we calculate
    i_losses from ref_isc to norm_imp
    v_losses from ref_vocc to norm_vmp
    '''
    inv_ff = 1 / ff_ref

    if qty_mlfm_vars == 6:  # ivcurve
        '''
        find factor to transform multiplicative to subtractive losses
        correction factor to scale losses to keep 1/ff --> pr_dc
        '''
        #  product
        prod = inv_ff * (
            dnorm['i_sc'] * dnorm['r_sc'] * dnorm['i_ff'] *
            dnorm['v_ff'] * dnorm['r_oc'] * dnorm['v_oc']
        )

        #  total
        tot = inv_ff + (
            dnorm['i_sc'] + dnorm['r_sc'] + dnorm['i_ff'] +
            dnorm['v_ff'] + dnorm['r_oc'] + dnorm['v_oc'] - 6
        )

        # correction factor
        corr = (inv_ff - prod) / (inv_ff - tot)

        # put mlfm values in a stack from pr_dc (bottom) to 1/ff_ref (top)
        dstack['pr_dc'] = +dnorm['pr_dc']  # initialise
        dstack['i_sc'] = -(dnorm['i_sc'] - 1) * corr
        dstack['r_sc'] = -(dnorm['r_sc'] - 1) * corr
        dstack['i_ff'] = -(dnorm['i_ff'] - 1) * corr - gap/2
        dstack['i_v'] = gap
        dstack['v_ff'] = -(dnorm['v_ff'] - 1) * corr - gap/2
        dstack['r_oc'] = -(dnorm['r_oc'] - 1) * corr
        dstack['v_oc'] = -(dnorm['v_oc'] - 1) * corr
        dstack['temp_module_corr'] = (
            -(dnorm['v_oc'] - dnorm['v_oc_temp_corr']) * corr)

    elif qty_mlfm_vars == 4:  # matrix
        '''
        find factor to transform multiplicative to subtractive losses
        correction factor to scale losses to keep 1/ff --> pr_dc
        '''

        #  product
        prod = inv_ff * (
            dnorm['i_sc'] * dnorm['i_mp'] *
            dnorm['v_mp'] * dnorm['v_oc']
        )

        #  total
        tot = inv_ff + (
            dnorm['i_sc'] + dnorm['i_mp'] +
            dnorm['v_mp'] + dnorm['v_oc'] - 4
        )

        # correction factor
        corr = (inv_ff - prod) / (inv_ff - tot)

        # put mlfm values in a stack from pr_dc (bottom) to 1/ff_ref (top)
        dstack['pr_dc'] = + dnorm['pr_dc']  # initialise
        dstack['i_sc'] = -(dnorm['i_sc'] - 1) * corr
        dstack['i_mp'] = -(dnorm['i_mp'] - 1) * corr - gap/2
        dstack['i_v'] = gap
        dstack['v_mp'] = -(dnorm['v_mp'] - 1) * corr - gap/2
        dstack['v_oc'] = -(dnorm['v_oc'] - 1) * corr

        dstack['temp_module_corr'] = (
            - (dnorm['v_oc'] - dnorm['v_oc_temp_corr']) * corr)

    return dstack


def mlfm_fit(dmeas, dnorm, mlfm_sel):
    '''
    Fit MLFM to normalised data e.g. norm_pr_dc,(norm_i_sc .. norm_v_oc).

    Parameters
    ----------

    dnorm : dataframe
        normalised multiplicative lfm loss values 'i_sc' .. 'v_oc'
        where pr_dc = 1/ff * product('i_sc', ... 'v_oc').

    mlfm_sel : string
        mlfm variable being fitted e.g. pr_dc.

    Returns
    -------
    dnorm : dataframe
        same data but with added mlfm_var fit values
        calc_mlfm_sel and diff_mlfm_sel.

    cc : list
        fit coefficients c_1 to c_6.

    ee : list
        error values.

    coeffs : string
        formatted coefficients for printing.

    errs
        formatted errors for printing.
    '''

    # drop missing data
    dnorm = dnorm.dropna()
    '''
    ensure correct number of p0 and bounds
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    '''
    # define function name
    f = mlfm_6

    # setup initial values and initial boundary conditions

    # initial   c1    c2    c3    c4    c5   c6<0
    p0 = (1.0, 0.01, 0.01, 0.01, 0.01, -0.01)
    # boundaries
    bounds = ([ -2,   -2,   -2,   -2,   -2,    -2],
              [  2,    2,    2,    2,    2,     0])

    if True:
        popt, pcov = optimize.curve_fit(
            f=f,                    # fit function
            xdata=dmeas,            # input data
            ydata=dnorm[mlfm_sel],  # fit parameter
            p0=p0,                  # initial
            bounds=bounds           # boundaries
        )

        # get mlfm coefficients
        c_1 = popt[0]
        c_2 = popt[1]
        c_3 = popt[2]
        c_4 = popt[3]
        c_5 = popt[4]
        c_6 = popt[5]

        cc = [c_1, c_2, c_3, c_4, c_5, c_6]

        # get mlfm error coefficients as sqrt of covariance
        perr = np.sqrt(np.diag(pcov))

        e_1 = perr[0]
        e_2 = perr[1]
        e_3 = perr[2]
        e_4 = perr[3]
        e_5 = perr[4]
        e_6 = perr[5]

        ee = [e_1, e_2, e_3, e_4, e_5, e_6]

        # format coefficients as strings, easier to read in graph title
        coeffs = (
            '  {:.4%}'.format(c_1) +
            ', {:.4%}'.format(c_2) +
            ', {:.4%}'.format(c_3) +
            ', {:.4%}'.format(c_4) +
            ', {:.4%}'.format(c_5) +
            ', {:.4%}'.format(c_6)
        )
        # print ('coeffs = ', mlfm_sel, coeffs)

        errs = (
            '  {:.4%}'.format(e_1) +
            ', {:.4%}'.format(e_2) +
            ', {:.4%}'.format(e_3) +
            ', {:.4%}'.format(e_4) +
            ', {:.4%}'.format(e_5) +
            ', {:.4%}'.format(e_6)
        )
        # print ('errs = ', mlfm_sel, errs)

        # save fit and error to dataframe
        dnorm['calc_' + mlfm_sel] = (
            mlfm_6(dmeas, c_1, c_2, c_3, c_4, c_5, c_6))

        dnorm['diff_' + mlfm_sel] = (
            dnorm[mlfm_sel] - dnorm['calc_' + mlfm_sel])

    return(dnorm, cc, ee, coeffs, errs)


"""
## References

The Loss Factors Model (LFM) and Mechanistic Performance Model (MPM)
together known as "MLFM" have been developed by SRCL and Gantner Instruments
(previously Oerlikon Solar and Tel Solar) since 2011 MLFM and 2017 MPM

.. [1] J. Sutterlueti(now Gantner Instruments) and S. Ransome
 '4AV.2.41 Characterising PV Modules under Outdoor Conditions:
What's Most Important for Energy Yield'
26th EU PVSEC 8 September 2011; Hamburg, Germany.
http://www.steveransome.com/pubs/2011Hamburg_4AV2_41.pdf

.. [2] Steve Ransome and Juergen Sutterlueti(Gantner Instruments)
  'Choosing the best Empirical Model for predicting energy yield'
  7th PV Energy Rating and Module Performance Modeling Workshop,
  Canobbio, Switzerland 30-31 March, 2017.

.. [3] S. Ransome and J. Sutterlueti (Gantner Instruments)
'Checking the new IEC 61853.1-4 with high quality 3rd party data to
benchmark its practical relevance in energy yield prediction'
PVSC June 2019 [Chicago], USA.
http://www.steveransome.com/PUBS/1906_PVSC46_Chicago_Ransome.pdf

.. [4] Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)
'5CV.4.35 Quantifying Long Term PV Performance and Degradation
under Real Outdoor and IEC 61853 Test Conditions
Using High Quality Module IV Measurements'.
36th EU PVSEC Sep 2019 [Marseille]

.. [5] Steve Ransome (SRCL)
'How to use the Loss Factors and Mechanistic Performance Models
effectively with PVPMC/PVLIB'
[PVPMC] Webinar on PV Performance Modeling Methods, Aug 2020.
https://pvpmc.sandia.gov/download/7879/

.. [6] W.Marion et al (NREL)
'New Data Set for Validating PV Module Performance Models'.
https://www.researchgate.net/publication/286746041_New_data_set_for_validating_PV_module_performance_models

Many more papers are available at www.steveransome.com
"""
