'''
This ``mlfm code`` module contains functions to analyse and predict
performance of PV modules using the mechanistic performance (MPM) and
loss factors models (LFM). The module also contains functions to display
performance of PV modules using the mechanistic performance (MPM) and
loss factors models (LFM)

Authors : Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)

https://pvlib-python.readthedocs.io/en/stable/variables_style_rules.html#variables-style-rules

https://github.com/python/peps/blob/master/pep-0008.txt
'''

import numpy as np
import pandas as pd
from scipy import optimize


#  DEFINE REFERENCE MEASUREMENT CONDITIONS
# or use existing definitions in pvlib

# NAME  value  comment         unit     PV_LIB name
#
T_STC = 25.0  # STC temperature [C]      temperature_ref
T_HTC = 75.0  # HTC temperature [C]
G_STC = 1.0   # STC irradiance  [kW/m^2]
G_LIC = 0.2   # LIC irradiance  [kW/m^2]


#  Define standardised MLFM graph colours as a dict ``clr``

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


def mlfm_meas_to_norm(dmeas, ref):
    '''
    Convert measured power, current and voltage to normalized values.

    Parameters
    ----------
    dmeas : DataFrame
        Measurements. Must include columns:

        * `'poa_global_kwm2'` global plane of array irradiance [kW/m^2]
        * `'temp_module'` module temperature [C]
        * `'p_mp'` - power at maximum power point [kW]
        * `'temp_module'` - module temperature [C]

        May include optional columns:

        * `'i_sc'` - current at short circuit condition [A]
        * `'v_oc'` - voltage at open circuit condition [V]
        * `'i_mp'` - current at maximum power point [A]. Must be accompanied
          by `'i_sc'`.
        * `'v_mp'` - voltage at maximum power point [V]. Must be accompanied
          by `'v_oc'`.
        * `'r_sc'` - inverse of slope of IV curve at short circuit condition.
          Requires both `'i_sc'` and `'v_oc'`. [V/A]
        * `'r_oc'` - slope of IV curve at open circuit condition. Requires
          both `'i_sc'` and `'v_oc'` [A/V]

    ref : dict
        Reference values. Must include:

        * `'p_mp'` - Power at maximum power point at Standard Test Condition
          (STC). [kW]
        * `'gamma_p_mp'` - Temperature coefficient of power at STC. [W/C]

        May include:

        * `'i_sc'` - Current at short circuit at STC. Required if `'i_sc'` is
          present in `'dmeas'`. [A]
        * `'v_oc'` - Voltage at open circuit at STC. Required if `'V_oc'` is
          present in `'dmeas'`. [A]
        * `'beta_v_oc'` - Temperature coefficient of open circuit voltage at
          STC. Required if `'v_oc'` is present in `'dmeas'`. [V/C]

    Returns
    -------
    dnorm : DataFrame
        Normalised values.
        * `'pr_dc'` is `'p_mp'` normalied by reference `'p_mp'` and 
          `'poa_global_kwm2'`
        * `'pr_dc_temp_corr'` is `'pr_dc'` adjusted to 25C.
        * Columns `'i_sc'`, `'i_mp'`, `'v_oc'`, `'v_mp'`, `'v_oc_temp_corr'`,
          `'r_sc'`, `'r_oc'`, `'i_ff'`, `'v_ff'` are returned when the
          the corresponding optional columns are included in `'dmeas'`.

    References
    ----------
    .. [1] Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)
       "Quantifying Long Term PV Performance and Degradation under Real Outdoor
       and IEC 61853 Test Conditions Using High Quality Module IV Measurements"
       36th EU PVSEC, Marseille, France. September 2019
    '''
    dnorm = pd.DataFrame()

    dnorm['pr_dc'] = (
        dmeas['p_mp'] /
        (ref['p_mp'] * dmeas['poa_global_kwm2']))

    # temperature corrected
    dnorm['pr_dc_temp_corr'] = (
        dnorm['pr_dc'] *
        (1 - ref['gamma_p_mp']*(dmeas['temp_module'] - T_STC)))

    if 'i_sc' in dmeas.columns:
        dnorm['i_sc'] = dmeas['i_sc'] / dmeas['poa_global_kwm2'] / ref['i_sc']
        if 'i_mp' in dmeas.columns:
            dnorm['i_mp'] = dmeas['i_mp'] / dmeas['i_sc']

    if 'v_oc' in dmeas.columns:
        dnorm['v_oc'] = dmeas['v_oc'] / ref['v_oc']
        if 'v_mp' in dmeas.columns:
            dnorm['v_mp'] = dmeas['v_mp'] / dmeas['v_oc']
        # temperature corrected
        dnorm['v_oc_temp_corr'] = dnorm['v_oc'] * \
            (1 - ref['beta_v_oc']*(dmeas['temp_module'] - T_STC))

    if all(c in dmeas.columns for c in ['i_sc', 'v_oc', 'r_sc', 'r_oc']):
        #  create temporary variables (i_r, v_r) from
        #  intercept of r_sc (at i_sc) with r_oc (at v_oc)
        #  to make maths easier

        i_r = ((dmeas['i_sc'] * dmeas['r_sc'] - dmeas['v_oc']) /
               (dmeas['r_sc'] - dmeas['r_oc']))

        v_r = ((dmeas['r_sc'] * (dmeas['v_oc'] - dmeas['i_sc'] *
               dmeas['r_oc']) / (dmeas['r_sc'] - dmeas['r_oc'])))

        # calculate normalised resistances r_sc and r_oc
        dnorm['r_sc'] = i_r / dmeas['i_sc']  # norm_r @ isc
        dnorm['r_oc'] = v_r / dmeas['v_oc']  # norm_r @ roc

        # calculate remaining fill factor losses partitioned to i_ff, v_ff
        dnorm['i_ff'] = dmeas['i_mp'] / i_r
        dnorm['v_ff'] = dmeas['v_mp'] / v_r

    return dnorm


def mlfm_norm_to_stack(dnorm, ff):
    '''
    Converts normalised values to stacked subtractive normalized losses.

    Ref:
    http://www.steveransome.com/PUBS/1909_5CV4_35_PVSEC36_Marseille_Ransome_PPT.pdf

    current losses :
        meas(imp) / ref(i_sc) =
        poa_global_kwm2 * (norm(i_sc) * norm(r_sc) * norm(i_ff))

    voltage losses :
        meas(vmp) / ref(v_oc) =
        (norm(v_ff) * norm(r_oc) * norm(v_oc))

    1/ff_ref = (ref(isc) / ref(imp)) * (ref(voc) / ref(vmp))


    Normalized values can reveal losses via scatter plots vs. irradiance or
    temperature.

    Stacked subtractive losses can show relative loss proportions. Stacked
    losses partition the difference between the normalized power and the
    power that corresponds to the reference fill factor.

    Parameters
    ----------
    dnorm : DataFrame
        Normalised values. Must include columns:

        * `'pr_dc'` normalized power at the maximum power point.
        * `'i_sc'` normalized short circuit current.
        * `'i_mp'` normalized current at maximum power point.
        * `'v_oc'` normalized open circuit voltage.
        * `'v_mp'` normalized voltage at maximum power point.
        * `'v_oc_temp_corr'` normalized open circuit voltage adjusted to 25C.        

        May include optional columns:

        * `'v_ff'` normalized multiplicative loss in fill factor apportioned
          to voltage.
        * `'i_ff'` normalized multiplicative loss in fill factor apportioned
          to current.
        * `'r_oc'` normalized slope of IV curve at open circuit.
        * `'r_sc'` normalized slope of IV curve at short circuit.

        ff : float
            Reference value of fill factor for IV curve at STC conditions.

    Returns
    -------
    dstack : DataFrame
        Stacked subtractive normalized losses.

    See also
    --------
    mlfm_meas_to_norm

    References
    ----------
    .. [1] Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)
       "Quantifying Long Term PV Performance and Degradation under Real Outdoor
       and IEC 61853 Test Conditions Using High Quality Module IV Measurements"
       36th EU PVSEC, Marseille, France. September 2019
    '''

    # create an empty DataFrame to put stack results
    dstack = pd.DataFrame()

    # create a gap to differentiate i and v losses : gap width~0.01
    gap = 0.01

    inv_ff = 1 / ff

    if all(c in dnorm.columns for c in ['v_ff', 'r_oc', 'i_ff', 'r_sc']):

        # include effects of series and shunt resistances in stacked losses
        # find factor to transform multiplicative to subtractive losses
        # correction factor to scale losses to keep 1/ff --> pr_dc

        # product
        prod = inv_ff * (
            dnorm['i_sc'] * dnorm['r_sc'] * dnorm['i_ff'] *
            dnorm['v_ff'] * dnorm['r_oc'] * dnorm['v_oc']
        )

        # total
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

        return dstack

    # subtractive losses without series and shunt resistance effects
    # find factor to transform multiplicative to subtractive losses
    # correction factor to scale losses to keep 1/ff --> pr_dc

    prod = inv_ff * (
        dnorm['i_sc'] * dnorm['i_mp'] *
        dnorm['v_mp'] * dnorm['v_oc']
    )

    tot = inv_ff + (
        dnorm['i_sc'] + dnorm['i_mp'] +
        dnorm['v_mp'] + dnorm['v_oc'] - 4
    )

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


def mlfm_6(dmeas, c_1, c_2, c_3, c_4, c_5=0., c_6=0.):
    r'''
    Predict normalised LFM values from data in ``dmeas``.

    The normalized LFM values are given by

    .. math::

        c_1 + c_2 (T_m - 25) + c_3 \log10(G_{POA}) + c_4 G_{POA}
        + c_5 WS + c_6 / G_{POA}

    where :math:`G_{POA}` is global plane-of-array (POA) irradiance in kW/m2,
    :math:`T_m` is module temperature in C and :math:`WS` is wind speed in
    m/s.

    Parameters
    ----------
    dmeas : DataFrame
        Must include columns:

        * `'poa_global_kwm2'` global plane of array irradiance [kW/m^2]
        * `'temp_module'` module temperature [C]
        May include optional column:

        * `'wind_speed'` wind speed [m/s].
    c_1 : float
        Constant term in model
    c_2 : float
        Temperature coefficient in model (1/K)
    c_3 : float
        Coefficient for low light log irradiance drop.
    c_4 : float
        Coefficient for high light linear irradiance drop.
    c_5 : float, default 0
        Coefficient for wind speed dependence
    c_6 : float, default 0
        Coefficient for dependence on inverse irradiance.

    Returns
    -------
    mlfm_6 : Series
        Predicted values.

    References
    ----------
    .. [1] Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)
       "Quantifying Long Term PV Performance and Degradation under Real Outdoor
       and IEC 61853 Test Conditions Using High Quality Module IV Measurements"
       36th EU PVSEC, Marseille, France. September 2019
     '''
    mlfm_out = c_1 + c_2 * (dmeas['temp_module'] - T_STC) + \
        c_3 * np.log10(dmeas['poa_global_kwm2']) + \
        c_4 * dmeas['poa_global_kwm2'] + c_6 / dmeas['poa_global_kwm2']
    if 'wind_speed' in dmeas.columns:
        mlfm_out += c_5 * dmeas['wind_speed']
    return mlfm_out


def mlfm_fit(data, var_to_fit):
    '''
    Fit MLFM to data.

    Parameters
    ----------
    data : DataFrame
        Must include columns:
        * 'poa_global_kwm2' global plane of array irradiance [kW/m^2]
        * 'temp_module' module temperature [C]
        Must include column named ``var_to_fit``.
        May include optional column:
        * 'wind_speed' wind speed [m/s].

    var_to_fit : string
        Column name in ``data`` containing variable being fit.

    Returns
    -------
    pred : Series
        Values predicted by the fitted model.

    coeff : list
        Model coefficients ``c_1`` to ``c_6``.

    resid : Series
        Residuals of the fitted model.

    See also
    --------
    mlfm_6
    '''

    # drop missing data
    data = data.dropna()

    c5_zero = 'wind_speed' not in data.columns
    # if wind_speed is not present, add it
    if c5_zero:
        data['wind_speed'] = 0.

    # define function name
    func = mlfm_6

    # setup initial values and initial boundary conditions

    # initial   c1    c2    c3    c4    c5   c6<0
    p_0 = (1.0, 0.01, 0.01, 0.01, 0.01, -0.01)
    # boundaries
    bounds = ([ -2,   -2,   -2,   -2,   -2,    -2],
              [  2,    2,    2,    2,    2,     0])

    popt, pcov = optimize.curve_fit(
        f=func,                 # fit function
        xdata=data,            # input data
        ydata=data[var_to_fit],  # fit parameter
        p0=p_0,                 # initial
        bounds=bounds           # boundaries
    )

    # get mlfm coefficients
    c_1 = popt[0]
    c_2 = popt[1]
    c_3 = popt[2]
    c_4 = popt[3]
    c_5 = popt[4]
    c_6 = popt[5]

    if c5_zero:
        c_5 = 0.

    coeff = [c_1, c_2, c_3, c_4, c_5, c_6]

    # get mlfm error coefficients as sqrt of covariance
    perr = np.sqrt(np.diag(pcov))

    e_1 = perr[0]
    e_2 = perr[1]
    e_3 = perr[2]
    e_4 = perr[3]
    e_5 = perr[4]
    e_6 = perr[5]

    err = [e_1, e_2, e_3, e_4, e_5, e_6]

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

    err = (
        '  {:.4%}'.format(e_1) +
        ', {:.4%}'.format(e_2) +
        ', {:.4%}'.format(e_3) +
        ', {:.4%}'.format(e_4) +
        ', {:.4%}'.format(e_5) +
        ', {:.4%}'.format(e_6)
    )
    # print ('errs = ', mlfm_sel, errs)

    # save fit and error to dataframe
    pred = mlfm_6(data, c_1, c_2, c_3, c_4, c_5, c_6)
    resid = pred - data[var_to_fit]

    return pred, coeff, resid


#  Define standardised MLFM graph colours as a dict ``clr``

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


def plot_mlfm_scatter(dmeas, dnorm, mlfm_file_name, qty_mlfm_vars):
    '''
    Scatter plot normalised MLFM parameters(y) vs. irradiance(x).

    y1_axis : e.g. norm(i_sc, ... v_oc),_temp_module_corr
    x_axis  : e.g. irradiance, poa_global_kwm2
    y2_axis : e.g. temp_air, temp_module (C/100 to fit graphs).

    Parameters
    ----------
    dmeas : dataframe
        measured weather data
        'poa_global', 'temp_module', 'wind_speed'
        and measured electrical/thermal values
        'i_sc' .. 'v_oc', temp_module.

    dnorm : dataframe
        multiplicative lfm loss values 'i_sc' ... 'v_oc'
        where pr_dc = 1/ff * product('i_sc', ... 'v_oc')

    mlfm_file_name : string
        mlfm_file_name used in graph title.

    qty_mlfm_vars : int
        number of mlfm_values present in data usually
        2 = (imp, vmp) from mpp tracker
        4 = (i_sc, i_mp, v_mp, v_oc) from matrix
        6 = (i_sc, i_mp, v_mp, v_oc, r_sc, r_oc) from iv curve.

    Returns
    -------
    fig : Figure
        Instance of matplotlib.figure.Figure

    '''
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('mlfm requires matplotlib')

    # offset legend to the right to not overlap graph, use ~1.2
    bbox = 1.2

    # set x_axis as irradiance
    xdata = dmeas['poa_global']

    fig, ax1 = plt.subplots()

    # get filename without ".csv" for title
    ax1.set_title('Plot mlfm scatter ' +
                  mlfm_file_name[:len(mlfm_file_name)-4])

    ax1.set_ylabel('normalised mlfm values')
    ax1.axhline(y=1, c='grey', linewidth=3)  # show 100% line
    ax1.set_ylim(0.8, 1.1)  # optional normalised y scale

    ax1.set_xlabel('poa_global [W/m$^2$]')
    ax1.axvline(x=1.0, c='grey', linewidth=3)  # show 1000W/m^2 STC
    ax1.axvline(x=0.8, c='grey', linewidth=3)  # show 800W/m^2 NOCT
    ax1.axvline(x=0.2, c='grey', linewidth=3)  # show 200W/m^2 LIC

    # plot the mlfm parameters depending on qty_mlfm_vars
    if qty_mlfm_vars == 1:  # only p_mp
        ax1.scatter(xdata, dnorm['pr_dc_temp_corr'],
                    c=clr['pr_dc'], label='pr_dc_temp_corr')

    # if (qty_mlfm_vars == 2) or (qty_mlfm_vars == 4):  # mppt or matrix
    if qty_mlfm_vars in (2, 4):  # mppt or matrix
        ax1.scatter(xdata, dnorm['i_mp'], c=clr['i_mp'], label='norm_i_mp')
        ax1.scatter(xdata, dnorm['v_mp'], c=clr['v_mp'], label='norm_v_mp')

    if qty_mlfm_vars >= 6:  # ivcurve
        ax1.scatter(xdata, dnorm['i_ff'], c=clr['i_ff'], label='norm_i_ff')
        ax1.scatter(xdata, dnorm['v_ff'], c=clr['v_ff'], label='norm_v_ff')
        ax1.scatter(xdata, dnorm['r_sc'], c=clr['r_sc'], label='norm_r_sc')
        ax1.scatter(xdata, dnorm['r_oc'], c=clr['r_oc'], label='norm_r_oc')

    if qty_mlfm_vars >= 4:  # matrix
        ax1.scatter(xdata, dnorm['i_sc'], c=clr['i_sc'], label='norm_i_sc')

        ax1.scatter(xdata, dnorm['v_oc_temp_corr'], c=clr['v_oc'],
                    label='norm_v_oc_temp_corr')

    ax1.legend(bbox_to_anchor=(bbox, 1), loc='upper left', borderaxespad=0.)

    # y2axis plot met on right y axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('temp_module, temp_air (C/100)')  # poa_global (kW/m$^2$);

    # set wide limits 0 to 4 so they don't overlap mlfm params
    ax2.set_ylim(0, 4)

    ax2.scatter(xdata,
                dmeas['temp_module']/100,
                c=clr['temp_module'],
                label='temp_module C/100')

    # temp_air may not exist particularly for indoor measurements
    try:
        ax2.scatter(xdata,
                    dmeas['temp_air']/100,
                    c=clr['temp_air'],
                    label='temp_air C/100')
    except KeyError:
        pass

    ax2.legend(bbox_to_anchor=(bbox, 0.5), loc='upper left', borderaxespad=0.)
    plt.show()

    return fig


def plot_mlfm_stack(dmeas, dnorm, dstack, ref,
                    mlfm_file_name, qty_mlfm_vars,
                    xaxis_labels=12, is_i_sc_self_ref=False,
                    is_v_oc_temp_module_corr=True):

    '''
    Plot graph of stacked MLFM losses from initial 1/FF down to pr_dc.

    Parameters
    ----------
    dmeas : DataFrame
        Measured weather data. Must include 'poa_global_kwm2' and
        'temp_module', may include 'temp_air'.
        and measured electrical/thermal values
        'i_sc' .. 'v_oc', temp_module (cwh: I don't see these are referenced).

    dnorm : dataframe
        Normalised multiplicative LFM loss values 'i_sc' .. 'v_oc'
        where pr_dc = 1/FF * product('i_sc', ... 'v_oc').

    dstack : dataframe
        normalised subtractive lfm loss values 'i_sc' .. 'v_oc'
        where pr_dc = 1/ff - sum('i_sc', ... 'v_oc').

    ref : dict
        reference stc values e.g. 'v_oc' and
        temperature coeffs e.g. 'beta_v_oc'.

    mlfm_file_name : string
        mlfm_file_name used in graph title.

    qty_mlfm_vars : int
        number of mlfm_values present in data usually
        2 = (imp, vmp) from mpp tracker
        4 = (i_sc, i_mp, v_mp, v_oc) from matrix
        6 = (i_sc, i_mp, v_mp, v_oc, r_sc, r_oc) from iv curve.

    xaxis_labels : int, default 12
        Number of xaxis labels to show. Use 0 to show all.

    is_i_sc_self_ref : bool, default False
       Self corrects i_sc to remove angle of incidence,
       spectrum, snow or soiling?.

    is_v_oc_temp_module_corr : bool, default True
       Calculate loss due to gamma, subtract from v_oc loss.

    '''
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('mlfm requires matplotlib')

    # offset legend right, use ~1.2
    bbox = 1.2

    # select x axis usually date_time
    xdata = dmeas.index
    fig, ax1 = plt.subplots()

    ax1.set_title('Plot_mlfm_stack : ' +
                  mlfm_file_name[:len(mlfm_file_name)-4])

    if qty_mlfm_vars == 6:  # iv curve
        labels_6 = [
            'pr_dc',
            'stack_t_mod',
            'stack_v_oc',
            'stack_r_oc',
            'stack_v_ff',
            '- - -',
            'stack_i_ff',
            'stack_r_sc',
            'stack_i_sc'
        ]

        color_map_6 = [
            'white',  # colour to bottom of graph
            clr['temp_module'],
            clr['v_oc'],
            clr['r_oc'],
            clr['v_ff'],
            clr['i_v'],
            clr['i_ff'],
            clr['r_sc'],
            clr['i_sc']
        ]

        # plot stack in order bottom to top,
        # allowing self_ref and temp_module corrections
        ax1.stackplot(
            xdata,
            dnorm['pr_dc'] + (dstack['i_sc'] * (is_i_sc_self_ref)),
            dstack['temp_module_corr'] * (is_v_oc_temp_module_corr),
            dstack['v_oc'] - (
                dstack['temp_module_corr'] * (is_v_oc_temp_module_corr)),
            dstack['r_oc'],
            dstack['v_ff'],
            dstack['i_v'],
            dstack['i_ff'],
            dstack['r_sc'],
            dstack['i_sc'] * (not is_i_sc_self_ref),
            labels=labels_6,
            colors=color_map_6
        )

    if qty_mlfm_vars == 4:  # matrix
        labels_4 = [
            'pr_dc',
            'stack_t_mod',
            'stack_v_oc',
            'stack_v_mp',
            '- - -',
            'stack_i_mp',
            'i_sc'
        ]

        color_map_4 = [
            'white',  # colour to bottom of graph
            clr['temp_module'],
            clr['v_oc'],
            clr['v_mp'],
            clr['i_v'],
            clr['i_mp'],
            clr['i_sc']
        ]

        # plot stack in order bottom to top,
        # allowing self_ref and temp_module corrections
        ax1.stackplot(
            xdata,
            dnorm['pr_dc'] + (dstack['i_sc'] * (is_i_sc_self_ref)),
            dstack['temp_module_corr'] * (is_v_oc_temp_module_corr),
            dstack['v_oc'] - (
                dstack['temp_module_corr'] * (is_v_oc_temp_module_corr)),
            dstack['v_mp'],
            dstack['i_v'],
            dstack['i_mp'],
            dstack['i_sc'] * (not is_i_sc_self_ref),
            labels=labels_4,
            colors=color_map_4
        )

    ax1.axhline(y=1/ref['ff'], c='grey', lw=3)  # show initial 1/FF
    ax1.axhline(y=1, c='grey', lw=3)  # show 100% line
    ax1.set_ylabel('stacked mlfm losses')

    # find number of x date values
    x_ticks = dmeas.shape[0]
    plt.xticks(np.arange(0, x_ticks), rotation=90)

    # if (xaxis_labels > 0 and xaxis_labels < x_ticks):
    if 0 < xaxis_labels < x_ticks:
        xaxis_skip = np.floor(x_ticks / xaxis_labels)
    else:
        xaxis_skip = 2

    #
    xax2 = [''] * x_ticks
    x_count = 0
    while x_count < x_ticks:
        if x_count % xaxis_skip == 0:
            #
            #  try to reformat any date indexes (not for matrices)
            #
            #  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
            #   y y y y - m m - d d t h h : m m : s s --> yy-mm-dd hh'h'
            #
            try:
                xax2[x_count] = xdata[x_count][2:13]+'h'

            except IndexError:
                xax2[x_count] = xdata[x_count]

        x_count += 1

    ax1.set_xticklabels(xax2)
    ax1.set_ylim(0.6, 1/ref['ff']+0.1)  # optional normalised y scale
    plt.legend(bbox_to_anchor=(bbox, 1), loc='upper left', borderaxespad=0.)

    # plot met data on right y axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('poa_global (kW/m^2), temp_module (C/100)')
    ax2.set_ylim(0, 4)  # set so doesn't overlap mlfm params

    plt.plot(xdata, dmeas['poa_global_kwm2'],
             c=clr['irradiance'], label='poa_global_kwm2')
    plt.plot(xdata, dmeas['temp_module'] / 100,
             c=clr['temp_module'], label='temp_module/100')

    # temp_air may not exist particularly for indoor measurements
    try:
        plt.plot(xdata, dmeas['temp_air']/100,
                 c=clr['temp_air'], label='temp_air/100')
    except KeyError:
        pass

    ax2.legend(bbox_to_anchor=(bbox, 0.3), loc='upper left', borderaxespad=0.)
    ax1.set_xticklabels(xax2, rotation=90)
    plt.show()


REFS = """
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
