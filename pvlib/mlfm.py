'''
This ``mlfm code`` module contains functions to analyse and predict
performance of PV modules using the mechanistic performance (MPM) and
loss factors models (LFM). The module also contains functions to display
performance of PV modules using the mechanistic performance (MPM) and
loss factors models (LFM)

Authors : Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)
Thanks to Cliff Hansen (Sandia National Laboratories)

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
        * `'p_mp'` - power at maximum power point [W]

        May include optional columns:

        * `'i_sc'` - current at short circuit condition [A]
        * `'v_oc'` - voltage at open circuit condition [V]
        * `'i_mp'` - current at maximum power point [A]. Must be accompanied
          by `'i_sc'`.
        * `'v_mp'` - voltage at maximum power point [V]. Must be accompanied
          by `'v_oc'`.
        * `'r_sc'` - inverse of slope of IV curve at short circuit condition.
          Requires both `'i_sc'` and `'v_oc'`. [Ohm]
        * `'r_oc'` - inverse slope of IV curve at open circuit condition.
          Requires both `'i_sc'` and `'v_oc'` [Ohm]

    ref : dict
        Reference values. Must include:

        * `'p_mp'` - Power at maximum power point at Standard Test Condition
          (STC). [W]
        * `'gamma_pdc'` - Temperature coefficient of power at STC. [1/C]

        May include:

        * `'i_sc'` - Current at short circuit at STC. Required if `'i_sc'` is
          present in `'dmeas'`. [A]
        * `'v_oc'` - Voltage at open circuit at STC. Required if `'V_oc'` is
          present in `'dmeas'`. [A]
        * `'beta_v_oc'` - Temperature coefficient of open circuit voltage at
          STC. Required if `'v_oc'` is present in `'dmeas'`. [1/C]

    Returns
    -------
    dnorm : DataFrame
        Normalised values.
        * `'pr_dc'` is `'p_mp'` normalised by reference `'p_mp'` and
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
        (1 - ref['gamma_pdc']*(dmeas['temp_module'] - T_STC)))

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


def mlfm_norm_to_stack(dnorm, fill_factor):
    '''
    Converts normalised values to stacked subtractive normalized losses.

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

        fill_factor : float
            Reference value of fill factor for IV curve at STC conditions.

    Returns
    -------
    dstack : DataFrame
        Stacked subtractive normalized losses. Includes columns:

        * `'pr_dc'` equal to `dnorm['pr_dc']`.
        * `'i_sc'`
        * `'r_sc'`
        * `'i_mp'`
        * `'i_v'`
        * `'v_mp'`
        * `'v_oc'`
        * `'temp_module_corr'`

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

    inv_ff = 1 / fill_factor

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
        # accounting for series and shunt resistance losses
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

    # if data has no wind_speed measurements then c_5 coefficient is
    # meaningless but a non-zero value may have been returned.
    if c5_zero:
        popt[4] = 0.

    # get error of mlfm coefficients as sqrt of covariance
    perr = np.sqrt(np.diag(pcov))

    # save fit and error to dataframe
    pred = mlfm_6(data, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
    resid = pred - data[var_to_fit]

    return pred, popt, resid, perr


def plot_mlfm_scatter(dmeas, dnorm, title):
    '''
    Scatterplot of normalised values (y) vs. irradiance (x).

    y1_axis : e.g. norm(i_sc, ... v_oc),_temp_module_corr
    x_axis  : e.g. irradiance, poa_global_kwm2
    y2_axis : e.g. temp_air, temp_module (C/100 to fit graphs).

    Parameters
    ----------
    dmeas : DataFrame
        Measurements. Must include columns:

        * `'poa_global_kwm2'` global plane of array irradiance [kW/m^2]
        * `'temp_module'` module temperature [C]

        May include optional columns:

        * `'temp_air'` - air temperature [C]

    dnorm : DataFrame
        Normalised values. May include columns:

        * `'pr_dc_temp_corr'` normalized power at the maximum power point.
        * `'i_sc'` normalized short circuit current.
        * `'i_mp'` normalized current at maximum power point.
        * `'v_mp'` normalized voltage at maximum power point.
        * `'v_oc_temp_corr'` normalized open circuit voltage adjusted to 25C.
        * `'v_ff'` normalized multiplicative loss in fill factor apportioned
          to voltage.
        * `'i_ff'` normalized multiplicative loss in fill factor apportioned
          to current.
        * `'r_oc'` normalized slope of IV curve at open circuit.
        * `'r_sc'` normalized slope of IV curve at short circuit.

    title : string
        Title for the figure.

    Returns
    -------
    fig : Figure
        Instance of matplotlib.figure.Figure

    See also
    --------
    mlfm_meas_to_norm
    '''
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('plot_mlfm_scatter requires matplotlib')

    # offset legend to the right to not overlap graph, use ~1.2
    bbox = 1.2

    # set x_axis as irradiance
    xdata = dmeas['poa_global_kwm2'] / 1000.

    fig, ax1 = plt.subplots()

    ax1.set_title(title)

    ax1.set_ylabel('Normalised values')
    ax1.axhline(y=1, c='grey', linewidth=3)  # show 100% line
    ax1.set_ylim(0.8, 1.1)  # optional normalised y scale

    ax1.set_xlabel('Plane of array irradiance [W/m$^2$]')
    ax1.axvline(x=1.0, c='grey', linewidth=3)  # show 1000W/m^2 STC
    ax1.axvline(x=0.8, c='grey', linewidth=3)  # show 800W/m^2 NOCT
    ax1.axvline(x=0.2, c='grey', linewidth=3)  # show 200W/m^2 LIC

    lines = {
        'pr_dc_temp_corr': 'pr_dc',
        'i_mp': 'i_mp',
        'v_mp': 'v_mp',
        'i_sc': 'i_sc',
        'r_sc': 'r_sc',
        'r_oc': 'r_oc',
        'i_ff': 'i_ff',
        'v_ff': 'v_ff',
        'v_oc_temp_corr': 'v_oc'}
    labels = {
        'pr_dc_temp_corr': 'pr_dc_temp-corr',
        'i_mp': 'norm_i_mp',
        'v_mp': 'norm_v_mp',
        'i_sc': 'norm_i_sc',
        'r_sc': 'norm_r_sc',
        'r_oc': 'norm_r_oc',
        'i_ff': 'norm_i_ff',
        'v_ff': 'norm_v_ff',
        'v_oc_temp_corr': 'norm_v_oc_temp_corr'}

    # plot the mlfm parameters depending on qty_mlfm_vars
    for k in lines.keys():
        try:
            ax1.scatter(xdata, dnorm[k], c=clr[lines[k]], label=labels[k])
        except KeyError:
            pass

    ax1.legend(bbox_to_anchor=(bbox, 1), loc='upper left', borderaxespad=0.)

    # y2axis plot met on right y axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature (C/100)')  # poa_global (kW/m$^2$);

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


def plot_mlfm_stack(dmeas, dnorm, dstack, fill_factor, title,
                    xaxis_labels=12, is_i_sc_self_ref=False,
                    is_v_oc_temp_module_corr=True):

    '''
    Plot stacked subtractive losses.

    Parameters
    ----------
    dmeas : DataFrame
        Measurements. Must include columns:

        * `'poa_global_kwm2'` global plane of array irradiance [kW/m^2]
        * `'temp_module'` module temperature [C]

        May include optional columns:

        * `'temp_air'` - air temperature [C]

    dnorm : DataFrame
        Normalised values. Must contain column `'pr_dc'`.

    dstack : DataFrame
        Stacked subtractive losses. Must contain columns `'v_oc'`, `'v_mp'`,
        `'i_v'`, `'i_mp'`, `'i_sc'`, `'temp_module_corr'`. If optional columns
        `'r_oc'` and `'v_ff'` are present, these columns are plotted instead
        of `'v_mp'`. If optional columns `'r_sc'` and `'i_ff'` are present,
        these columns are plotted instead of `'i_mp'`.

    fill_factor : float
        Reference value of fill factor for IV curve at STC conditions.

    title : string
        Title for the figure.

    xaxis_labels : int, default 12
        Number of xaxis labels to show. Use 0 to show all.

    is_i_sc_self_ref : bool, default False
       Self corrects i_sc to remove angle of incidence,
       spectrum, snow or soiling.

    is_v_oc_temp_module_corr : bool, default True
       Calculate loss due to gamma, subtract from v_oc loss.

    Returns
    -------
    fig : Figure
        Instance of matplotlib.figure.Figure

    See also
    --------
    mlfm_norm_to_stack
    '''
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('plt_mlfm_stack requires matplotlib')

    # offset legend right, use ~1.2
    bbox = 1.2

    # select x axis usually date_time
    xdata = dmeas.index
    fig, ax1 = plt.subplots()

    ax1.set_title(title)

    ydata = [dnorm['pr_dc'] + (dstack['i_sc'] * (is_i_sc_self_ref)),
             dstack['temp_module_corr'] * (is_v_oc_temp_module_corr),
             dstack['v_oc'] - (
                 dstack['temp_module_corr'] * (is_v_oc_temp_module_corr)),
             dstack['v_mp'],
             dstack['i_v'],
             dstack['i_mp'],
             dstack['i_sc'] * (not is_i_sc_self_ref)]
    labels = [
        'pr_dc',
        'stack_t_mod',
        'stack_v_oc',
        'stack_v_mp',
        '- - -',
        'stack_i_mp',
        'stack_i_sc']
    color_map = [
        'white',  # colour to bottom of graph
        clr['temp_module'],
        clr['v_oc'],
        clr['v_mp'],
        clr['i_v'],
        clr['i_mp'],
        clr['i_sc']]

    if all([c in dstack.columns for c in ['v_ff', 'r_oc']]):
        # replace v_mp with v_ff and r_oc
        ydata.pop(3)
        ydata.insert(2, dstack['v_ff'])
        ydata.insert(2, dstack['r_oc'])
        labels.pop(3)
        labels.insert(2, 'stack_v_ff')
        labels.insert(2, 'stack_r_oc')
        color_map.pop(3)
        color_map.insert(2, clr['v_ff'])
        color_map.insert(2, clr['r_oc'])

    if all([c in dstack.columns for c in ['i_ff', 'r_sc']]):
        # replace i_mp with i_ff and r_sc
        ydata.pop(-1)
        ydata.append(dstack['r_sc'])
        ydata.append(dstack['i_ff'])
        labels.pop(-1)
        labels.append('stack_r_sc')
        labels.append('stack_i_ff')
        color_map.pop(-1)
        color_map.append(clr['r_scf'])
        color_map.append(clr['i_ff'])

    # plot stack in order bottom to top,
    # allowing self_ref and temp_module corrections
    ax1.stackplot(xdata, *tuple(ydata), labels=labels, colors=color_map)

    ax1.axhline(y=1/fill_factor, c='grey', lw=3)  # show initial 1/FF
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
    ax1.set_ylim(0.6, 1/fill_factor + 0.1)  # optional normalised y scale
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

    return fig


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
