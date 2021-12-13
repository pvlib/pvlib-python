'''
This ``mlfm graphs`` module contains functions to display
performance of PV modules using the mechanistic performance (MPM) and
loss factors models (LFM)

Authors : Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)

https://pvlib-python.readthedocs.io/en/stable/variables_style_rules.html#variables-style-rules

https://github.com/python/peps/blob/master/pep-0008.txt
'''

import numpy as np
import matplotlib.pyplot as plt

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
    '''

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


def plot_mlfm_stack(dmeas, dnorm, dstack, ref,
                    mlfm_file_name, qty_mlfm_vars,
                    xaxis_labels=12, is_i_sc_self_ref=False,
                    is_v_oc_temp_module_corr=True):

    '''
    Plot graph of stacked MLFM losses from intital 1/FF down to pr_dc.

    Parameters
    ----------
    dmeas : dataframe
        measured weather data
        'poa_global', 'temp_module', 'wind_speed'
        and measured electrical/thermal values
        'i_sc' .. 'v_oc', temp_module.

    dnorm : dataframe
        normalised multiplicative lfm loss values 'i_sc' .. 'v_oc'
        where pr_dc = 1/ff * product('i_sc', ... 'v_oc').

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

    xaxis_labels : int
        number of xaxis labels to show (~12) or 0 to show all.

    is_i_sc_self_ref : bool
       self corrects i_sc to remove angle of incidence,
       spectrum, snow or soiling?.

    is_v_oc_temp_module_corr : bool
       calc loss due to gamma, subtract from v_oc loss.

    '''

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
References

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
