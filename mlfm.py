"""Analyse, fit + predict PV performance measurements using MPM & LFM."""
import numpy as np
import os
import pandas as pd
from scipy import optimize

# import pvlib

"""
ver : 221213t22 <-- delete when finalised 

``mlfm.py`` module contains functions to analyse, fit, predict and display
performance of PV modules using the mechanistic performance model (MPM) and
loss factors model (LFM).

Authors : Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)
Comments : Cliff Hansen, Kevin Anderson, Anton Driesse and Mark Campanelli
https://pvlib-python.readthedocs.io/en/stable/variables_style_rules.html#variables-style-rules
https://github.com/python/peps/blob/master/pep-0008.txt

OVERVIEW

I)   The Loss Factors Model (LFM) 2011 ref [1] quantifies
normalised losses from module parameters (e.g. pr_dc, i_sc, r_sc, i_mp,
v_mp, r_oc and v_oc) by analysing module measurements or the shape of the
IV curve and comparing it with STC reference values from the datasheet.

    Depending on the number of measurements available the LFM is defined
with a suffix number x = 1..12 LFM_n as in ref [4] -

                            parameters modelled
|LFM_1 |                          ``p_mp``                        |
|LFM_2 |                     ``i_mp``, ``v_mp``,                  |
|LFM_4 | ``i_sc``,           ``i_mp``, ``v_mp``,         ``v_oc`` |
|LFM_6 | ``i_sc``, ``r_sc``, ``i_ff`, ``v_ff`, ``r_oc``, ``v_oc`` |

|LFM_>6| (can include normalised losses for :
          soiling, reflectivity vs. aoi, spectrum <- affecting i_sc,
          current mismatch/shading, rollover,
          clipping etc.)

    This file just contains -
LFM_6 : 'measurements with r_sc and r_oc'
    e.g. iv curves with good smooth data.

LFM_4 : 'measurements without r_sc or r_oc'
    e.g. indoor matrix measurements or iv curves without smoooth data.

II)  The Mechanistic performance model (MPM) 2017 ref [2]
has "meaningful,independent, robust and normalised" coefficients
which fit how the LFM values depend on irradiance, module temperature
(and windspeed) and time.

Two MPM versions have been included here :

mpm_a : (mpm_original 2017 ref [2] now deprecated)
    The original model to fit normalised parameters such as
    pr_dc, v_oc, r_sc, v_mp, i_mp, ff ...
    with an extra low light coefficient c_6 to help fit data with
    unusual low light performance and/or poor measurements.
    c_5 is only used if there is windspeed data, otherwise it is ignored

    mpm_a = c_1 +c_2*(t_mod-25) +c_3*log10(g) +c_4*g +c_5*ws +c_6/g

mpm_b : (GI name 'mpm_advanced' 2022 ref [7])
    Is an improved model to fit normalised parameters such as
    pr_dc, v_oc, r_sc, v_mp, i_mp, ff ...
    It better fits precise measurements (see CFV and GI) where the
    low light data is measured well and has an improvement for even
    better v_oc fitting [ref 7 : 2022 PVSC PHILADELPHIA]
    c_5 is only used if there is windspeed data, otherwise it is ignored

    mpm_b = c_1 +c_2*(t_mod–25) +c_3*log10(g)*(t_k/t_stc_k) +c_4*g +c_5*ws

for mpm_a and mpm_b :
     g = (G_POA (W/m^2) / G_STC=1000 (W/m^2))  --> 'suns'
     t_mod = module temperature (C)
     ws = windspeed (ms^-1)

Note that both mpm_a or mpm_b can be used with either LFM_6 or LFM_4

    A later MPM version (not detailed here) can be used to model clipping and
other effects [See ref [8] Sutterlueti et al PVPMC 2022] 'mpm professional'

The pairs of functions "mpm_a_calc and mpm_b_calc", and 
"mpm_a_fit and mom_b_fit" should probably be merged but so far I haven't 
found a way to do this as they call each other and at least one combination
breaks.

Using DATAFRAMES or SERIES for variables
----------------------------------------

Many pvlib functions pass series of weather data separately for parameters e.g.
    poa_global, temp_module, wind_speed
and measurements such as
    pr_dc or p_mp

This mlfm code keeps all its met and measurement data in dataframes -
    meas, norm etc. e.g.

meas.columns
    Index(['module_id', 'poa_global', 'wind_speed', 'temp_air',
           'temp_module', 'v_oc', 'i_sc', 'i_mp', 'v_mp', 'r_sc',
           'r_oc', 'p_mp', 'pr_dc', 'v_oc_temp_corr', 'pr_dc_temp_corr'],
          dtype='object')

    It's easier when modelling all 6 or more measurement parameters in one
frame and then use an lfm_sel var to choose which to analyse
e.g. lfm_sel = 'pr_dc'

If individual series are needed to interface with existing code and
methodolgies they can be easily created by the following


#pvlib series  <-- mlfm dataframe
    poa_global   = meas['poa_global']
    temp_module  = meas['temp_module']
    wind_speed   = meas['wind_speed']
    pr_dc        = meas['pr_dc']

# mlfm dataframe      <-- pvlib series
    meas['poa_global']  = poa_global
    meas['temp_module'] = temp_module
    meas['wind_speed']  = wind_speed
    meas['pr_dc']       = pr_dc

DATAFRAME DEFINITIONS (for this python file and tutorials)
----------------------------------------------------------

A full definition is given here to keep the code in each function shorter

dmeas : DataFrame
-----------------
    Measured weather and module electrical values per time or measurement

    Parameters                                                      [units]
    ----------
    Index either -
        date_time : usually for external measurements or
        measurement_number : for indoor measurements e.g. IEC 61853

    * ``module_id`` - unique identifier to match data in ref       [alpha num]

    Weather measurements -

    * ``poa_global`` - global plane of array irradiance             [W/m^2]
    * ``temp_module`` - module temperature                          [C]
    * ``wind_speed`` - wind speed optional                          [m/s]

    [optional weather]

    * ``temp_air``  - air temperature optional                      [C]

    /Columns as needed by LFM_4 and/or LFM_6/ :

    * ``i_sc`` | 4 6 | current at short circuit condition           [A]
    * ``i_mp`` | 4 6 | current at maximum power point               [A]
    * ``v_mp`` | 4 6 | voltage at maximum power point               [V]
    * ``v_oc`` | 4 6 | voltage at open circuit condition            [V]

    * ``r_sc`` |   6 | -1/ (dI/dV|V=0) of IV curve at short circuit [Ohm]
    * ``r_oc`` |   6 | -1/(dI/dV|I=0) of IV curve at open circuit   [Ohm]

    Optional columns include

    * ``p_mp`` - power at maximum power point = i_mp * v_mp         [W]

ref : dict
----------
    Reference electrical and thermal datasheet module values at STC.

    Parameters                                                      [units]
    ----------
    Index
    * ``module_id`` - unique identifier to match data in dmeas     [alpha num]

    * ``p_mp``      - Max Power at Standard Test Condition (STC).   [W]
    * ``i_sc``      - Current at short circuit at STC.              [A]
    * ``i_mp``      - Current at max power at STC.                  [A]
    * ``v_mp``      - Voltage at max power at STC.                  [V]
    * ``v_oc``      - Voltage at open circuit at STC.               [V]
    * ``ff``        - Fill Factor                                   [1]

    * ``gamma_pdc`` - Temperature coefficient of max power point
                          power at STC.                             [1/C]
    * ``beta_v_oc`` - Temperature coefficient of open circuit
                          voltage at STC.                           [1/C]
    [optional thermal]

    * ``alpha_i_sc`` - Temperature coefficient of short circuit
                          current STC.                              [1/C]

    * ``alpha_i_mp`` - Temperature coefficient of max power point
                          current at STC.                           [1/C]

    * ``beta_v_mp``  - Temperature coefficient of max power point
                          voltage at STC.                           [1/C]

    [optional ID related]
    * ``source``        - Data Source                              [alpha num]
    * ``site``          - Sitename                                 [alpha num]
    * ``manufacturer``  - Module manufacturer                      [alpha num]
    * ``technology``    - Module technology e.g. cSi, HIT, CdTe    [alpha num]
    * ``module_type``   - Type ID e.g. ABC-123                     [alpha num]
    * ``module_serial`` - Serial number                            [alpha num]
    * ``comments``      - General comments                         [alpha num]


dnorm : DataFrame
-----------------
    Normalised multiplicative loss factors per parameter to model fall from
    start 1/ref_ff to meas pr_dc where -

    LFM_6 - multiplicative
    pr_dc = 1/ff * ( norm(i_sc) *norm(r_sc) *norm(i_ff)
                *norm(v_ff) *norm(r_oc) *norm(v_oc_t) *norm(temp_corr) ).

    LFM_4 - multiplicative
    pr_dc = 1/ff * ( norm(i_sc) *norm(i_mp)
                *norm(v_mp) *norm(v_oc_t) *norm(temp_corr) ).

    Parameters                                                        [units]
    ----------
    Index (copied from dmeas) either
        date_time : usually for external measurements or
        measurement_number : for indoor measurements e.g. IEC 61853

    * ``poa_global`` - global plane of array                          [W/m^2]
    * ``temp_module`` - module temperature                            [C]
    * ``wind_speed`` - wind speed optional                            [m/

    |Columns as used by LFM_4 and/or LFM_6| :

    * ``pr_dc``| 4 6 | Performance ratio dc.
                pr_dc = meas_p_mp / ref_p_mp /(poa_global/G_STC)      [%]
    * ``pr_dc_temp_corr``
               | 4 6 | pr_dc adjusted to 25C by gamma_p_mp.
    * ``i_sc`` | 4 6 | loss due to current at short circuit condition [%]
    * ``v_oc`` | 4 6 | Loss due to voltage at open circuit condition  [%]
    * ``v_oc_temp_corr``
               | 4 6 | v_oc adjusted to 25C by gamma_p_mp (not beta_v_oc)
                           for simplicity

    * ``i_mp`` | 4   | Loss due to current part of ff                 [%]
    * ``v_mp`` | 4   | Loss due to voltage part of ff                 [%]

    * ``r_sc`` |   6 | Loss due to r_sc ~r_shunt                      [%]
    * ``i_ff`` |   6 | Loss due to r_sc corrected current part of ff  [%]
    * ``v_ff`` |   6 | Loss due to r_oc corrected voltage part of ff  [%]
    * ``r_oc`` |   6 | Loss due to r_oc related to r_series           [%]

dstack : DataFrame
------------------
    Stacked subtractive normalized loss factors per parameter to model fall
    from start 1/ref_ff to meas pr_dc where -

    LFM_6 -  subtractive losses
    pr_dc = 1/ff - (stack(i_sc) +stack(r_sc) +stack(i_ff)
                +stack(v_   ff) +stack(r_oc) +stack(v_oc_t) +stack(temp_corr))

    LFM_4 -  subtractive losses
    pr_dc = 1/ff - (stack(i_sc) +stack(i_mp)
                +stack(v_mp) +stack(v_oc_t) +stack(temp_corr) ).

    Parameters                                                      [units]
    ----------
    Index (copied from dmeas)
        date_time : usually for external measurements or
        measurement_number : for indoor measurements e.g. IEC 61853

    * ``poa_global`` - global plane of array irradiance               [W/m^2]
    * ``temp_module`` - module temperature                            [C]
    * ``wind_speed`` - wind speed optional                            [m/

    |Columns as needed by LFM_4 and/or LFM_6| :

    * ``pr_dc`` equal to `dnorm['pr_dc']`

    * ``i_sc`` | 4 6 | loss due to current at short circuit condition [%]
    * ``v_oc`` | 4 6 | Loss due to voltage at open circuit condition  [%]
    * ``v_oc_temp_corr``
               | 4 6 | v_oc adjusted to 25C by gamma_p_mp (not beta_v_oc)
                           for simplicity

    * ``i_mp`` | 4   | Loss due to current part of ff                 [%]
    * ``v_mp`` | 4   | Loss due to voltage part of ff                 [%]

    * ``r_sc`` |   6 | Loss due to r_sc ~r_shunt                      [%]
    * ``i_ff`` |   6 | Loss due to r_sc corrected current part of ff  [%]
    * ``v_ff`` |   6 | Loss due to r_oc corrected voltage part of ff  [%]
    * ``r_oc`` |   6 | Loss due to r_oc related to r_series           [%]
"""

# DEFINE REFERENCE MEASUREMENT CONDITIONS
# or use existing definitions in pvlib. These might not all have
# been used in this code but are included for completeness

# NAME  value     # comment          unit       PV_LIB name

T_STC = 25.0      # STC temperature  [C]        temperature_ref
G_STC = 1000.0    # STC irradiance   [W/m^2]

# not all yet used below , added here for completeness
T_LIC = 25.0      # LIC temperature  [C]
G_LIC = 200.0     # LIC irradiance   [W/m^2]

T_HTC = 75.0      # HTC temperature  [C]
G_HTC = 1000.0    # HTC irradiance   [W/m^2]

T_PTC = 55.0      # HTC temperature  [C]
G_PTC = 1000.0    # HTC irradiance   [W/m^2]

G_LTC = 500.0    # HTC irradiance   [W/m^2]
T_LTC = 15.0      # LTC temperature  [C]

G_NOCT = 800      # NOCT irradiance  [W/m^2]
T_NOCT = 45       # NOCT temperature [C]

T_MAX = 100       # maximum temperature on right y axis

T0C_K = 273.15    # 0C  to Kelvin
T25C_K = 298.15   # 25C to Kelvin

#  Define standardised LFM graph colours as a dict ``CLR``
CLR = {
    # parameter_CLR colour            R   G   B
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


def meas_to_norm(dmeas, ref):
    """
    Convert measured P(W), I(A), V(V), R(Ohms) to values normalized to STC.

    Parameters
    ----------
    dmeas : DataFrame
        Measured weather and module electrical values per time or measurement.
        Contains 'poa_global', 'temp_module' and optional 'wind_speed'

    ref : dict
        Reference electrical and thermal datasheet module values at STC.

    Returns
    -------
    dnorm : DataFrame
        Normalised multiplicative loss values (values approx 1).
        Contains 'poa_global', 'temp_module' and optional 'wind_speed'

    References
    ----------
    .. [1] Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)
       'Quantifying Long Term PV Performance and Degradation under Real
       Outdoor and IEC 61853 Test Conditions Using High Quality Module
       IV Measurements' 36th EU PVSEC, Marseille, France. September 2019.

    """
    dnorm = pd.DataFrame()

    # copy weather data to meas dataframe for ease of use later
    dnorm['poa_global'] = dmeas['poa_global']
    dnorm['temp_module'] = dmeas['temp_module']
    dnorm['wind_speed'] = dmeas['wind_speed']

    dnorm['pr_dc'] = dmeas['p_mp']/ref['p_mp'] / (dmeas['poa_global']/G_STC)

    # calc temperature corrected pr_dc
    dnorm['pr_dc_temp_corr'] = (
        dnorm['pr_dc']
        * (1 - ref['gamma_pdc']*(dmeas['temp_module'] - T_STC)))

    # calculate normalised loss coefficients
    if 'i_sc' in dmeas.columns:
        dnorm['i_sc'] = (dmeas['i_sc'] / ref['i_sc']
                         / (dmeas['poa_global'] / G_STC))

        if 'i_mp' in dmeas.columns:
            dnorm['i_mp'] = dmeas['i_mp'] / dmeas['i_sc']

    if 'v_oc' in dmeas.columns:
        dnorm['v_oc'] = dmeas['v_oc'] / ref['v_oc']

        # temperature corrected
        dnorm['v_oc_temp_corr'] = (
                dnorm['v_oc']
                * (1 - ref['beta_v_oc']*(dmeas['temp_module'] - T_STC)))

        if 'v_mp' in dmeas.columns:
            dnorm['v_mp'] = dmeas['v_mp'] / dmeas['v_oc']

    if all(c in dmeas.columns for c in ['i_sc', 'v_oc', 'r_sc', 'r_oc']):
        ''' LFM_6 including r_sc and r_oc

        create temporary variables (i_r, v_r) from the
        intercept of r_sc (at i_sc) with r_oc (at v_oc)
        to make maths easier '''

        i_r = ((dmeas['i_sc'] * dmeas['r_sc'] - dmeas['v_oc'])
               / (dmeas['r_sc'] - dmeas['r_oc']))

        v_r = ((dmeas['r_sc'] * (dmeas['v_oc'] - dmeas['i_sc']
               * dmeas['r_oc']) / (dmeas['r_sc'] - dmeas['r_oc'])))

        # calculate normalised resistances r_sc and r_oc
        dnorm['r_sc'] = i_r / dmeas['i_sc']  # norm_r @ isc
        dnorm['r_oc'] = v_r / dmeas['v_oc']  # norm_r @ roc

        # calculate remaining fill factor losses partitioned to i_ff, v_ff
        dnorm['i_ff'] = dmeas['i_mp'] / i_r
        dnorm['v_ff'] = dmeas['v_mp'] / v_r

    return dnorm


def mpm_a_calc(dmeas, c_1, c_2, c_3, c_4, c_5=0., c_6=0.):
    """
    Predict norm LFM values from weather data (g,t,w) in ``dmeas``.

            const  temp_coeff     low_light   high_light wind  extra
             |     |              |             |        |      |
    norm = c_1 +c_2*(t_mod-25) +c_3*log10(g) +c_4*g +c_5*ws +c_6/g

    where :
        g = G_POA (W/m^2) / G_STC --> 'suns'
        t_mod = module temperature (C)
        ws = windspeed (ms^-1)

    Parameters                                                         [units]
    ----------
    dmeas : DataFrame
        Measured weather and module electrical values per time or measurement.
        Contains 'poa_global', 'temp_module' and optional 'wind_speed'.

    c_1 : float
        Constant term in model.                                            [%]
    c_2 : float
        Temperature coefficient in model.                                [1/C]
    c_3 : float
        Coefficient for low light log irradiance drop.                  [suns]
    c_4 : float
        Coefficient for high light linear irradiance drop.            [1/suns]
    c_5 : float, default 0
        Coefficient for wind speed dependence optional.              [1/(m/s)]
    c_6 : float, default 0                                              [suns]
        Coefficient for dependence on inverse irradiance.

    Returns
    -------
    mpm_a_out : Series
        Predicted values of mpm coefficient.

    References
    ----------
    .. [1] Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)
       "Quantifying Long Term PV Performance and Degradation under Real
       Outdoor        and IEC 61853 Test Conditions Using High Quality
       Module IV Measurements"
       36th EU PVSEC, Marseille, France. September 2019

    """
    mpm_a_out = (
        c_1
        + c_2 * (dmeas['temp_module'] - T_STC)
        + c_3 * np.log10(dmeas['poa_global'] / G_STC)
        + c_4 * (dmeas['poa_global'] / G_STC)
        + c_6 / (dmeas['poa_global'] / G_STC)
    )

    if 'wind_speed' in dmeas.columns:
        mpm_a_out += c_5 * dmeas['wind_speed']

    return mpm_a_out


def mpm_a_fit(data, var_to_fit):
    """
    Fit mpm_a to normalised measured data 'var_to_fit' using mpm_a model.

            const  temp_coeff     low_light   high_light wind  extra
             |     |              |             |        |      |
    fit = = c_1 +c_2*(t_mod-25) +c_3*log10(g) +c_4*g +c_5*ws +c_6/g

    where :
        g = G_POA (W/m^2) / G_STC --> 'suns'
        t_mod = module temperature (C)
        ws = windspeed (ms^-1)

    Parameters
    ----------
    data : DataFrame (see norm)
        Normalised multiplicative loss values (values approx 1).

    var_to_fit : string
        Column name in ``data`` containing variable being fitted.
        e.g. pr_dc, i_mp, v_mp, v_oc ...

    Returns
    -------
    pred : Series
        Values predicted by the fitted model.

    coeff : list
        Model coefficients ``c_1`` to ``c_6``.

    resid : Series
        Residuals of the fitted model.

    coeff_err : list
        Standard deviation of error in each model coefficient.

    See Also
    --------
    mpm_a_calc

    """
    # drop any missing data
    data = data.dropna()

    c5_zero = 'wind_speed' not in data.columns
    # if wind_speed is not present, add it and force it to 0
    if c5_zero:
        data['wind_speed'] = 0.

    # define function name
    func = mpm_a_calc

    # setup initial values and initial boundary conditions
    # init  c1   c2    c3    c4    c5   c6<0

    p_0 = (1.0, 0.01, 0.01, 0.01, 0.01, -0.01)
    # boundaries
    bounds = ([-2,  -2,  -2,  -2,  -2,   -2],
              [+2,  +2,  +2,  +2,  +2,    0])

    """
    # full_outputboolean, optional
    If True, this function returns additioal information:
        infodict, mesg, and ier.
    """

    coeff, pcov, infodict, mesg, ier = optimize.curve_fit(
        f=func,                  # fit function
        xdata=data,              # input data
        ydata=data[var_to_fit],  # fit parameter
        p0=p_0,                  # initial
        bounds=bounds,           # boundaries
        full_output=True
    )

    # if data had no wind_speed measurements then c_5 coefficient is
    # meaningless but a non-zero value may have been returned.
    if c5_zero:
        coeff[4] = 0.

    # get error of mpm coefficients as sqrt of covariance
    perr = np.sqrt(np.diag(pcov))
    coeff_err = list(perr)

    # save fit and error to dataframe
    pred = mpm_a_calc(data, *coeff)

    resid = pred - data[var_to_fit]

    return pred, coeff, resid, coeff_err, infodict, mesg, ier


def mpm_b_fit(data, var_to_fit):
    """
    Fit mpm_b to normalised measured data 'var_to_fit' using mpm_b model.

        const  temp_coeff     low_light  improvement high_light     ws
          |     |               |               |           |       |
    fit =c_1 +c_2*(t_mod–25) +c_3*log10(g)*(t_k/t_stc_k) +c_4*g +c_5*ws

        where :
        g = G_POA (W/m^2) / G_STC --> 'suns'
        t_mod = module temperature (C)
        ws = windspeed (ms^-1)

    Parameters
    ----------
    data : DataFrame (see norm)
        Normalised multiplicative loss values (values approx 1).

    var_to_fit : string
        Column name in ``data`` containing variable being fitted.
        e.g. pr_dc, i_mp, v_mp ...

    Returns
    -------
    pred : Series
        Values predicted by the fitted model.

    coeff : list
        Model coefficients ``c_1`` to ``c_5``.

    resid : Series
        Residuals of the fitted model.

    coeff_err : list
        Standard deviation of error in each model coefficient.

    See Also
    --------
    mpm_a

    """
    # drop missing data
    data = data.dropna()

    # define function name
    func = mpm_b_calc

    # setup initial values and initial boundary conditions
    # init  c1   c2    c3    c4    c5

    p_0 = (1.0, 0.01, 0.01, 0.01, 0.01)
    # boundaries
    bounds = ([-2,  -2,  -2,  -2,  -2],
              [+2,  +2,  +2,  +2,  +2])

    coeff, pcov, infodict, mesg, ier = optimize.curve_fit(
        f=func,                  # fit function
        xdata=data,              # input data
        ydata=data[var_to_fit],  # fit parameter
        p0=p_0,                  # initial
        bounds=bounds,           # boundaries
        full_output=True
    )

    # get error of mpm coefficients as sqrt of covariance
    perr = np.sqrt(np.diag(pcov))
    coeff_err = list(perr)

    # save fit and error to dataframe
    pred = mpm_b_calc(data, *coeff)

    resid = pred - data[var_to_fit]

    # fvec = infodict["fvec"]

    return pred, coeff, resid, coeff_err, infodict, mesg, ier


def mpm_b_calc(dmeas, c_1, c_2, c_3, c_4, c_5=0.):
    """
    Predict normalised LFM values from weather data (g,t,w) in ``dmeas``.

         const  temp_coeff     low_light  improvement high_light     ws
           |     |               |               |           |       |
    norm =c_1 +c_2*(t_mod–25) +c_3*log10(g)*(t_k/t_stc_k) +c_4*g +c_5*ws

    where :
        g = G_POA (W/m^2) / G_STC --> 'suns'
        t_mod = module temperature (C)
        ws = windspeed (ms^-1)

    Parameters                                                         [units]
    ----------
    dmeas : DataFrame
        Measured weather and module electrical values per time or measurement.
        Contains 'poa_global', 'temp_module' and optional 'wind_speed'.

    c_1 : float
        Constant term in model.                                            [%]
    c_2 : float
        Temperature coefficient in model.                                [1/C]
    c_3 : float
        Coefficient for low light log irradiance drop.                  [suns]
    c_4 : float
        Coefficient for high light linear irradiance drop.            [1/suns]
    c_5 : float, default 0
        Coefficient for wind speed dependence optional.              [1/(m/s)]

    Returns
    -------
    mpm_b_out : Series
        Predicted values of mpm coefficient.

    References
    ----------
    .. [1] Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)
       "Quantifying Long Term PV Performance and Degradation under Real
       Outdoor and IEC 61853 Test Conditions Using High Quality Module 
       IV Measurements"
       36th EU PVSEC, Marseille, France. September 2019

    """
    mpm_b_out = (
        c_1
        + c_2 * (dmeas['temp_module'] - T_STC)
        + c_3 * ((np.log10(dmeas['poa_global'] / G_STC)
                  * (dmeas['temp_module'] + T0C_K) / T25C_K))
        + c_4 * (dmeas['poa_global'] / G_STC)
    )

    return mpm_b_out


def plot_scatter(dnorm, title, qty_lfm_vars, save_figs=False):
    """
    Scatterplot of normalised values (y) vs. irradiance (x).

    Electrical quantities are plotted on the left y-axis, temperature
    quantities are plotted on the right y-axis.

    Parameters
    ----------
    dnorm : DataFrame
        Normalised multiplicative loss values (values approx 1).
        Contains 'poa_global', 'temp_module' and optional 'wind_speed'

    title : string
        Title for the figure.

    qty_lfm_vars : int
        number of lfm_vars : 6=iv with rsc, roc ; 4=indoor

    save_figs : boolean
        save a high resolution png file of figure

    Returns
    -------
    fig : Figure
        Instance of matplotlib.figure.Figure

    See Also
    --------
    meas_to_norm

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('plot_scatter requires matplotlib')

    # offset legend to the right to not overlap graph, use ~1.2
    bbox = 1.2

    # set x_axis as irradiance in W/m2
    xdata = dnorm['poa_global']

    fig, ax1 = plt.subplots()

    ax1.set_title(title)

    ax1.set_ylabel('Normalised values')
    ax1.axhline(y=1, c='grey', linewidth=3)  # show 100% line

    # optional normalised y scale usually ~0.8 to 1.1
    ax1.set_ylim(0.8, 1.1)

    ax1.set_xlabel('Plane of array irradiance [W/m$^2$]')
    ax1.axvline(x=G_STC, c='grey', linewidth=3)  # show 1000W/m^2 STC
    ax1.axvline(x=G_NOCT, c='grey', linewidth=3)  # show 800W/m^2 NOCT
    ax1.axvline(x=G_LIC, c='grey', linewidth=3)  # show 200W/m^2 LIC

    # check which lines to plot
    if qty_lfm_vars == 6:
        # LFM_6
        lines = {
            'pr_dc_temp_corr': 'pr_dc',
            'i_sc': 'i_sc',
            'r_sc': 'r_sc',
            'r_oc': 'r_oc',
            'i_ff': 'i_ff',
            'v_ff': 'v_ff',
            'v_oc_temp_corr': 'v_oc'}

        labels = {
            'pr_dc_temp_corr': 'pr_dc_temp_corr',
            'i_sc': 'norm_i_sc',
            'r_sc': 'norm_r_sc',
            'r_oc': 'norm_r_oc',
            'i_ff': 'norm_i_ff',
            'v_ff': 'norm_v_ff',
            'v_oc_temp_corr': 'norm_v_oc_temp_corr'}

    elif qty_lfm_vars == 4:
        # LFM_4
        lines = {
            'pr_dc_temp_corr': 'pr_dc',
            'i_mp': 'i_mp',
            'v_mp': 'v_mp',
            'i_sc': 'i_sc',
            'v_oc_temp_corr': 'v_oc'}

        labels = {
            'pr_dc_temp_corr': 'pr_dc_temp_corr',
            'i_mp': 'norm_i_mp',
            'v_mp': 'norm_v_mp',
            'i_sc': 'norm_i_sc',
            'v_oc_temp_corr': 'norm_v_oc_temp_corr'}

    # plot the LFM parameters depending on qty_lfm_vars
    for k in lines.keys():
        try:
            ax1.scatter(xdata, dnorm[k], c=CLR[lines[k]], label=labels[k])
        except KeyError:
            pass

    ax1.legend(bbox_to_anchor=(bbox, 1),
               loc='upper left', borderaxespad=0.)

    # y2axis plot met on right y axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature (C/100)')

    # set wide limits 0 to 4 so they don't overlap with LFM params
    ax2.set_ylim(0, 4)

    ax2.scatter(xdata,
                dnorm['temp_module']/T_MAX,
                c=CLR['temp_module'],
                label='temp_module C/' + str(T_MAX))

    # temp_air may not exist particularly for indoor measurements
    try:
        ax2.scatter(xdata,
                    dnorm['temp_air']/T_MAX,
                    c=CLR['temp_air'],
                    label='temp_air C/' + str(T_MAX))
    except KeyError:
        pass

    # make second legend box low enough ~0.1 not to overlap first box
    ax2.legend(bbox_to_anchor=(bbox, 0.1),
               loc='upper left', borderaxespad=0.)

    if save_figs:
        # remove '.csv', high resolution= 300 dots per inch
        plt.savefig(os.path.join('mlfm_data', 'output',
                    'scatter_' + title[:len(title)-4]), dpi=300)

    plt.show()

    return fig


def plot_stack(dstack, fill_factor, title,
               xaxis_labels=0, is_i_sc_self_ref=False,
               save_figs=False
               ):
    """
    Plot stacked subtractive losses from 1/ref_ff down to pr_dc.

    Parameters
    ----------
    dstack : DataFrame
        Stacked subtractive losses.

    fill_factor : float
        Reference value of fill factor for IV curve at STC conditions.

    title : string
        Title for the figure.

    xaxis_labels : int, default 0
        Number of x-axis labels to show. Default 0 shows all.

    is_i_sc_self_ref : bool, default False
       Self-correct ``i_sc`` to remove angle of incidence,
       spectrum, snow or soiling.

    save_figs : boolean
        save a high resolution png file of figure

    # is_v_oc_temp_module_corr : bool, default True
    #    Calculate loss due to temperature and subtract from ``v_oc`` loss.

    Returns
    -------
    fig : Figure
        Instance of matplotlib.figure.Figure

    See Also
    --------
    norm_to_stack

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('plt_stack requires matplotlib')

    # label names for LFM_6
    stack6 = ['i_sc', 'r_sc', 'i_ff', 'i_v',
              'v_ff', 'r_oc', 'v_oc_temp_corr']

    if all([c in dstack.columns for c in stack6]):

        # data order from bottom to top
        ydata = [dstack['pr_dc'] + (dstack['i_sc'] * (is_i_sc_self_ref)),
                 dstack['v_oc_temp_corr'],
                 dstack['temp_module_corr'],
                 dstack['r_oc'],
                 dstack['v_ff'],
                 dstack['i_v'],
                 dstack['i_ff'],
                 dstack['r_sc'],
                 dstack['i_sc'] * (not is_i_sc_self_ref)]

        labels = [
            'pr_dc',
            'stack_t_mod',
            'stack_v_oc',
            'stack_r_oc',
            'stack_v_ff',
            '- - -',
            'stack_i_ff',
            'stack_r_sc',
            'stack_i_sc']

        color_map = [
            'white',  # colour to bottom of graph
            CLR['temp_module'],
            CLR['v_oc'],
            CLR['r_oc'],
            CLR['v_ff'],
            CLR['i_v'],
            CLR['i_ff'],
            CLR['r_sc'],
            CLR['i_sc']]

    stack4 = ['i_sc', 'i_mp', 'i_v',
              'v_mp', 'v_oc_temp_corr']

    if all([c in dstack.columns for c in stack4]):

        # data order from bottom to top
        ydata = [dstack['pr_dc'] + (dstack['i_sc'] * (is_i_sc_self_ref)),
                 dstack['v_oc_temp_corr'],
                 dstack['temp_module_corr'],
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
            CLR['temp_module'],
            CLR['v_oc'],
            CLR['v_mp'],
            CLR['i_v'],
            CLR['i_mp'],
            CLR['i_sc']]

    # offset legend right, use ~1.2
    bbox = 1.2

    # select x axis usually date_time
    xdata = dstack.index.values
    fig, ax1 = plt.subplots()

    ax1.set_title(title)

    # plot stack in order bottom to top,
    # allowing self_ref and temp_module corrections
    ax1.stackplot(xdata, *tuple(ydata), labels=labels, colors=color_map)

    ax1.axhline(y=1/fill_factor, c='grey', lw=3)  # show initial 1/FF
    ax1.axhline(y=1, c='grey', lw=3)  # show 100% line
    ax1.set_ylabel('stacked lfm losses')

    # find number of x date values
    x_ticks = dstack.shape[0]
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
            except TypeError:  # xdata can't be subscripted
                xax2[x_count] = xdata[0]

        x_count += 1

    ax1.set_xticklabels(xax2)
    ax1.set_ylim(0.6, 1/fill_factor + 0.1)  # optional normalised y scale
    plt.legend(bbox_to_anchor=(bbox, 1), loc='upper left', borderaxespad=0.)

    # plot met data on right y axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('poa_global (kW/m^2), temp_module (C/ ' + str(T_MAX))
    ax2.set_ylim(0, 4)  # set so doesn't overlap lfm params

    plt.plot(xdata, dstack['poa_global'] / G_STC,
             c=CLR['irradiance'], label='poa_global (kW/m^2)')
    plt.plot(xdata, dstack['temp_module'] / T_MAX,
             c=CLR['temp_module'], label='temp_module / ' + str(T_MAX))

    # temp_air may not exist particularly for indoor measurements
    try:
        plt.plot(xdata, dstack['temp_air']/100,
                 c=CLR['temp_air'], label='temp_air/ ' + str(T_MAX))
    except KeyError:
        pass

    ax2.legend(bbox_to_anchor=(bbox, 0.3), loc='upper left', borderaxespad=0.)
    ax1.set_xticklabels(xax2, rotation=90)

    # remove '.csv', high resolution= 300 dots per inch
    plt.savefig(os.path.join('mlfm_data', 'output',
                'stack_' + title[:len(title)-4]), dpi=300)

    return fig


def meas_to_stack_lin(dmeas, ref, qty_lfm_vars, gap=0.01):
    """
    Convert measured values to stacked subtractive normalized losses.

    Stacked subtractive losses show the relative loss proportions
    from max possible "ref_i_sc * ref_v_oc" (1/reference fill factor)
    to the measured normalized power.

    This version is done in a linear fashion so that LFM4 and LFM6 give the
    same answers for Isc and Voc and the loss(i_mp)=loss(r_sc)+loss(i_ff)

    Parameters
    ----------
    dmeas : DataFrame
        Measured weather and module electrical values per time or measurement.
        Contains 'poa_global', 'temp_module' and optional 'wind_speed'.

    ref : dict
        Reference electrical and thermal datasheet module values at STC.

    gap : float
        create a gap to differentiate i and v losses ~ 0.01

    qty_lfm_vars : int
        number of lfm_vars : 6=iv with rsc, roc ; 4=without rsc, roc

    Returns
    -------
    dstack : DataFrame
        Stacked subtractive normalized losses

    See Also
    --------
    meas_to_norm

    References
    ----------
    .. [1] Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)
       "Quantifying Long Term PV Performance and Degradation under Real 
       Outdoor and IEC 61853 Test Conditions Using High Quality Module 
       IV Measurements" 36th EU PVSEC, Marseille, France. September 2019
    """
    # create an empty DataFrame to put stack results
    dstack = pd.DataFrame()

    # copy weather data for ease of use
    dstack['poa_global'] = dmeas['poa_global']
    dstack['temp_module'] = dmeas['temp_module']
    dstack['wind_speed'] = dmeas['wind_speed']

    # ref['p_mp'] = ref['i_mp'] * ref['v_mp']

    # ref['ff'] = ref['p_mp'] / (ref['i_sc'] * ref['v_oc'])

    # ref['ff'] = (ref['i_mp']*ref['v_mp'])/(ref['i_sc']*ref['v_oc'])
    inv_ff = 1 / ref['ff']

    dstack['pr_dc'] = dmeas['pr_dc']

    # Find linear values on i and v axes normalised to i_mp, v_mp
    lin_i_ratio = ref['i_sc']/ref['i_mp']
    lin_v_ratio = ref['v_oc']/ref['v_mp']

    lin_i_sc = dmeas['i_sc']/ref['i_mp']/(dmeas['poa_global']/G_STC)

    lin_v_oc = dmeas['v_oc']/ref['v_mp']
    lin_v_oc_temp_corr = dmeas['v_oc_temp_corr']/ref['v_mp']

    # transform multiplicative to subtractive losses find
    # correction factor to scale losses to keep 1/ff --> pr_dc

    if qty_lfm_vars == 6:
        # subtractive losses with series and shunt resistance effects
        i_r = ((dmeas['i_sc'] * dmeas['r_sc'] - dmeas['v_oc']) /
               (dmeas['r_sc'] - dmeas['r_oc']))

        v_r = ((dmeas['r_sc'] * (dmeas['v_oc'] - dmeas['i_sc'] *
                dmeas['r_oc']) / (dmeas['r_sc'] - dmeas['r_oc'])))

        lin_i_r = i_r/ref['i_mp'] / (dmeas['poa_global']/G_STC)
        lin_i_ff = dmeas['i_mp'] / ref['i_mp']/(dmeas['poa_global']/G_STC)

        lin_v_ff = dmeas['v_mp'] / ref['v_mp']
        lin_v_r = v_r / ref['v_mp']

        sub_i = lin_i_ratio - lin_i_ff  # current drop
        sub_v = lin_v_ratio - lin_v_ff  # voltage drop

        # correction factor mult --> lin loss
        corr = (inv_ff - dstack['pr_dc']) / (sub_i + sub_v)

        # put 6 LFM values in a stack from pr_dc (bottom) to 1/ff_ref (top)
        # accounting for series and shunt resistance losses

        dstack['i_sc'] = (lin_i_ratio-lin_i_sc) * corr
        dstack['r_sc'] = (lin_i_sc-lin_i_r) * corr
        dstack['i_ff'] = (lin_i_r-lin_i_ff) * corr - gap/2
        dstack['i_v'] = gap
        dstack['v_ff'] = (lin_v_r-lin_v_ff) * corr - gap/2
        dstack['r_oc'] = (lin_v_oc-lin_v_r) * corr
        dstack['v_oc_temp_corr'] = (lin_v_oc_temp_corr-lin_v_oc) * corr
        dstack['temp_module_corr'] = (lin_v_ratio-lin_v_oc_temp_corr) * corr

    if qty_lfm_vars == 4:

        lin_i_mp = dmeas['i_mp'] / ref['i_mp'] / (dmeas['poa_global']/G_STC)
        lin_v_mp = dmeas['v_mp'] / ref['v_mp']

        sub_i = lin_i_ratio - lin_i_mp  # current drop
        sub_v = lin_v_ratio - lin_v_mp  # voltage drop

        # correction factor mult --> lin loss
        corr = (inv_ff-dstack['pr_dc']) / (sub_i + sub_v)

        # put 4 LFM values in a stack from pr_dc (bottom) to 1/ff_ref (top)
        # accounting for series and shunt resistance losse

        dstack['i_sc'] = (lin_i_ratio-lin_i_sc) * corr
        dstack['i_mp'] = (lin_i_sc-lin_i_mp) * corr - gap/2
        dstack['i_v'] = gap
        dstack['v_mp'] = (lin_v_oc-lin_v_mp) * corr - gap/2
        dstack['v_oc_temp_corr'] = (lin_v_oc_temp_corr-lin_v_oc) * corr
        dstack['temp_module_corr'] = (lin_v_ratio-lin_v_oc_temp_corr) * corr

    return dstack


"""
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

.. [7] Steve Ransome (SRCL)
'Benchmarking PV performance models with high quality IEC 61853 Matrix
measurements (Bilinear interpolation, SAPM, PVGIS, MLFM and 1-diode)'
http://www.steveransome.com/pubs/2206_PVSC49_philadelphia_4_presented.pdf

.. [8] Juergen Sutterlueti (Gantner Instruments)
'Advanced system monitoring and artificial intelligent data-driven analytics
to serve GW-scale photovoltaic power plant and energy storage requirements'
https://pvpmc.sandia.gov/download/8574/

"""
