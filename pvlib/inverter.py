# -*- coding: utf-8 -*-
"""
This module contains functions for inverter modeling, primarily conversion of
DC to AC power.
"""

import numpy as np
import pandas as pd


def sandia(v_dc, p_dc, inverter):
    r'''
    Converts DC power and voltage to AC power using Sandia's
    Grid-Connected PV Inverter model.

    Parameters
    ----------
    v_dc : numeric
        DC voltages, in volts, which are provided as input to the
        inverter. Vdc should be >= 0.

    p_dc : numeric
        A scalar or DataFrame of DC powers, in watts, which are provided
        as input to the inverter. Pdc should be >= 0.

    inverter : dict-like
        A dict-like object defining the inverter to be used, giving the
        inverter performance parameters according to the Sandia
        Grid-Connected Photovoltaic Inverter Model (SAND 2007-5036) [1]_.
        A set of inverter performance parameters are provided with
        pvlib, or may be generated from a System Advisor Model (SAM) [2]_
        library using retrievesam. See Notes for required keys.

    Returns
    -------
    ac_power : numeric
        Modeled AC power output given the input DC voltage, Vdc, and
        input DC power, Pdc. When ac_power would be greater than Pac0,
        it is set to Pac0 to represent inverter "clipping". When
        ac_power would be less than Ps0 (startup power required), then
        ac_power is set to -1*abs(Pnt) to represent nightly power
        losses. ac_power is not adjusted for maximum power point
        tracking (MPPT) voltage windows or maximum current limits of the
        inverter.

    Notes
    -----

    Determines the AC power output of an inverter given the DC voltage and DC
    power. Output AC power is clipped at the inverter's maximum power output
    and output power can be negative during low-input power conditions. The
    Sandia inverter model does NOT account for maximum power point
    tracking voltage windows nor maximum current or voltage limits on
    the inverter.

    Required inverter keys are:

    ======   ============================================================
    Column   Description
    ======   ============================================================
    Pac0     AC-power output from inverter based on input power
             and voltage (W)
    Pdc0     DC-power input to inverter, typically assumed to be equal
             to the PV array maximum power (W)
    Vdc0     DC-voltage level at which the AC-power rating is achieved
             at the reference operating condition (V)
    Ps0      DC-power required to start the inversion process, or
             self-consumption by inverter, strongly influences inverter
             efficiency at low power levels (W)
    C0       Parameter defining the curvature (parabolic) of the
             relationship between ac-power and dc-power at the reference
             operating condition, default value of zero gives a
             linear relationship (1/W)
    C1       Empirical coefficient allowing Pdco to vary linearly
             with dc-voltage input, default value is zero (1/V)
    C2       Empirical coefficient allowing Pso to vary linearly with
             dc-voltage input, default value is zero (1/V)
    C3       Empirical coefficient allowing Co to vary linearly with
             dc-voltage input, default value is zero (1/V)
    Pnt      AC-power consumed by inverter at night (night tare) to
             maintain circuitry required to sense PV array voltage (W)
    ======   ============================================================

    References
    ----------
    .. [1] SAND2007-5036, "Performance Model for Grid-Connected
       Photovoltaic Inverters by D. King, S. Gonzalez, G. Galbraith, W.
       Boyson

    .. [2] System Advisor Model web page. https://sam.nrel.gov.

    See also
    --------
    pvlib.pvsystem.sapm
    '''

    Paco = inverter['Paco']
    Pdco = inverter['Pdco']
    Vdco = inverter['Vdco']
    Pso = inverter['Pso']
    C0 = inverter['C0']
    C1 = inverter['C1']
    C2 = inverter['C2']
    C3 = inverter['C3']
    Pnt = inverter['Pnt']

    A = Pdco * (1 + C1*(v_dc - Vdco))
    B = Pso * (1 + C2*(v_dc - Vdco))
    C = C0 * (1 + C3*(v_dc - Vdco))

    ac_power = (Paco/(A-B) - C*(A-B)) * (p_dc-B) + C*((p_dc-B)**2)
    ac_power = np.minimum(Paco, ac_power)
    ac_power = np.where(p_dc < Pso, -1.0 * abs(Pnt), ac_power)

    if isinstance(p_dc, pd.Series):
        ac_power = pd.Series(ac_power, index=p_dc.index)

    return ac_power


def adrinverter(v_dc, p_dc, inverter, vtol=0.10):
    r'''
    Converts DC power and voltage to AC power using Anton Driesse's
    Grid-Connected PV Inverter efficiency model

    Parameters
    ----------
    v_dc : numeric
        A scalar or pandas series of DC voltages, in volts, which are provided
        as input to the inverter. If Vdc and Pdc are vectors, they must be
        of the same size. v_dc must be >= 0. [V]

    p_dc : numeric
        A scalar or pandas series of DC powers, in watts, which are provided
        as input to the inverter. If Vdc and Pdc are vectors, they must be
        of the same size. p_dc must be >= 0. [W]

    inverter : dict-like
        A dict-like object defining the inverter to be used, giving the
        inverter performance parameters according to the model
        developed by Anton Driesse [1]_.
        A set of inverter performance parameters may be loaded from the
        supplied data table using retrievesam.
        See Notes for required keys.

    vtol : numeric, default 0.1
        A unit-less fraction that determines how far the efficiency model is
        allowed to extrapolate beyond the inverter's normal input voltage
        operating range. 0.0 <= vtol <= 1.0

    Returns
    -------
    ac_power : numeric
        A numpy array or pandas series of modeled AC power output given the
        input DC voltage, v_dc, and input DC power, p_dc. When ac_power would
        be greater than pac_max, it is set to p_max to represent inverter
        "clipping". When ac_power would be less than -p_nt (energy consumed
        rather  than produced) then ac_power is set to -p_nt to represent
        nightly power losses. ac_power is not adjusted for maximum power point
        tracking (MPPT) voltage windows or maximum current limits of the
        inverter. [W]

    Notes
    -----

    Required inverter keys are:

    =======   ============================================================
    Column    Description
    =======   ============================================================
    Pnom      The nominal power value used to normalize all power values,
              typically the DC power needed to produce maximum AC power
              output. [W]

    Vnom      The nominal DC voltage value used to normalize DC voltage
              values, typically the level at which the highest efficiency
              is achieved. [V]

    Vmax      . [V]

    Vmin      . [V]

    Vdcmax    . [V]

    MPPTHi    . [unit]

    MPPTLow    . [unit]

    Pacmax    The maximum AC output power value, used to clip the output
              if needed, (W).

    ADRCoefficients  A list of 9 coefficients that capture the influence
              of input voltage and power on inverter losses, and thereby
              efficiency.

    Pnt       AC power consumed by inverter at night (night tare) to
              maintain circuitry required to sense PV array voltage. [W]

    =======   ============================================================

    References
    ----------
    .. [1] Beyond the Curves: Modeling the Electrical Efficiency
       of Photovoltaic Inverters, PVSC 2008, Anton Driesse et. al.

    See also
    --------
    pvlib.inverter.sandia
    '''

    p_nom = inverter['Pnom']
    v_nom = inverter['Vnom']
    pac_max = inverter['Pacmax']
    p_nt = inverter['Pnt']
    ce_list = inverter['ADRCoefficients']
    v_max = inverter['Vmax']
    v_min = inverter['Vmin']
    vdc_max = inverter['Vdcmax']
    mppt_hi = inverter['MPPTHi']
    mppt_low = inverter['MPPTLow']

    v_lim_upper = float(np.nanmax([v_max, vdc_max, mppt_hi]) * (1 + vtol))
    v_lim_lower = float(np.nanmax([v_min, mppt_low]) * (1 - vtol))

    pdc = p_dc / p_nom
    vdc = v_dc / v_nom
    # zero voltage will lead to division by zero, but since power is
    # set to night time value later, these errors can be safely ignored
    with np.errstate(invalid='ignore', divide='ignore'):
        poly = np.array([pdc**0,  # replace with np.ones_like?
                         pdc,
                         pdc**2,
                         vdc - 1,
                         pdc * (vdc - 1),
                         pdc**2 * (vdc - 1),
                         1. / vdc - 1,  # divide by 0
                         pdc * (1. / vdc - 1),  # invalid 0./0. --> nan
                         pdc**2 * (1. / vdc - 1)])  # divide by 0
    p_loss = np.dot(np.array(ce_list), poly)
    ac_power = p_nom * (pdc-p_loss)
    p_nt = -1 * np.absolute(p_nt)

    # set output to nan where input is outside of limits
    # errstate silences case where input is nan
    with np.errstate(invalid='ignore'):
        invalid = (v_lim_upper < v_dc) | (v_dc < v_lim_lower)
    ac_power = np.where(invalid, np.nan, ac_power)

    # set night values
    ac_power = np.where(vdc == 0, p_nt, ac_power)
    ac_power = np.maximum(ac_power, p_nt)

    # set max ac output
    ac_power = np.minimum(ac_power, pac_max)

    if isinstance(p_dc, pd.Series):
        ac_power = pd.Series(ac_power, index=pdc.index)

    return ac_power



def pvwatts_ac(pdc, pdc0, eta_inv_nom=0.96, eta_inv_ref=0.9637):
    r"""
    Implements NREL's PVWatts inverter model [1]_.

    .. math::

        \eta = \frac{\eta_{nom}}{\eta_{ref}} (-0.0162\zeta - \frac{0.0059}
        {\zeta} + 0.9858)

    .. math::

        P_{ac} = \min(\eta P_{dc}, P_{ac0})

    where :math:`\zeta=P_{dc}/P_{dc0}` and :math:`P_{dc0}=P_{ac0}/\eta_{nom}`.

    Note that  pdc0 is also used as a symbol in
    :py:func:`pvlib.pvsystem.pvwatts_dc`. pdc0 in this function refers to the
    DC power input limit of the inverter. pdc0 in
    :py:func:`pvlib.pvsystem.pvwatts_dc` refers to the DC power of the module's
    at reference conditions.

    Parameters
    ----------
    pdc: numeric
        DC power.
    pdc0: numeric
        DC input limit of the inverter.
    eta_inv_nom: numeric, default 0.96
        Nominal inverter efficiency.
    eta_inv_ref: numeric, default 0.9637
        Reference inverter efficiency. PVWatts defines it to be 0.9637
        and is included here for flexibility.

    Returns
    -------
    pac: numeric
        AC power.

    References
    ----------
    .. [1] A. P. Dobos, "PVWatts Version 5 Manual,"
           http://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
           (2014).
    """

    pac0 = eta_inv_nom * pdc0
    zeta = pdc / pdc0

    # arrays to help avoid divide by 0 for scalar and array
    eta = np.zeros_like(pdc, dtype=float)
    pdc_neq_0 = ~np.equal(pdc, 0)

    # eta < 0 if zeta < 0.006. pac is forced to be >= 0 below. GH 541
    eta = eta_inv_nom / eta_inv_ref * (
        - 0.0162*zeta
        - np.divide(0.0059, zeta, out=eta, where=pdc_neq_0)
        + 0.9858)

    pac = eta * pdc
    pac = np.minimum(pac0, pac)
    pac = np.maximum(0, pac)     # GH 541

    return pac