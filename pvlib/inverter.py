# -*- coding: utf-8 -*-
"""
This module contains functions for inverter modeling, primarily conversion of
DC to AC power.
"""

import numpy as np
import pandas as pd


def sandia(v_dc, p_dc, inverter):
    r'''
    Convert DC power and voltage to AC power using Sandia's
    Grid-Connected PV Inverter model.

    Parameters
    ----------
    v_dc : numeric
        DC voltage input to the inverter. [V]

    p_dc : numeric
        DC power input to the inverter. [W]

    inverter : dict-like
        Defines parameters for the inverter model in [1]_.  See Notes for
        required model parameters. A copy of the parameter database from the
        System Advisor Model (SAM) [2]_ is provided with pvlib and may be read
        using :py:func:`pvlib.pvsystem.retrieve_sam`.

    Returns
    -------
    power_ac : numeric
        AC power output. [W]

    Notes
    -----

    Determines the AC power output of an inverter given the DC voltage and DC
    power. Output AC power is bounded above by the parameter ``Paco``, to
    represent inverter "clipping".  When `power_ac` would be less than
    parameter ``Pso`` (startup power required), then `power_ac` is set to
    ``-Pnt``, representing self-consumption. `power_ac` is not adjusted for
    maximum power point tracking (MPPT) voltage windows or maximum current
    limits of the inverter.

    Required model parameters are:

    ======   ============================================================
    Column   Description
    ======   ============================================================
    Paco     AC power rating of the inverter. [W]
    Pdco     DC power input to inverter, typically assumed to be equal
             to the PV array maximum power. [W]
    Vdco     DC voltage at which the AC power rating is achieved
             at the reference operating condition. [V]
    Pso      DC power required to start the inversion process, or
             self-consumption by inverter, strongly influences inverter
             efficiency at low power levels. [W]
    C0       Parameter defining the curvature (parabolic) of the
             relationship between AC power and DC power at the reference
             operating condition. [1/W]
    C1       Empirical coefficient allowing ``Pdco`` to vary linearly
             with DC voltage input. [1/V]
    C2       Empirical coefficient allowing ``Pso`` to vary linearly with
             DC voltage input. [1/V]
    C3       Empirical coefficient allowing ``C0`` to vary linearly with
             DC voltage input. [1/V]
    Pnt      AC power consumed by the inverter at night (night tare). [W]
    ======   ============================================================

    References
    ----------
    .. [1] D. King, S. Gonzalez, G. Galbraith, W. Boyson, "Performance Model
       for Grid-Connected Photovoltaic Inverters", SAND2007-5036, Sandia
       National Laboratories.

    .. [2] System Advisor Model web page. https://sam.nrel.gov.

    See also
    --------
    pvlib.pvsystem.retrieve_sam
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

    A = Pdco * (1 + C1 * (v_dc - Vdco))
    B = Pso * (1 + C2 * (v_dc - Vdco))
    C = C0 * (1 + C3 * (v_dc - Vdco))

    power_ac = (Paco / (A - B) - C * (A - B)) * (p_dc - B) + C * (p_dc - B)**2
    power_ac = np.minimum(Paco, power_ac)
    power_ac = np.where(p_dc < Pso, -1.0 * abs(Pnt), power_ac)

    if isinstance(p_dc, pd.Series):
        power_ac = pd.Series(power_ac, index=p_dc.index)

    return power_ac


def adr(v_dc, p_dc, inverter, vtol=0.10):
    r'''
    Converts DC power and voltage to AC power using Anton Driesse's
    Grid-Connected PV Inverter efficiency model

    Parameters
    ----------
    v_dc : numeric
        DC voltage input to the inverter, should be >= 0. [V]

    p_dc : numeric
        DC power input to the inverter, should be >= 0. [W]

    inverter : dict-like
        Defines parameters for the inverter model in [1]_.  See Notes for
        required model parameters. A parameter database is provided with pvlib
        and may be read using :py:func:`pvlib.pvsystem.retrieve_sam`.

    vtol : numeric, default 0.1
        Fraction of DC voltage that determines how far the efficiency model is
        extrapolated beyond the inverter's normal input voltage operating
        range. 0.0 <= vtol <= 1.0. [unitless]

    Returns
    -------
    power_ac : numeric
        AC power output. [W]

    Notes
    -----

    Determines the AC power output of an inverter given the DC voltage and DC
    power. Output AC power is bounded above by the parameter ``Pacmax``, to
    represent inverter "clipping". AC power is bounded below by ``-Pnt``
    (negative when power is consumed rather than produced) which represents
    self-consumption. `power_ac` is not adjusted for maximum power point
    tracking (MPPT) voltage windows or maximum current limits of the inverter.

    Required model parameters are:

    =======   ============================================================
    Column    Description
    =======   ============================================================
    Pnom      Nominal DC power, typically the DC power needed to produce
              maximum AC power output. [W]
    Vnom      Nominal DC input voltage. Typically the level at which the
              highest efficiency is achieved. [V]
    Vmax      Maximum DC input voltage. [V]
    Vmin      Minimum DC input voltage. [V]
    Vdcmax    . [V]
    MPPTHi    Maximum DC voltage for MPPT range. [V]
    MPPTLow   Minimum DC voltage for MPPT range. [V]
    Pacmax    Maximum AC output power, used to clip the output power
              if needed. [W]
    ADRCoefficients  A list of 9 coefficients that capture the influence
              of input voltage and power on inverter losses, and thereby
              efficiency. Corresponds to terms from [1]_ (in order): :math:
              `b_{0,0}, b_{1,0}, b_{2,0}, b_{0,1}, b_{1,1}, b_{2,1}, b_{0,2},
               b_{1,2},  b_{1,2}`. See [1]_ for the use of each coefficient
               and the associated unit.
    Pnt       AC power consumed by inverter at night (night tare) to
              maintain circuitry required to sense PV array voltage. [W]

    =======   ============================================================

    References
    ----------
    .. [1] Driesse, A. "Beyond the Curves: Modeling the Electrical Efficiency
       of Photovoltaic Inverters", 33rd IEEE Photovoltaic Specialist
       Conference (PVSC), June 2008

    See also
    --------
    pvlib.inverter.sandia
    pvlib.pvsystem.retrieve_sam
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
    power_ac = p_nom * (pdc - p_loss)
    p_nt = -1 * np.absolute(p_nt)

    # set output to nan where input is outside of limits
    # errstate silences case where input is nan
    with np.errstate(invalid='ignore'):
        invalid = (v_lim_upper < v_dc) | (v_dc < v_lim_lower)
    power_ac = np.where(invalid, np.nan, power_ac)

    # set night values
    power_ac = np.where(vdc == 0, p_nt, power_ac)
    power_ac = np.maximum(power_ac, p_nt)

    # set max ac output
    power_ac = np.minimum(power_ac, pac_max)

    if isinstance(p_dc, pd.Series):
        power_ac = pd.Series(power_ac, index=pdc.index)

    return power_ac


def pvwatts(pdc, pdc0, eta_inv_nom=0.96, eta_inv_ref=0.9637):
    r"""
    Implements NREL's PVWatts inverter model [1]_.

    .. math::

        \eta = \frac{\eta_{nom}}{\eta_{ref}} (-0.0162\zeta - \frac{0.0059}
        {\zeta} + 0.9858)

    .. math::

        P_{ac} = \min(\eta P_{dc}, P_{ac0})

    where :math:`\zeta=P_{dc}/P_{dc0}` and :math:`P_{dc0}=P_{ac0}/\eta_{nom}`.

    Note that ``pdc0`` is also used as a symbol in
    :py:func:`pvlib.pvsystem.pvwatts_dc`. ``pdc0`` in this function refers to
    the DC power input limit of the inverter. ``pdc0`` in
    :py:func:`pvlib.pvsystem.pvwatts_dc` refers to the DC power of the modules
    at reference conditions.

    Parameters
    ----------
    pdc: numeric
        DC power. Same unit as ``pdc0``.
    pdc0: numeric
        DC input limit of the inverter.  Same unit as ``pdc``.
    eta_inv_nom: numeric, default 0.96
        Nominal inverter efficiency. [unitless]
    eta_inv_ref: numeric, default 0.9637
        Reference inverter efficiency. PVWatts defines it to be 0.9637
        and is included here for flexibility. [unitless]

    Returns
    -------
    power_ac: numeric
        AC power.  Same unit as ``pdc0``.

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

    # eta < 0 if zeta < 0.006. power_ac is forced to be >= 0 below. GH 541
    eta = eta_inv_nom / eta_inv_ref * (
        -0.0162 * zeta - np.divide(0.0059, zeta, out=eta, where=pdc_neq_0)
        + 0.9858)  # noQA: W503

    power_ac = eta * pdc
    power_ac = np.minimum(pac0, power_ac)
    power_ac = np.maximum(0, power_ac)     # GH 541

    return power_ac
