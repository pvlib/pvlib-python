"""
Faster ways to calculate single diode model currents and voltages using
methods from J.W. Bishop (Solar Cells, 1988).
"""

import logging
from collections import OrderedDict
import numpy as np
from scipy.optimize import fminbound, newton

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

EPS = np.finfo(float).eps
DAMP = 1.5
DELTA = EPS**0.33


def est_voc(photocurrent, saturation_current, nNsVth):
    """
    Rough estimate of open circuit voltage useful for bounding searches for
    ``i`` of ``v`` when using :func:`~pvlib.way_faster`.

    :param photocurrent: photo-generated current [A]
    :param saturation_current: diode one reverse saturation current [A]
    :param nNsVth: product of thermal voltage ``Vth`` [V], diode ideality
        factor ``n``, and number of series cells ``Ns``
    :returns: rough estimate of open circuit voltage [V]
    """
    # http://www.pveducation.org/pvcdrom/open-circuit-voltage
    return nNsVth * np.log(photocurrent / saturation_current + 1.0)


def bishop88(vd, photocurrent, saturation_current, resistance_series,
             resistance_shunt, nNsVth):
    """
    Explicit calculation single-diode-model (SDM) currents and voltages using
    diode junction voltages [1].

    [1] "Computer simulation of the effects of electrical mismatches in
        photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
        https://doi.org/10.1016/0379-6787(88)90059-2

    :param vd: diode voltages [V}]
    :param photocurrent: photo-generated current [A]
    :param saturation_current: diode one reverse saturation current [A]
    :param resistance_series: series resitance [ohms]
    :param resistance_shunt: shunt resitance [ohms]
    :param nNsVth" product of thermal voltage ``Vth`` [V], diode ideality
        factor ``n``, and number of series cells ``Ns``
    :returns: tuple containing currents [A], voltages [V], gradient ``di/dvd``,
        gradient ``dv/dvd``, power [W], gradient ``dp/dv``, and gradient
        ``d2p/dv/dvd``
    """
    a = np.exp(vd / nNsVth)
    b = 1.0 / resistance_shunt
    i = photocurrent - saturation_current * (a - 1.0) - vd * b
    v = vd - i * resistance_series
    c = saturation_current * a / nNsVth
    grad_i = - c - b  # di/dvd
    grad_v = 1.0 - grad_i * resistance_series  # dv/dvd
    # dp/dv = d(iv)/dv = v * di/dv + i
    grad = grad_i / grad_v  # di/dv
    grad_p = v * grad + i  # dp/dv
    grad2i = -c / nNsVth
    grad2v = -grad2i * resistance_series
    grad2p = (
        grad_v * grad + v * (grad2i/grad_v - grad_i*grad2v/grad_v**2) + grad_i
    )
    return i, v, grad_i, grad_v, i*v, grad_p, grad2p


def slow_i_from_v(v, photocurrent, saturation_current, resistance_series,
                  resistance_shunt, nNsVth):
    """
    This is a slow but reliable way to find current given any voltage.
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = est_voc(photocurrent, saturation_current, nNsVth)
    vd = fminbound(lambda x: (v - bishop88(x, *args)[1])**2, 0.0, voc_est)
    return bishop88(vd, *args)[0]


def slow_v_from_i(i, photocurrent, saturation_current, resistance_series,
                  resistance_shunt, nNsVth):
    """
    This is a slow but reliable way to find voltage given any current.
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = est_voc(photocurrent, saturation_current, nNsVth)
    vd = fminbound(lambda x: (i - bishop88(x, *args)[0])**2, 0.0, voc_est)
    return bishop88(vd, *args)[1]


def slow_mppt(photocurrent, saturation_current, resistance_series,
              resistance_shunt, nNsVth):
    """
    This is a slow but reliable way to find mpp.
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = est_voc(photocurrent, saturation_current, nNsVth)
    vd = fminbound(lambda x: -(bishop88(x, *args)[4])**2, 0.0, voc_est)
    i, v, _, _, p, _, _ = bishop88(vd, *args)
    return i, v, p


def slower_way(photocurrent, saturation_current, resistance_series,
               resistance_shunt, nNsVth, ivcurve_pnts=None):
    """
    This is the slow but reliable way.
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    voc = slow_v_from_i(0.0, *args)
    i_sc = slow_i_from_v(0.0, *args)
    i_mp, v_mp, p_mp = slow_mppt(*args)
    out = OrderedDict()
    out['i_sc'] = i_sc
    out['v_oc'] = voc
    out['i_mp'] = i_mp
    out['v_mp'] = v_mp
    out['p_mp'] = p_mp
    out['i_x'] = None
    out['i_xx'] = None
    # calculate the IV curve if requested using bishop88
    if ivcurve_pnts:
        vd = voc * (
            (11.0 - np.logspace(np.log10(11.0), 0.0, ivcurve_pnts)) / 10.0
        )
        i, v, _, _, p, _, _ = bishop88(vd, *args)
        out['i'] = i
        out['v'] = v
        out['p'] = p
    return out


def faster_way(photocurrent, saturation_current, resistance_series,
               resistance_shunt, nNsVth, ivcurve_pnts=None,
               tol=EPS, damp=DAMP, log=True, test=True):
    """a faster way"""
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)  # collect args
    # first estimate Voc
    voc_est = est_voc(photocurrent, saturation_current, nNsVth)
    # find the real voc
    resnorm = np.inf  # nor of residuals
    while resnorm > tol:
        i_test, v_test, grad, _, _, _, _ = bishop88(voc_est, *args)
        newton_step = i_test / grad
        # don't let step get crazy
        if np.abs(newton_step / voc_est) > damp:
            break
        voc_est -= newton_step
        resnorm = i_test**2
        if log:
            LOGGER.debug(
                'voc_est=%g, step=%g, i_test=%g, v_test=%g, resnorm=%g',
                voc_est, newton_step, i_test, v_test, resnorm
            )
        if test:
            delta = EPS**0.3
            i_test2, _, _, _, _, _, _ = bishop88(voc_est * (1.0 + delta), *args)
            LOGGER.debug('test_grad=%g', (i_test2 - i_test) / voc_est / delta)
            LOGGER.debug('grad=%g', grad)
    # find isc too
    isc_est = 0.0
    vd_sc = 0.0
    resnorm = np.inf  # nor of residuals
    while resnorm > tol:
        isc_est, v_test, _, grad, _, _, _ = bishop88(vd_sc, *args)
        newton_step = v_test / grad
        # don't let step get crazy
        if np.abs(newton_step / voc_est) > damp:
            break
        vd_sc -= newton_step
        resnorm = v_test**2
        if log:
            LOGGER.debug(
                'vd_sc=%g, step=%g, isc_est=%g, v_test=%g, resnorm=%g',
                vd_sc, newton_step, isc_est, v_test, resnorm
            )
        if test:
            delta = EPS**0.3
            _, v_test2, _, _, _, _, _ = bishop88(vd_sc * (1.0 + delta), *args)
            LOGGER.debug('test_grad=%g', (v_test2 - v_test) / vd_sc / delta)
            LOGGER.debug('grad=%g', grad)
    # find the mpp
    imp_est, vmp_est, pmp_est = 0.0, 0.0, 0.0
    vd_mp = voc_est
    resnorm = np.inf  # nor of residuals
    while resnorm > tol:
        imp_est, vmp_est, _, _, pmp_est, grad_p, grad2p = bishop88(vd_mp, *args)
        newton_step = grad_p / grad2p
        # don't let step get crazy
        if np.abs(newton_step / voc_est) > damp:
            break
        vd_mp -= newton_step
        resnorm = grad_p**2
        if log:
            LOGGER.debug(
                'vd_mp=%g, step=%g, pmp_est=%g, resnorm=%g',
                vd_mp, newton_step, pmp_est, resnorm
            )
        if test:
            delta = EPS**0.3
            _, _, _, _, _, grad_p2, _ = bishop88(vd_mp * (1.0 + delta), *args)
            LOGGER.debug('test_grad=%g', (grad_p2 - grad_p) / vd_mp / delta)
            LOGGER.debug('grad=%g', grad2p)
    out = OrderedDict()
    out['i_sc'] = isc_est
    out['v_oc'] = voc_est
    out['i_mp'] = imp_est
    out['v_mp'] = vmp_est
    out['p_mp'] = pmp_est
    out['i_x'] = None
    out['i_xx'] = None
    # calculate the IV curve if requested using bishop88
    if ivcurve_pnts:
        vd = voc_est * (
            (11.0 - np.logspace(np.log10(11.0), 0.0, ivcurve_pnts)) / 10.0
        )
        i, v, _, _, p, _, _ = bishop88(vd, *args)
        out['i'] = i
        out['v'] = v
        out['p'] = p
    return out
