"""
Faster ways to calculate single diode model currents and voltages using
methods from J.W. Bishop (Solar Cells, 1988).
"""

import logging
from collections import OrderedDict
import numpy as np
from scipy.optimize import fminbound

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

EPS = np.finfo(float).eps
DAMP = 1.5
DELTA = EPS**0.33


def est_voc(il, io, nnsvt):
    """
    Rough estimate of open circuit voltage useful for bounding searches for
    ``i`` of ``v`` when using :func:`~pvlib.way_faster`.

    :param il: photo-generated current [A]
    :param io: diode one reverse saturation or "dark" current [A]
    :param nnsvt" product of thermal voltage ``Vt`` [V], ideality factor ``n``,
        and number of series cells ``Ns``
    :returns: rough estimate of open circuit voltage [V]
    """
    # http://www.pveducation.org/pvcdrom/open-circuit-voltage
    return nnsvt * np.log(il / io + 1.0)


def bishop88(vd, il, io, rs, rsh, nnsvt):
    """
    Explicit calculation single-diode-model (SDM) currents and voltages using
    diode junction voltages [1].

    [1] "Computer simulation of the effects of electrical mismatches in
        photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
        https://doi.org/10.1016/0379-6787(88)90059-2

    :param vd: diode voltages [V}]
    :param il: photo-generated current [A]
    :param io: diode one reverse saturation or "dark" current [A]
    :param rs: series resitance [ohms]
    :param rsh: shunt resitance [ohms]
    :param nnsvt" product of thermal voltage ``Vt`` [V], ideality factor ``n``,
        and number of series cells ``Ns``
    :returns: tuple containing currents [A], voltages [V], gradient ``di/dvd``,
        gradient ``dv/dvd``, power [W], gradient ``dp/dv``, and gradient
        ``d2p/dv/dvd``
    """
    a = np.exp(vd / nnsvt)
    b = 1.0 / rsh
    i = il - io * (a - 1.0) - vd * b
    v = vd - i * rs
    c = io * a / nnsvt
    grad_i = - c - b  # di/dvd
    grad_v = 1.0 - grad_i * rs  # dv/dvd
    # dp/dv = d(iv)/dv = v * di/dv + i
    grad = grad_i / grad_v  # di/dv
    grad_p = v * grad + i  # dp/dv
    grad2i = -c / nnsvt
    grad2v = -grad2i * rs
    grad2p = (
        grad_v * grad + v * (grad2i/grad_v - grad_i*grad2v/grad_v**2) + grad_i
    )
    return i, v, grad_i, grad_v, i*v, grad_p, grad2p


def newton_solver(fjx, x0, x, tol=EPS, damp=DAMP, log=False, test=True):
    resnorm = np.inf  # nor of residuals
    while resnorm > tol:
        f, j = fjx(x0, *x)
        newton_step = f / j
        # don't let step get crazy
        if np.abs(newton_step / x0) > damp:
            break
        x0 -= newton_step
        resnorm = f**2
        if log:
            LOGGER.debug(
                'x0=%g, newton step=%g, f=%g, resnorm=%g',
                x0, newton_step, f, resnorm
            )
        if test:
            f2, _ = fjx(x0 * (1.0 + DELTA), *x)
            LOGGER.debug('test_grad=%g', (f2 - f) / x0 / DELTA)
            LOGGER.debug('grad=%g', j)
    return x0, f, j


def slow_i_from_v(v, photocurrent, saturation_current, resistance_series,
                  resistance_shunt, nNsVth):
    """
    This is a slow but reliable way to find current given any voltage.
    """
    # FIXME: everything is named the wrong thing!
    il = photocurrent
    io = saturation_current
    rs = resistance_series
    rsh = resistance_shunt
    nnsvt = nNsVth
    x = (il, io, rs, rsh, nnsvt)  # collect args
    # first bound the search using voc
    voc_est = est_voc(il, io, nnsvt)
    vd = fminbound(lambda vd: (v - bishop88(vd, *x)[1])**2, 0.0, voc_est)
    return bishop88(vd, il, io, rs, rsh, nnsvt)[0]


def slow_v_from_i(i, photocurrent, saturation_current, resistance_series,
                  resistance_shunt, nNsVth):
    """
    This is a slow but reliable way to find voltage given any current.
    """
    # FIXME: everything is named the wrong thing!
    il = photocurrent
    io = saturation_current
    rs = resistance_series
    rsh = resistance_shunt
    nnsvt = nNsVth
    x = (il, io, rs, rsh, nnsvt)  # collect args
    # first bound the search using voc
    voc_est = est_voc(il, io, nnsvt)
    vd = fminbound(lambda vd: (i - bishop88(vd, *x)[0])**2, 0.0, voc_est)
    return bishop88(vd, il, io, rs, rsh, nnsvt)[1]

def slow_mppt(photocurrent, saturation_current, resistance_series,
              resistance_shunt, nNsVth):
    """
    This is a slow but reliable way to find mpp.
    """
    # FIXME: everything is named the wrong thing!
    il = photocurrent
    io = saturation_current
    rs = resistance_series
    rsh = resistance_shunt
    nnsvt = nNsVth
    x = (il, io, rs, rsh, nnsvt)  # collect args
    # first bound the search using voc
    voc_est = est_voc(il, io, nnsvt)
    vd = fminbound(lambda vd: -(bishop88(vd, *x)[4])**2, 0.0, voc_est)
    i, v, _, _, p, _, _ = bishop88(vd, il, io, rs, rsh, nnsvt)
    return i, v, p


def slower_way(photocurrent, saturation_current, resistance_series,
               resistance_shunt, nNsVth, ivcurve_pnts=None):
    """
    This is the slow but reliable way.
    """
    # FIXME: everything is named the wrong thing!
    il = photocurrent
    io = saturation_current
    rs = resistance_series
    rsh = resistance_shunt
    nnsvt = nNsVth
    x = (il, io, rs, rsh, nnsvt)  # collect args
    voc = slow_v_from_i(0, *x)
    i_sc = slow_i_from_v(0, *x)
    i_mp, v_mp, p_mp = slow_mppt(*x)
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
        i, v, _, _, p, _, _ = bishop88(vd, *x)
        out['i'] = i
        out['v'] = v
        out['p'] = p
    return out


def faster_way(photocurrent, saturation_current, resistance_series,
               resistance_shunt, nNsVth, ivcurve_pnts=None,
               tol=EPS, damp=DAMP, log=True, test=True):
    """a faster way"""
    # FIXME: everything is named the wrong thing!
    il = photocurrent
    io = saturation_current
    rs = resistance_series
    rsh = resistance_shunt
    nnsvt = nNsVth
    x = (il, io, rs, rsh, nnsvt)  # collect args
    # first estimate Voc
    voc_est = est_voc(il, io, nnsvt)
    # find the real voc
    resnorm = np.inf  # nor of residuals
    while resnorm > tol:
        i_test, v_test, grad, _, _, _, _ = bishop88(voc_est, *x)
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
            i_test2, _, _, _, _, _, _ = bishop88(voc_est * (1.0 + delta), *x)
            LOGGER.debug('test_grad=%g', (i_test2 - i_test) / voc_est / delta)
            LOGGER.debug('grad=%g', grad)
    # find isc too
    vd_sc = 0.0
    resnorm = np.inf  # nor of residuals
    while resnorm > tol:
        isc_est, v_test, _, grad, _, _, _ = bishop88(vd_sc, *x)
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
            _, v_test2, _, _, _, _, _ = bishop88(vd_sc * (1.0 + delta), *x)
            LOGGER.debug('test_grad=%g', (v_test2 - v_test) / vd_sc / delta)
            LOGGER.debug('grad=%g', grad)
    # find the mpp
    vd_mp = voc_est
    resnorm = np.inf  # nor of residuals
    while resnorm > tol:
        imp_est, vmp_est, _, _, pmp_est, grad_p, grad2p = bishop88(vd_mp, *x)
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
            _, _, _, _, _, grad_p2, _ = bishop88(vd_mp * (1.0 + delta), *x)
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
        i, v, _, _, p, _, _ = bishop88(vd, *x)
        out['i'] = i
        out['v'] = v
        out['p'] = p
    return out
