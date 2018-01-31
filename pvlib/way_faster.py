"""
Faster ways to calculate single diode model currents and voltages using
methods from J.W. Bishop (Solar Cells, 1988).
"""

from collections import OrderedDict
import numpy as np
from scipy.optimize import fminbound, newton


# TODO: make fast_i_from_v, fast_v_from_i, fast_mppt using newton
# TODO: remove grad calcs from bishop88
# TODO: add new residual and f_prime calcs for fast_ methods to use newton
# TODO: refactor singlediode to be a wrapper with a method argument
# TODO: update pvsystem.singlediode to use slow_ methods by default
# TODO: ditto for i_from_v and v_from_i
# TODO: add new mppt function to pvsystem


def est_voc(photocurrent, saturation_current, nNsVth):
    """
    Rough estimate of open circuit voltage useful for bounding searches for
    ``i`` of ``v`` when using :func:`~pvlib.way_faster`.

    :param numeric photocurrent: photo-generated current [A]
    :param numeric saturation_current: diode one reverse saturation current [A]
    :param numeric nNsVth: product of thermal voltage ``Vth`` [V], diode
        ideality factor ``n``, and number of series cells ``Ns``
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

    :param numeric vd: diode voltages [V]
    :param numeric photocurrent: photo-generated current [A]
    :param numeric saturation_current: diode one reverse saturation current [A]
    :param numeric resistance_series: series resitance [ohms]
    :param numeric resistance_shunt: shunt resitance [ohms]
    :param numeric nNsVth: product of thermal voltage ``Vth`` [V], diode
        ideality factor ``n``, and number of series cells ``Ns``
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


def fast_i_from_v(v, photocurrent, saturation_current, resistance_series,
                  resistance_shunt, nNsVth):
    """
    This is a fast but unreliable way to find current given any voltage.
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    vd = newton(func=lambda x, *a: bishop88(x, *a)[1] - v, x0=v,
                fprime=lambda x, *a: bishop88(x, *a)[3], args=args)
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


def fast_v_from_i(i, photocurrent, saturation_current, resistance_series,
                  resistance_shunt, nNsVth):
    """
    This is a fast but unreliable way to find voltage given any current.
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = est_voc(photocurrent, saturation_current, nNsVth)
    vd = newton(func=lambda x, *a: bishop88(x, *a)[0] - i, x0=voc_est,
                fprime=lambda x, *a: bishop88(x, *a)[2], args=args)
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


def fast_mppt(photocurrent, saturation_current, resistance_series,
              resistance_shunt, nNsVth):
    """
    This is a fast but unreliable way to find mpp.
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = est_voc(photocurrent, saturation_current, nNsVth)
    vd = newton(func=lambda x, *a: bishop88(x, *a)[5], x0=voc_est,
                fprime=lambda x, *a: bishop88(x, *a)[6], args=args)
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
    v_oc = slow_v_from_i(0.0, *args)
    i_mp, v_mp, p_mp = slow_mppt(*args)
    out = OrderedDict()
    out['i_sc'] = slow_i_from_v(0.0, *args)
    out['v_oc'] = v_oc
    out['i_mp'] = i_mp
    out['v_mp'] = v_mp
    out['p_mp'] = p_mp
    out['i_x'] = slow_i_from_v(v_oc / 2.0, *args)
    out['i_xx'] = slow_i_from_v((v_oc + v_mp) / 2.0, *args)
    # calculate the IV curve if requested using bishop88
    if ivcurve_pnts:
        vd = v_oc * (
            (11.0 - np.logspace(np.log10(11.0), 0.0, ivcurve_pnts)) / 10.0
        )
        i, v, _, _, p, _, _ = bishop88(vd, *args)
        out['i'] = i
        out['v'] = v
        out['p'] = p
    return out


def faster_way(photocurrent, saturation_current, resistance_series,
               resistance_shunt, nNsVth, ivcurve_pnts=None):
    """a faster way"""
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)  # collect args
    v_oc = fast_v_from_i(0.0, *args)
    i_mp, v_mp, p_mp = fast_mppt(*args)
    out = OrderedDict()
    out['i_sc'] = fast_i_from_v(0.0, *args)
    out['v_oc'] = v_oc
    out['i_mp'] = i_mp
    out['v_mp'] = v_mp
    out['p_mp'] = p_mp
    out['i_x'] = fast_i_from_v(v_oc / 2.0, *args)
    out['i_xx'] = fast_i_from_v((v_oc + v_mp) / 2.0, *args)
    # calculate the IV curve if requested using bishop88
    if ivcurve_pnts:
        vd = v_oc * (
            (11.0 - np.logspace(np.log10(11.0), 0.0, ivcurve_pnts)) / 10.0
        )
        i, v, _, _, p, _, _ = bishop88(vd, *args)
        out['i'] = i
        out['v'] = v
        out['p'] = p
    return out
