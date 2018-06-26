"""
Faster ways to calculate single diode model currents and voltages using
methods from J.W. Bishop (Solar Cells, 1988).
"""

from collections import OrderedDict
from functools import wraps
import numpy as np
try:
    from scipy.optimize import brentq, newton
except ImportError:
    brentq = NotImplemented
    newton = NotImplemented

# TODO: update pvsystem.i_from_v and v_from_i to use "gold" method by default


def returns_nan(exc=None):
    """
    Decorator that changes the return to NaN if either
    """
    if not exc:
        exc = (ValueError, RuntimeError)

    def wrapper(f):
        @wraps(f)
        def wrapped_fcn(*args, **kwargs):
            try:
                rval = f(*args, **kwargs)
            except exc:
                rval = np.nan
            return rval
        return wrapped_fcn
    return wrapper


def estimate_voc(photocurrent, saturation_current, nNsVth):
    """
    Rough estimate of open circuit voltage useful for bounding searches for
    ``i`` of ``v`` when using :func:`~pvlib.pvsystem.singlediode`.

    Parameters
    ----------
    photocurrent : numeric
        photo-generated current [A]
    saturation_current : numeric
        diode one reverse saturation current [A]
    nNsVth : numeric
        product of thermal voltage ``Vth`` [V], diode ideality factor ``n``,
        and number of series cells ``Ns``

    Returns
    -------
    numeric
        rough estimate of open circuit voltage [V]

    Calculating the open circuit voltage, :math:`V_{oc}`, of an ideal device
    with infinite shunt resistance, :math:`R_{sh} \\to \\infty`, and zero series
    resistance, :math:`R_s = 0`, yields the following equation [1]. As an
    estimate of :math:`V_{oc}` it is useful as an upper bound for the bisection
    method.

    .. math::

        V_{oc, est}=n Ns V_{th} \\log \\left( \\frac{I_L}{I_0} + 1 \\right)

    [1] http://www.pveducation.org/pvcdrom/open-circuit-voltage
    """

    return nNsVth * np.log(photocurrent / saturation_current + 1.0)


def bishop88(vd, photocurrent, saturation_current, resistance_series,
             resistance_shunt, nNsVth, gradients=False):
    """
    Explicit calculation single-diode-model (SDM) currents and voltages using
    diode junction voltages [1].

    [1] "Computer simulation of the effects of electrical mismatches in
    photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
    https://doi.org/10.1016/0379-6787(88)90059-2

    Parameters
    ----------
    vd : numeric
        diode voltages [V]
    photocurrent : numeric
        photo-generated current [A]
    saturation_current : numeric
        diode one reverse saturation current [A]
    resistance_series : numeric
        series resistance [ohms]
    resistance_shunt: numeric
        shunt resistance [ohms]
    nNsVth : numeric
        product of thermal voltage ``Vth`` [V], diode ideality factor ``n``,
        and number of series cells ``Ns``
    gradients : bool
        default returns only i, v, and p, returns gradients if true

    Returns
    -------
    tuple
        containing currents [A], voltages [V], power [W], gradient ``di/dvd``,
        gradient ``dv/dvd``, gradient ``di/dv``, gradient ``dp/dv``, and
        gradient ``d2p/dv/dvd``
    """
    a = np.exp(vd / nNsVth)
    b = 1.0 / resistance_shunt
    i = photocurrent - saturation_current * (a - 1.0) - vd * b
    v = vd - i * resistance_series
    retval = (i, v, i*v)
    if gradients:
        c = saturation_current * a / nNsVth
        grad_i = - c - b  # di/dvd
        grad_v = 1.0 - grad_i * resistance_series  # dv/dvd
        # dp/dv = d(iv)/dv = v * di/dv + i
        grad = grad_i / grad_v  # di/dv
        grad_p = v * grad + i  # dp/dv
        grad2i = -c / nNsVth  # d2i/dvd
        grad2v = -grad2i * resistance_series  # d2v/dvd
        grad2p = (
            grad_v * grad + v * (grad2i/grad_v - grad_i*grad2v/grad_v**2) + grad_i
        )  # d2p/dv/dvd
        retval += (grad_i, grad_v, grad, grad_p, grad2p)
    return retval


def slow_i_from_v(v, photocurrent, saturation_current, resistance_series,
                  resistance_shunt, nNsVth):
    """
    This is a slow but reliable way to find current given any voltage.
    """
    if brentq is NotImplemented:
        raise ImportError('This function requires scipy')
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = estimate_voc(photocurrent, saturation_current, nNsVth)
    vd = brentq(lambda x, *a: v - bishop88(x, *a)[1], 0.0, voc_est, args)
    return bishop88(vd, *args)[0]


def fast_i_from_v(v, photocurrent, saturation_current, resistance_series,
                  resistance_shunt, nNsVth):
    """
    This is a possibly faster way to find current given any voltage.
    """
    if newton is NotImplemented:
        raise ImportError('This function requires scipy')
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    vd = newton(func=lambda x, *a: bishop88(x, *a)[1] - v, x0=v,
                fprime=lambda x, *a: bishop88(x, *a, gradients=True)[4],
                args=args)
    return bishop88(vd, *args)[0]


def slow_v_from_i(i, photocurrent, saturation_current, resistance_series,
                  resistance_shunt, nNsVth):
    """
    This is a slow but reliable way to find voltage given any current.
    """
    if brentq is NotImplemented:
        raise ImportError('This function requires scipy')
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = estimate_voc(photocurrent, saturation_current, nNsVth)
    vd = brentq(lambda x, *a: i - bishop88(x, *a)[0], 0.0, voc_est, args)
    return bishop88(vd, *args)[1]


def fast_v_from_i(i, photocurrent, saturation_current, resistance_series,
                  resistance_shunt, nNsVth):
    """
    This is a possibly faster way to find voltage given any current.
    """
    if newton is NotImplemented:
        raise ImportError('This function requires scipy')
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = estimate_voc(photocurrent, saturation_current, nNsVth)
    vd = newton(func=lambda x, *a: bishop88(x, *a)[0] - i, x0=voc_est,
                fprime=lambda x, *a: bishop88(x, *a, gradients=True)[3],
                args=args)
    return bishop88(vd, *args)[1]


def slow_mpp(photocurrent, saturation_current, resistance_series,
             resistance_shunt, nNsVth):
    """
    This is a slow but reliable way to find mpp.
    """
    if brentq is NotImplemented:
        raise ImportError('This function requires scipy')
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = estimate_voc(photocurrent, saturation_current, nNsVth)
    vd = brentq(lambda x, *a: bishop88(x, *a, gradients=True)[6], 0.0, voc_est,
                args)
    return bishop88(vd, *args)


def fast_mpp(photocurrent, saturation_current, resistance_series,
             resistance_shunt, nNsVth):
    """
    This is a possibly faster way to find mpp.
    """
    if newton is NotImplemented:
        raise ImportError('This function requires scipy')
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = estimate_voc(photocurrent, saturation_current, nNsVth)
    vd = newton(
        func=lambda x, *a: bishop88(x, *a, gradients=True)[6], x0=voc_est,
        fprime=lambda x, *a: bishop88(x, *a, gradients=True)[7], args=args
    )
    return bishop88(vd, *args)


def slower_way(photocurrent, saturation_current, resistance_series,
               resistance_shunt, nNsVth, ivcurve_pnts=None):
    """
    This is the slow but reliable way.
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    v_oc = slow_v_from_i(0.0, *args)
    i_mp, v_mp, p_mp = slow_mpp(*args)
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
        i, v, p = bishop88(vd, *args)
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
    i_mp, v_mp, p_mp = fast_mpp(*args)
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
        i, v, p = bishop88(vd, *args)
        out['i'] = i
        out['v'] = v
        out['p'] = p
    return out
