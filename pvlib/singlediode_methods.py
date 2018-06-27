"""
Faster ways to calculate single diode model currents and voltages using
methods from J.W. Bishop (Solar Cells, 1988).
"""

from collections import OrderedDict
from functools import wraps, partial
import numpy as np
import pandas as pd
try:
    from scipy.optimize import brentq
except ImportError:
    brentq = NotImplemented
# FIXME: change this to newton when scipy-1.2 is released
try:
    from scipy.optimize import _array_newton
except ImportError:
    from pvlib import tools
    from pvlib.tools import _array_newton
# rename newton and set keyword arguments
newton = partial(_array_newton, tol=1e-6, maxiter=100, fprime2=None)

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


def bishop88(diode_voltage, photocurrent, saturation_current, resistance_series,
             resistance_shunt, nNsVth, gradients=False):
    """
    Explicit calculation single-diode-model (SDM) currents and voltages using
    diode junction voltages [1].

    [1] "Computer simulation of the effects of electrical mismatches in
    photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
    https://doi.org/10.1016/0379-6787(88)90059-2

    Parameters
    ----------
    diode_voltage : numeric
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
    a = np.exp(diode_voltage / nNsVth)
    b = 1.0 / resistance_shunt
    i = photocurrent - saturation_current * (a - 1.0) - diode_voltage * b
    v = diode_voltage - i * resistance_series
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


def bishop88_i_from_v(voltage, photocurrent, saturation_current,
                      resistance_series, resistance_shunt, nNsVth,
                      method='newton'):
    """
    Find current given any voltage.

    Parameters
    ----------
    voltage : numeric
        voltage (V) in volts [V]
    photocurrent : numeric
        photogenerated current (Iph or IL) in amperes [A]
    saturation_current : numeric
        diode dark or saturation current (Io or Isat) in amperes [A]
    resistance_series : numeric
        series resistance (Rs) in ohms
    resistance_shunt : numeric
        shunt resistance (Rsh) in ohms
    nNsVth : numeric
        product of diode ideality factor (n), number of series cells (Ns), and
        thermal voltage (Vth = k_b * T / q_e) in volts [V]
    method : str
        one of two ptional search methods: either `brentq`, a reliable and
        bounded method or `newton` the default, a gradient descent method.

    Returns
    -------
    current : numeric
        current (I) at the specified voltage (V) in amperes [A]
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)

    def fv(x, v, *a):
        # calculate voltage
        return bishop88(x, *a)[1] - v

    if method.lower() == 'brentq':
        if brentq is NotImplemented:
            raise ImportError('This function requires scipy')
        # first bound the search using voc
        voc_est = estimate_voc(photocurrent, saturation_current, nNsVth)
        # break out arguments for numpy.vectorize to handle broadcasting
        vec_fun = np.vectorize(
            lambda voc, v, iph, isat, rs, rsh, gamma:
                brentq(fv, 0.0, voc, args=(v, iph, isat, rs, rsh, gamma))
        )
        vd = vec_fun(voc_est, voltage, *args)
    elif method.lower() == 'newton':
        size, shape = _get_size_and_shape((voltage,) + args)
        if shape is not None:
            voltage = np.broadcast_to(voltage, shape).copy()
        vd = newton(func=fv, x0=voltage,
                    fprime=lambda x, *a: bishop88(x, *a, gradients=True)[4],
                    args=args)
    else:
        raise NotImplementedError("Method '%s' isn't implemented" % method)
    return bishop88(vd, *args)[0]


def bishop88_v_from_i(current, photocurrent, saturation_current,
                      resistance_series, resistance_shunt, nNsVth,
                      method='newton'):
    """
    Find voltage given any current.

    Parameters
    ----------
    current : numeric
        current (I) in amperes [A]
    photocurrent : numeric
        photogenerated current (Iph or IL) in amperes [A]
    saturation_current : numeric
        diode dark or saturation current (Io or Isat) in amperes [A]
    resistance_series : numeric
        series resistance (Rs) in ohms
    resistance_shunt : numeric
        shunt resistance (Rsh) in ohms
    nNsVth : numeric
        product of diode ideality factor (n), number of series cells (Ns), and
        thermal voltage (Vth = k_b * T / q_e) in volts [V]
    method : str
        one of two ptional search methods: either `brentq`, a reliable and
        bounded method or `newton` the default, a gradient descent method.

    Returns
    -------
    voltage : numeric
        voltage (V) at the specified current (I) in volts [V]
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = estimate_voc(photocurrent, saturation_current, nNsVth)

    def fi(x, i, *a):
        # calculate current
        return bishop88(x, *a)[0] - i

    if method.lower() == 'brentq':
        if brentq is NotImplemented:
            raise ImportError('This function requires scipy')
        # break out arguments for numpy.vectorize to handle broadcasting
        vec_fun = np.vectorize(
            lambda voc, i, iph, isat, rs, rsh, gamma:
                brentq(fi, 0.0, voc, args=(i, iph, isat, rs, rsh, gamma))
        )
        vd = vec_fun( voc_est, current,*args)
    elif method.lower() == 'newton':
        size, shape = _get_size_and_shape((current,) + args)
        if shape is not None:
            voc_est = np.broadcast_to(voc_est, shape).copy()
        vd = newton(func=fi, x0=voc_est,
                    fprime=lambda x, *a: bishop88(x, *a, gradients=True)[3],
                    args=args)
    else:
        raise NotImplementedError("Method '%s' isn't implemented" % method)
    return bishop88(vd, *args)[1]


def slow_mpp(photocurrent, saturation_current, resistance_series,
             resistance_shunt, nNsVth):
    """
    This is a slow but reliable way to find mpp.
    """
    # recursion
    try:
        len(photocurrent)
    except TypeError:
        pass
    else:
        vecfun = np.vectorize(slow_mpp)
        ivp = vecfun(photocurrent, saturation_current, resistance_series,
                     resistance_shunt, nNsVth)
        if isinstance(photocurrent, pd.Series):
            ivp = {k: v for k, v in zip(('i_mp', 'v_mp', 'p_mp'), ivp)}
            out = pd.DataFrame(ivp, index=photocurrent.index)
        else:
            out = OrderedDict()
            out['i_mp'] = ivp[0]
            out['v_mp'] = ivp[1]
            out['p_mp'] = ivp[2]
        return out
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
    slow_v_from_i = partial(bishop88_v_from_i, method='brentq')
    slow_i_from_v = partial(bishop88_i_from_v, method='brentq')
    # recursion
    try:
        len(photocurrent)
    except TypeError:
        pass
    else:
        vecfun = np.vectorize(slower_way)
        out = vecfun(photocurrent, saturation_current, resistance_series,
                     resistance_shunt, nNsVth, ivcurve_pnts)
        if isinstance(photocurrent, pd.Series) and not ivcurve_pnts:
            out = pd.DataFrame(out.tolist(), index=photocurrent.index)
        else:
            out_array = pd.DataFrame(out.tolist())
            out = OrderedDict()
            out['i_sc'] = out_array.i_sc.values
            out['v_oc'] = out_array.v_oc.values
            out['i_mp'] = out_array.i_mp.values
            out['v_mp'] = out_array.v_mp.values
            out['p_mp'] = out_array.p_mp.values
            out['i_x'] = out_array.i_x.values
            out['i_xx'] = out_array.i_xx.values
            if ivcurve_pnts:
                out['i'] = np.vstack(out_array.i.values)
                out['v'] = np.vstack(out_array.v.values)
                out['p'] = np.vstack(out_array.p.values)
        return out
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
    fast_v_from_i = partial(bishop88_v_from_i, method='newton')
    fast_i_from_v = partial(bishop88_i_from_v, method='newton')
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


def _get_size_and_shape(args):
    # find the right size and shape for returns
    size, shape = 0, None  # 0 or None both mean scalar
    for arg in args:
        try:
            this_shape = arg.shape  # try to get shape
        except AttributeError:
            this_shape = None
            try:
                this_size = len(arg)  # try to get the size
            except TypeError:
                this_size = 0
        else:
            this_size = arg.size  # if it has shape then it also has size
            if shape is None:
                shape = this_shape  # set the shape if None
        # update size and shape
        if this_size > size:
            size = this_size
            if this_shape is not None:
                shape = this_shape
    return size, shape
