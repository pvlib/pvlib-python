"""
Faster ways to calculate single diode model currents and voltages using
methods from J.W. Bishop (Solar Cells, 1988).
"""

from functools import partial
import numpy as np
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

    return nNsVth * np.log(np.asarray(photocurrent) / saturation_current + 1.0)


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
            grad_v * grad + v * (grad2i/grad_v - grad_i*grad2v/grad_v**2)
            + grad_i
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
        # calculate voltage residual given diode voltage "x"
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
        # make sure all args are numpy arrays if max size > 1
        size, shape = _get_size_and_shape((voltage,) + args)
        if size > 1:
            args = [np.asarray(arg) for arg in args]
        # newton uses initial guess for the output shape
        # copy v0 to a new array and broadcast it to the shape of max size
        if shape is not None:
            v0 = np.broadcast_to(voltage, shape).copy()
        else:
            v0 = voltage
        # x0 and x in func are the same reference! DO NOT set x0 to voltage!
        # voltage in fv(x, voltage, *a) MUST BE CONSTANT!
        vd = newton(func=lambda x, *a: fv(x, voltage, *a), x0=v0,
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
        # calculate current residual given diode voltage "x"
        return bishop88(x, *a)[0] - i

    if method.lower() == 'brentq':
        if brentq is NotImplemented:
            raise ImportError('This function requires scipy')
        # break out arguments for numpy.vectorize to handle broadcasting
        vec_fun = np.vectorize(
            lambda voc, i, iph, isat, rs, rsh, gamma:
                brentq(fi, 0.0, voc, args=(i, iph, isat, rs, rsh, gamma))
        )
        vd = vec_fun(voc_est, current, *args)
    elif method.lower() == 'newton':
        # make sure all args are numpy arrays if max size > 1
        size, shape = _get_size_and_shape((current,) + args)
        if size > 1:
            args = [np.asarray(arg) for arg in args]
        # newton uses initial guess for the output shape
        # copy v0 to a new array and broadcast it to the shape of max size
        if shape is not None:
            v0 = np.broadcast_to(voc_est, shape).copy()
        else:
            v0 = voc_est
        # x0 and x in func are the same reference! DO NOT set x0 to current!
        # voltage in fi(x, current, *a) MUST BE CONSTANT!
        vd = newton(func=lambda x, *a: fi(x, current, *a), x0=v0,
                    fprime=lambda x, *a: bishop88(x, *a, gradients=True)[3],
                    args=args)
    else:
        raise NotImplementedError("Method '%s' isn't implemented" % method)
    return bishop88(vd, *args)[1]


def bishop88_mpp(photocurrent, saturation_current, resistance_series,
                 resistance_shunt, nNsVth, method='newton'):
    """
    Find max power point.

    Parameters
    ----------
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
    OrderedDict or pandas.DataFrame
        max power current ``i_mp`` [A], max power voltage ``v_mp`` [V], and
        max power ``p_mp`` [W]
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)
    # first bound the search using voc
    voc_est = estimate_voc(photocurrent, saturation_current, nNsVth)

    def fmpp(x, *a):
        return bishop88(x, *a, gradients=True)[6]

    if method.lower() == 'brentq':
        if brentq is NotImplemented:
            raise ImportError('This function requires scipy')
        # break out arguments for numpy.vectorize to handle broadcasting
        vec_fun = np.vectorize(
            lambda voc, iph, isat, rs, rsh, gamma:
                brentq(fmpp, 0.0, voc, args=(iph, isat, rs, rsh, gamma))
        )
        vd = vec_fun(voc_est, *args)
    elif method.lower() == 'newton':
        # make sure all args are numpy arrays if max size > 1
        size, shape = _get_size_and_shape(args)
        if size > 1:
            args = [np.asarray(arg) for arg in args]
        # newton uses initial guess for the output shape
        # copy v0 to a new array and broadcast it to the shape of max size
        if shape is not None:
            v0 = np.broadcast_to(voc_est, shape).copy()
        else:
            v0 = voc_est
        # x0 and x in func are the same reference! DO NOT set x0 to current!
        # voltage in fi(x, current, *a) MUST BE CONSTANT!
        vd = newton(
            func=fmpp, x0=v0,
            fprime=lambda x, *a: bishop88(x, *a, gradients=True)[7], args=args
        )
    else:
        raise NotImplementedError("Method '%s' isn't implemented" % method)
    return bishop88(vd, *args)


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
