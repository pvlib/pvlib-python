"""
Low-level functions for solving the single diode equation.
"""

from functools import partial
import numpy as np
from pvlib.tools import _golden_sect_DataFrame

from scipy.optimize import brentq, newton
from scipy.special import lambertw

# set keyword arguments for all uses of newton in this module
newton = partial(newton, tol=1e-6, maxiter=100, fprime2=None)

# intrinsic voltage per cell junction for a:Si, CdTe, Mertens et al.
VOLTAGE_BUILTIN = 0.9  # [V]


def estimate_voc(photocurrent, saturation_current, nNsVth):
    """
    Rough estimate of open circuit voltage useful for bounding searches for
    ``i`` of ``v`` when using :func:`~pvlib.pvsystem.singlediode`.

    Parameters
    ----------
    photocurrent : numeric
        photo-generated current [A]
    saturation_current : numeric
        diode reverse saturation current [A]
    nNsVth : numeric
        product of thermal voltage ``Vth`` [V], diode ideality factor ``n``,
        and number of series cells ``Ns``

    Returns
    -------
    numeric
        rough estimate of open circuit voltage [V]

    Notes
    -----
    Calculating the open circuit voltage, :math:`V_{oc}`, of an ideal device
    with infinite shunt resistance, :math:`R_{sh} \\to \\infty`, and zero
    series resistance, :math:`R_s = 0`, yields the following equation [1]. As
    an estimate of :math:`V_{oc}` it is useful as an upper bound for the
    bisection method.

    .. math::

        V_{oc, est}=n Ns V_{th} \\log \\left( \\frac{I_L}{I_0} + 1 \\right)

    .. [1] http://www.pveducation.org/pvcdrom/open-circuit-voltage
    """

    return nNsVth * np.log(np.asarray(photocurrent) / saturation_current + 1.0)


def bishop88(diode_voltage, photocurrent, saturation_current,
             resistance_series, resistance_shunt, nNsVth, d2mutau=0,
             NsVbi=np.Inf, breakdown_factor=0., breakdown_voltage=-5.5,
             breakdown_exp=3.28, gradients=False):
    r"""
    Explicit calculation of points on the IV curve described by the single
    diode equation.  Values are calculated as described in [1]_.

    The single diode equation with recombination current and reverse bias
    breakdown is

    .. math::

        I = I_{L} - I_{0} \left (\exp \frac{V_{d}}{nN_{s}V_{th}} - 1 \right )
        - \frac{V_{d}}{R_{sh}}
        - \frac{I_{L} \frac{d^{2}}{\mu \tau}}{N_{s} V_{bi} - V_{d}}
        - a \frac{V_{d}}{R_{sh}} \left (1 - \frac{V_{d}}{V_{br}} \right )^{-m}

    The input `diode_voltage` must be :math:`V + I R_{s}`.


    .. warning::
       * Usage of ``d2mutau`` is required with PVSyst
         coefficients for cadmium-telluride (CdTe) and amorphous-silicon
         (a:Si) PV modules only.
       * Do not use ``d2mutau`` with CEC coefficients.

    Parameters
    ----------
    diode_voltage : numeric
        diode voltage :math:`V_d` [V]
    photocurrent : numeric
        photo-generated current :math:`I_{L}` [A]
    saturation_current : numeric
        diode reverse saturation current :math:`I_{0}` [A]
    resistance_series : numeric
        series resistance :math:`R_{s}` [ohms]
    resistance_shunt: numeric
        shunt resistance :math:`R_{sh}` [ohms]
    nNsVth : numeric
        product of thermal voltage :math:`V_{th}` [V], diode ideality factor
        :math:`n`, and number of series cells :math:`N_{s}` [V]
    d2mutau : numeric, default 0
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that accounts for recombination current in the
        intrinsic layer. The value is the ratio of intrinsic layer thickness
        squared :math:`d^2` to the diffusion length of charge carriers
        :math:`\mu \tau`. [V]
    NsVbi : numeric, default np.inf
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that is the product of the PV module number of series
        cells :math:`N_{s}` and the builtin voltage :math:`V_{bi}` of the
        intrinsic layer. [V].
    breakdown_factor : numeric, default 0
        fraction of ohmic current involved in avalanche breakdown :math:`a`.
        Default of 0 excludes the reverse bias term from the model. [unitless]
    breakdown_voltage : numeric, default -5.5
        reverse breakdown voltage of the photovoltaic junction :math:`V_{br}`
        [V]
    breakdown_exp : numeric, default 3.28
        avalanche breakdown exponent :math:`m` [unitless]
    gradients : bool
        False returns only I, V, and P. True also returns gradients

    Returns
    -------
    tuple
        currents [A], voltages [V], power [W], and optionally
        :math:`\frac{dI}{dV_d}`, :math:`\frac{dV}{dV_d}`,
        :math:`\frac{dI}{dV}`, :math:`\frac{dP}{dV}`, and
        :math:`\frac{d^2 P}{dV dV_d}`

    Notes
    -----
    The PVSyst thin-film recombination losses parameters ``d2mutau`` and
    ``NsVbi`` should only be applied to cadmium-telluride (CdTe) and amorphous-
    silicon (a-Si) PV modules, [2]_, [3]_. The builtin voltage :math:`V_{bi}`
    should account for all junctions. For example: tandem and triple junction
    cells would have builtin voltages of 1.8[V] and 2.7[V] respectively, based
    on the default of 0.9[V] for a single junction. The parameter ``NsVbi``
    should only account for the number of series cells in a single parallel
    sub-string if the module has cells in parallel greater than 1.

    References
    ----------
    .. [1] "Computer simulation of the effects of electrical mismatches in
       photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
       :doi:`10.1016/0379-6787(88)90059-2`

    .. [2] "Improved equivalent circuit and Analytical Model for Amorphous
       Silicon Solar Cells and Modules." J. Mertens, et al., IEEE Transactions
       on Electron Devices, Vol 45, No 2, Feb 1998.
       :doi:`10.1109/16.658676`

    .. [3] "Performance assessment of a simulation model for PV modules of any
       available technology", André Mermoud and Thibault Lejeune, 25th EUPVSEC,
       2010
       :doi:`10.4229/25thEUPVSEC2010-4BV.1.114`
    """
    # calculate recombination loss current where d2mutau > 0
    is_recomb = d2mutau > 0  # True where there is thin-film recombination loss
    v_recomb = np.where(is_recomb, NsVbi - diode_voltage, np.inf)
    i_recomb = np.where(is_recomb, photocurrent * d2mutau / v_recomb, 0)
    # calculate temporary values to simplify calculations
    v_star = diode_voltage / nNsVth  # non-dimensional diode voltage
    g_sh = 1.0 / resistance_shunt  # conductance
    if breakdown_factor > 0:  # reverse bias is considered
        brk_term = 1 - diode_voltage / breakdown_voltage
        brk_pwr = np.power(brk_term, -breakdown_exp)
        i_breakdown = breakdown_factor * diode_voltage * g_sh * brk_pwr
    else:
        i_breakdown = 0.
    i = (photocurrent - saturation_current * np.expm1(v_star)  # noqa: W503
         - diode_voltage * g_sh - i_recomb - i_breakdown)   # noqa: W503
    v = diode_voltage - i * resistance_series
    retval = (i, v, i*v)
    if gradients:
        # calculate recombination loss current gradients where d2mutau > 0
        grad_i_recomb = np.where(is_recomb, i_recomb / v_recomb, 0)
        grad_2i_recomb = np.where(is_recomb, 2 * grad_i_recomb / v_recomb, 0)
        g_diode = saturation_current * np.exp(v_star) / nNsVth  # conductance
        if breakdown_factor > 0:  # reverse bias is considered
            brk_pwr_1 = np.power(brk_term, -breakdown_exp - 1)
            brk_pwr_2 = np.power(brk_term, -breakdown_exp - 2)
            brk_fctr = breakdown_factor * g_sh
            grad_i_brk = brk_fctr * (brk_pwr + diode_voltage *
                                     -breakdown_exp * brk_pwr_1)
            grad2i_brk = (brk_fctr * -breakdown_exp        # noqa: W503
                          * (2 * brk_pwr_1 + diode_voltage   # noqa: W503
                             * (-breakdown_exp - 1) * brk_pwr_2))  # noqa: W503
        else:
            grad_i_brk = 0.
            grad2i_brk = 0.
        grad_i = -g_diode - g_sh - grad_i_recomb - grad_i_brk  # di/dvd
        grad_v = 1.0 - grad_i * resistance_series  # dv/dvd
        # dp/dv = d(iv)/dv = v * di/dv + i
        grad = grad_i / grad_v  # di/dv
        grad_p = v * grad + i  # dp/dv
        grad2i = -g_diode / nNsVth - grad_2i_recomb - grad2i_brk  # d2i/dvd
        grad2v = -grad2i * resistance_series  # d2v/dvd
        grad2p = (
            grad_v * grad + v * (grad2i/grad_v - grad_i*grad2v/grad_v**2)
            + grad_i
        )  # d2p/dv/dvd
        retval += (grad_i, grad_v, grad, grad_p, grad2p)
    return retval


def bishop88_i_from_v(voltage, photocurrent, saturation_current,
                      resistance_series, resistance_shunt, nNsVth,
                      d2mutau=0, NsVbi=np.Inf, breakdown_factor=0.,
                      breakdown_voltage=-5.5, breakdown_exp=3.28,
                      method='newton'):
    """
    Find current given any voltage.

    Parameters
    ----------
    voltage : numeric
        voltage (V) in volts [V]
    photocurrent : numeric
        photogenerated current (Iph or IL) [A]
    saturation_current : numeric
        diode dark or saturation current (Io or Isat) [A]
    resistance_series : numeric
        series resistance (Rs) in [Ohm]
    resistance_shunt : numeric
        shunt resistance (Rsh) [Ohm]
    nNsVth : numeric
        product of diode ideality factor (n), number of series cells (Ns), and
        thermal voltage (Vth = k_b * T / q_e) in volts [V]
    d2mutau : numeric, default 0
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that accounts for recombination current in the
        intrinsic layer. The value is the ratio of intrinsic layer thickness
        squared :math:`d^2` to the diffusion length of charge carriers
        :math:`\\mu \\tau`. [V]
    NsVbi : numeric, default np.inf
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that is the product of the PV module number of series
        cells ``Ns`` and the builtin voltage ``Vbi`` of the intrinsic layer.
        [V].
    breakdown_factor : numeric, default 0
        fraction of ohmic current involved in avalanche breakdown :math:`a`.
        Default of 0 excludes the reverse bias term from the model. [unitless]
    breakdown_voltage : numeric, default -5.5
        reverse breakdown voltage of the photovoltaic junction :math:`V_{br}`
        [V]
    breakdown_exp : numeric, default 3.28
        avalanche breakdown exponent :math:`m` [unitless]
    method : str, default 'newton'
       Either ``'newton'`` or ``'brentq'``. ''method'' must be ``'newton'``
       if ``breakdown_factor`` is not 0.

    Returns
    -------
    current : numeric
        current (I) at the specified voltage (V). [A]
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth, d2mutau, NsVbi,
            breakdown_factor, breakdown_voltage, breakdown_exp)

    def fv(x, v, *a):
        # calculate voltage residual given diode voltage "x"
        return bishop88(x, *a)[1] - v

    if method.lower() == 'brentq':
        # first bound the search using voc
        voc_est = estimate_voc(photocurrent, saturation_current, nNsVth)

        # brentq only works with scalar inputs, so we need a set up function
        # and np.vectorize to repeatedly call the optimizer with the right
        # arguments for possible array input
        def vd_from_brent(voc, v, iph, isat, rs, rsh, gamma, d2mutau, NsVbi,
                          breakdown_factor, breakdown_voltage, breakdown_exp):
            return brentq(fv, 0.0, voc,
                          args=(v, iph, isat, rs, rsh, gamma, d2mutau, NsVbi,
                                breakdown_factor, breakdown_voltage,
                                breakdown_exp))

        vd_from_brent_vectorized = np.vectorize(vd_from_brent)
        vd = vd_from_brent_vectorized(voc_est, voltage, *args)
    elif method.lower() == 'newton':
        # make sure all args are numpy arrays if max size > 1
        # if voltage is an array, then make a copy to use for initial guess, v0
        args, v0 = _prepare_newton_inputs((voltage,), args, voltage)
        vd = newton(func=lambda x, *a: fv(x, voltage, *a), x0=v0,
                    fprime=lambda x, *a: bishop88(x, *a, gradients=True)[4],
                    args=args)
    else:
        raise NotImplementedError("Method '%s' isn't implemented" % method)
    return bishop88(vd, *args)[0]


def bishop88_v_from_i(current, photocurrent, saturation_current,
                      resistance_series, resistance_shunt, nNsVth,
                      d2mutau=0, NsVbi=np.Inf, breakdown_factor=0.,
                      breakdown_voltage=-5.5, breakdown_exp=3.28,
                      method='newton'):
    """
    Find voltage given any current.

    Parameters
    ----------
    current : numeric
        current (I) in amperes [A]
    photocurrent : numeric
        photogenerated current (Iph or IL) [A]
    saturation_current : numeric
        diode dark or saturation current (Io or Isat) [A]
    resistance_series : numeric
        series resistance (Rs) in [Ohm]
    resistance_shunt : numeric
        shunt resistance (Rsh) [Ohm]
    nNsVth : numeric
        product of diode ideality factor (n), number of series cells (Ns), and
        thermal voltage (Vth = k_b * T / q_e) in volts [V]
    d2mutau : numeric, default 0
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that accounts for recombination current in the
        intrinsic layer. The value is the ratio of intrinsic layer thickness
        squared :math:`d^2` to the diffusion length of charge carriers
        :math:`\\mu \\tau`. [V]
    NsVbi : numeric, default np.inf
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that is the product of the PV module number of series
        cells ``Ns`` and the builtin voltage ``Vbi`` of the intrinsic layer.
        [V].
    breakdown_factor : numeric, default 0
        fraction of ohmic current involved in avalanche breakdown :math:`a`.
        Default of 0 excludes the reverse bias term from the model. [unitless]
    breakdown_voltage : numeric, default -5.5
        reverse breakdown voltage of the photovoltaic junction :math:`V_{br}`
        [V]
    breakdown_exp : numeric, default 3.28
        avalanche breakdown exponent :math:`m` [unitless]
    method : str, default 'newton'
       Either ``'newton'`` or ``'brentq'``. ''method'' must be ``'newton'``
       if ``breakdown_factor`` is not 0.

    Returns
    -------
    voltage : numeric
        voltage (V) at the specified current (I) in volts [V]
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth, d2mutau, NsVbi, breakdown_factor,
            breakdown_voltage, breakdown_exp)
    # first bound the search using voc
    voc_est = estimate_voc(photocurrent, saturation_current, nNsVth)

    def fi(x, i, *a):
        # calculate current residual given diode voltage "x"
        return bishop88(x, *a)[0] - i

    if method.lower() == 'brentq':
        # brentq only works with scalar inputs, so we need a set up function
        # and np.vectorize to repeatedly call the optimizer with the right
        # arguments for possible array input
        def vd_from_brent(voc, i, iph, isat, rs, rsh, gamma, d2mutau, NsVbi,
                          breakdown_factor, breakdown_voltage, breakdown_exp):
            return brentq(fi, 0.0, voc,
                          args=(i, iph, isat, rs, rsh, gamma, d2mutau, NsVbi,
                                breakdown_factor, breakdown_voltage,
                                breakdown_exp))

        vd_from_brent_vectorized = np.vectorize(vd_from_brent)
        vd = vd_from_brent_vectorized(voc_est, current, *args)
    elif method.lower() == 'newton':
        # make sure all args are numpy arrays if max size > 1
        # if voc_est is an array, then make a copy to use for initial guess, v0
        args, v0 = _prepare_newton_inputs((current,), args, voc_est)
        vd = newton(func=lambda x, *a: fi(x, current, *a), x0=v0,
                    fprime=lambda x, *a: bishop88(x, *a, gradients=True)[3],
                    args=args)
    else:
        raise NotImplementedError("Method '%s' isn't implemented" % method)
    return bishop88(vd, *args)[1]


def bishop88_mpp(photocurrent, saturation_current, resistance_series,
                 resistance_shunt, nNsVth, d2mutau=0, NsVbi=np.Inf,
                 breakdown_factor=0., breakdown_voltage=-5.5,
                 breakdown_exp=3.28, method='newton'):
    """
    Find max power point.

    Parameters
    ----------
    photocurrent : numeric
        photogenerated current (Iph or IL) [A]
    saturation_current : numeric
        diode dark or saturation current (Io or Isat) [A]
    resistance_series : numeric
        series resistance (Rs) in [Ohm]
    resistance_shunt : numeric
        shunt resistance (Rsh) [Ohm]
    nNsVth : numeric
        product of diode ideality factor (n), number of series cells (Ns), and
        thermal voltage (Vth = k_b * T / q_e) in volts [V]
    d2mutau : numeric, default 0
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that accounts for recombination current in the
        intrinsic layer. The value is the ratio of intrinsic layer thickness
        squared :math:`d^2` to the diffusion length of charge carriers
        :math:`\\mu \\tau`. [V]
    NsVbi : numeric, default np.inf
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that is the product of the PV module number of series
        cells ``Ns`` and the builtin voltage ``Vbi`` of the intrinsic layer.
        [V].
    breakdown_factor : numeric, default 0
        fraction of ohmic current involved in avalanche breakdown :math:`a`.
        Default of 0 excludes the reverse bias term from the model. [unitless]
    breakdown_voltage : numeric, default -5.5
        reverse breakdown voltage of the photovoltaic junction :math:`V_{br}`
        [V]
    breakdown_exp : numeric, default 3.28
        avalanche breakdown exponent :math:`m` [unitless]
    method : str, default 'newton'
       Either ``'newton'`` or ``'brentq'``. ''method'' must be ``'newton'``
       if ``breakdown_factor`` is not 0.

    Returns
    -------
    OrderedDict or pandas.DataFrame
        max power current ``i_mp`` [A], max power voltage ``v_mp`` [V], and
        max power ``p_mp`` [W]
    """
    # collect args
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth, d2mutau, NsVbi, breakdown_factor,
            breakdown_voltage, breakdown_exp)
    # first bound the search using voc
    voc_est = estimate_voc(photocurrent, saturation_current, nNsVth)

    def fmpp(x, *a):
        return bishop88(x, *a, gradients=True)[6]

    if method.lower() == 'brentq':
        # break out arguments for numpy.vectorize to handle broadcasting
        vec_fun = np.vectorize(
            lambda voc, iph, isat, rs, rsh, gamma, d2mutau, NsVbi, vbr_a, vbr,
            vbr_exp: brentq(fmpp, 0.0, voc,
                            args=(iph, isat, rs, rsh, gamma, d2mutau, NsVbi,
                                  vbr_a, vbr, vbr_exp))
        )
        vd = vec_fun(voc_est, *args)
    elif method.lower() == 'newton':
        # make sure all args are numpy arrays if max size > 1
        # if voc_est is an array, then make a copy to use for initial guess, v0
        args, v0 = _prepare_newton_inputs((), args, voc_est)
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


def _prepare_newton_inputs(i_or_v_tup, args, v0):
    # broadcast arguments for newton method
    # the first argument should be a tuple, eg: (i,), (v,) or ()
    size, shape = _get_size_and_shape(i_or_v_tup + args)
    if size > 1:
        args = [np.asarray(arg) for arg in args]
    # newton uses initial guess for the output shape
    # copy v0 to a new array and broadcast it to the shape of max size
    if shape is not None:
        v0 = np.broadcast_to(v0, shape).copy()
    return args, v0


def _lambertw_v_from_i(resistance_shunt, resistance_series, nNsVth, current,
                       saturation_current, photocurrent):
    # Record if inputs were all scalar
    output_is_scalar = all(map(np.isscalar,
                               [resistance_shunt, resistance_series, nNsVth,
                                current, saturation_current, photocurrent]))

    # This transforms Gsh=1/Rsh, including ideal Rsh=np.inf into Gsh=0., which
    #  is generally more numerically stable
    conductance_shunt = 1. / resistance_shunt

    # Ensure that we are working with read-only views of numpy arrays
    # Turns Series into arrays so that we don't have to worry about
    #  multidimensional broadcasting failing
    Gsh, Rs, a, I, I0, IL = \
        np.broadcast_arrays(conductance_shunt, resistance_series, nNsVth,
                            current, saturation_current, photocurrent)

    # Intitalize output V (I might not be float64)
    V = np.full_like(I, np.nan, dtype=np.float64)

    # Determine indices where 0 < Gsh requires implicit model solution
    idx_p = 0. < Gsh

    # Determine indices where 0 = Gsh allows explicit model solution
    idx_z = 0. == Gsh

    # Explicit solutions where Gsh=0
    if np.any(idx_z):
        V[idx_z] = a[idx_z] * np.log1p((IL[idx_z] - I[idx_z]) / I0[idx_z]) - \
                   I[idx_z] * Rs[idx_z]

    # Only compute using LambertW if there are cases with Gsh>0
    if np.any(idx_p):
        # LambertW argument, cannot be float128, may overflow to np.inf
        # overflow is explicitly handled below, so ignore warnings here
        with np.errstate(over='ignore'):
            argW = (I0[idx_p] / (Gsh[idx_p] * a[idx_p]) *
                    np.exp((-I[idx_p] + IL[idx_p] + I0[idx_p]) /
                           (Gsh[idx_p] * a[idx_p])))

        # lambertw typically returns complex value with zero imaginary part
        # may overflow to np.inf
        lambertwterm = lambertw(argW).real

        # Record indices where lambertw input overflowed output
        idx_inf = np.logical_not(np.isfinite(lambertwterm))

        # Only re-compute LambertW if it overflowed
        if np.any(idx_inf):
            # Calculate using log(argW) in case argW is really big
            logargW = (np.log(I0[idx_p]) - np.log(Gsh[idx_p]) -
                       np.log(a[idx_p]) +
                       (-I[idx_p] + IL[idx_p] + I0[idx_p]) /
                       (Gsh[idx_p] * a[idx_p]))[idx_inf]

            # Three iterations of Newton-Raphson method to solve
            #  w+log(w)=logargW. The initial guess is w=logargW. Where direct
            #  evaluation (above) results in NaN from overflow, 3 iterations
            #  of Newton's method gives approximately 8 digits of precision.
            w = logargW
            for _ in range(0, 3):
                w = w * (1. - np.log(w) + logargW) / (1. + w)
            lambertwterm[idx_inf] = w

        # Eqn. 3 in Jain and Kapoor, 2004
        #  V = -I*(Rs + Rsh) + IL*Rsh - a*lambertwterm + I0*Rsh
        # Recast in terms of Gsh=1/Rsh for better numerical stability.
        V[idx_p] = (IL[idx_p] + I0[idx_p] - I[idx_p]) / Gsh[idx_p] - \
            I[idx_p] * Rs[idx_p] - a[idx_p] * lambertwterm

    if output_is_scalar:
        return V.item()
    else:
        return V


def _lambertw_i_from_v(resistance_shunt, resistance_series, nNsVth, voltage,
                       saturation_current, photocurrent):
    # Record if inputs were all scalar
    output_is_scalar = all(map(np.isscalar,
                               [resistance_shunt, resistance_series, nNsVth,
                                voltage, saturation_current, photocurrent]))

    # This transforms Gsh=1/Rsh, including ideal Rsh=np.inf into Gsh=0., which
    #  is generally more numerically stable
    conductance_shunt = 1. / resistance_shunt

    # Ensure that we are working with read-only views of numpy arrays
    # Turns Series into arrays so that we don't have to worry about
    #  multidimensional broadcasting failing
    Gsh, Rs, a, V, I0, IL = \
        np.broadcast_arrays(conductance_shunt, resistance_series, nNsVth,
                            voltage, saturation_current, photocurrent)

    # Intitalize output I (V might not be float64)
    I = np.full_like(V, np.nan, dtype=np.float64)           # noqa: E741, N806

    # Determine indices where 0 < Rs requires implicit model solution
    idx_p = 0. < Rs

    # Determine indices where 0 = Rs allows explicit model solution
    idx_z = 0. == Rs

    # Explicit solutions where Rs=0
    if np.any(idx_z):
        I[idx_z] = IL[idx_z] - I0[idx_z] * np.expm1(V[idx_z] / a[idx_z]) - \
                   Gsh[idx_z] * V[idx_z]

    # Only compute using LambertW if there are cases with Rs>0
    # Does NOT handle possibility of overflow, github issue 298
    if np.any(idx_p):
        # LambertW argument, cannot be float128, may overflow to np.inf
        argW = Rs[idx_p] * I0[idx_p] / (
                    a[idx_p] * (Rs[idx_p] * Gsh[idx_p] + 1.)) * \
               np.exp((Rs[idx_p] * (IL[idx_p] + I0[idx_p]) + V[idx_p]) /
                      (a[idx_p] * (Rs[idx_p] * Gsh[idx_p] + 1.)))

        # lambertw typically returns complex value with zero imaginary part
        # may overflow to np.inf
        lambertwterm = lambertw(argW).real

        # Eqn. 2 in Jain and Kapoor, 2004
        #  I = -V/(Rs + Rsh) - (a/Rs)*lambertwterm + Rsh*(IL + I0)/(Rs + Rsh)
        # Recast in terms of Gsh=1/Rsh for better numerical stability.
        I[idx_p] = (IL[idx_p] + I0[idx_p] - V[idx_p] * Gsh[idx_p]) / \
                   (Rs[idx_p] * Gsh[idx_p] + 1.) - (
                               a[idx_p] / Rs[idx_p]) * lambertwterm

    if output_is_scalar:
        return I.item()
    else:
        return I


def _lambertw(photocurrent, saturation_current, resistance_series,
              resistance_shunt, nNsVth, ivcurve_pnts=None):
    # Compute short circuit current
    i_sc = _lambertw_i_from_v(resistance_shunt, resistance_series, nNsVth, 0.,
                              saturation_current, photocurrent)

    # Compute open circuit voltage
    v_oc = _lambertw_v_from_i(resistance_shunt, resistance_series, nNsVth, 0.,
                              saturation_current, photocurrent)

    params = {'r_sh': resistance_shunt,
              'r_s': resistance_series,
              'nNsVth': nNsVth,
              'i_0': saturation_current,
              'i_l': photocurrent}

    # Find the voltage, v_mp, where the power is maximized.
    # Start the golden section search at v_oc * 1.14
    p_mp, v_mp = _golden_sect_DataFrame(params, 0., v_oc * 1.14,
                                        _pwr_optfcn)

    # Find Imp using Lambert W
    i_mp = _lambertw_i_from_v(resistance_shunt, resistance_series, nNsVth,
                              v_mp, saturation_current, photocurrent)

    # Find Ix and Ixx using Lambert W
    i_x = _lambertw_i_from_v(resistance_shunt, resistance_series, nNsVth,
                             0.5 * v_oc, saturation_current, photocurrent)

    i_xx = _lambertw_i_from_v(resistance_shunt, resistance_series, nNsVth,
                              0.5 * (v_oc + v_mp), saturation_current,
                              photocurrent)

    out = (i_sc, v_oc, i_mp, v_mp, p_mp, i_x, i_xx)

    # create ivcurve
    if ivcurve_pnts:
        ivcurve_v = (np.asarray(v_oc)[..., np.newaxis] *
                     np.linspace(0, 1, ivcurve_pnts))

        ivcurve_i = _lambertw_i_from_v(resistance_shunt, resistance_series,
                                       nNsVth, ivcurve_v.T, saturation_current,
                                       photocurrent).T

        out += (ivcurve_i, ivcurve_v)

    return out


def _pwr_optfcn(df, loc):
    '''
    Function to find power from ``i_from_v``.
    '''

    I = _lambertw_i_from_v(df['r_sh'], df['r_s'],           # noqa: E741, N806
                           df['nNsVth'], df[loc], df['i_0'], df['i_l'])

    return I * df[loc]
