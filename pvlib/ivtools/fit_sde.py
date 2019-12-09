"""
The ``fit_sde`` module contains functions to fit the single diode equation.

Function names should follow the pattern "fit_sde_" + fitting method.

"""

import numpy as np

from utility import schumaker_qspline


def fit_sde_sandia(voltage, current, v_oc=None, i_sc=None, v_mp_i_mp=None,
                   vlim=0.2, ilim=0.1):
    r"""
    Fits the single diode equation (SDE) to an IV curve.

    Parameters
    ----------
    voltage : ndarray
        1D array of `float` type containing voltage at each point on the IV
        curve, increasing from 0 to ``v_oc`` inclusive [V]

    current : ndarray
        1D array of `float` type containing current at each point on the IV
        curve, from ``i_sc`` to 0 inclusive [A]

    v_oc : float, default None
        Open circuit voltage [V]. If not provided, ``v_oc`` is taken as the
        last point in the ``voltage`` array.

    i_sc : float, default None
        Short circuit current [A]. If not provided, ``i_sc`` is taken as the
        first point in the ``current`` array.

    v_mp_i_mp : tuple of float, default None
        Voltage, current at maximum power point in units of [V], [A].
        If not provided, the maximum power point is found at the maximum of
        ``voltage`` \times ``current``.

    vlim : float, default 0.2
        Defines portion of IV curve where the exponential term in the single
        diode equation can be neglected, i.e.
        ``voltage`` <= ``vlim`` x ``v_oc`` [V]

    ilim : float, default 0.1
        Defines portion of the IV curve where the exponential term in the
        single diode equation is signficant, approximately defined by
        ``current`` < (1 - ``ilim``) x ``i_sc`` [A]

    Returns
    -------
    tuple of the following elements:

        * photocurrent : float
            photocurrent [A]
        * saturation_current : float
            dark (saturation) current [A]
        * resistance_shunt : float
            shunt (parallel) resistance, in ohms
        * resistance_series : float
            series resistance, in ohms
        * nNsVth : float
            product of thermal voltage ``Vth`` [V], diode ideality factor
            ``n``, and number of series cells ``Ns``

    Raises
    ------
    RuntimeError if parameter extraction is not successful.

    Notes
    -----
    Inputs ``voltage``, ``current``, ``v_oc``, ``i_sc`` and ``v_mp_i_mp`` are
    assumed to be from a single IV curve at constant irradiance and cell
    temperature.

    :py:func:`fit_single_diode_sandia` obtains values for the five parameters
    for the single diode equation [1]:

    .. math::

        I = I_{L} - I_{0} (\exp \frac{V + I R_{s}}{nNsVth} - 1)
        - \frac{V + I R_{s}}{R_{sh}}

    See :py:func:`pvsystem.singlediode` for definition of the parameters.

    The extraction method [2] proceeds in six steps.

    1. In the single diode equation, replace :math:`R_{sh} = 1/G_{p}` and
       re-arrange

    .. math::

        I = \frac{I_{L}}{1 + G_{p} R_{s}} - \frac{G_{p} V}{1 + G_{p} R_{s}}
        - \frac{I_{0}}{1 + G_{p} R_{s}} (\exp(\frac{V + I R_{s}}{nNsVth}) - 1)

    2. The linear portion of the IV curve is defined as
       :math:`V \le vlim \times v_oc`. Over this portion of the IV curve,

    .. math::

        \frac{I_{0}}{1 + G_{p} R_{s}} (\exp(\frac{V + I R_{s}}{nNsVth}) - 1)
        \approx 0

    3. Fit the linear portion of the IV curve with a line.

    .. math::

        I &\approx \frac{I_{L}}{1 + G_{p} R_{s}} - \frac{G_{p} V}{1 + G_{p}
        R_{s}} \\
        &= \beta_{0} + \beta_{1} V

    4. The exponential portion of the IV curve is defined by
       :math:`\beta_{0} + \beta_{1} \times V - I > ilim \times i_sc`.
       Over this portion of the curve, :math:`exp((V + IRs)/nNsVth) >> 1`
       so that

    .. math::

        \exp(\frac{V + I R_{s}}{nNsVth}) - 1 \approx
        \exp(\frac{V + I R_{s}}{nNsVth})

    5. Fit the exponential portion of the IV curve.

    .. math::

        \log(\beta_{0} - \beta_{1} V - I)
        &\approx \log(\frac{I_{0}}{1 + G_{p} R_{s}} + \frac{V}{nNsVth}
        + \frac{I R_{s}}{nNsVth} \\
        &= \beta_{2} + beta_{3} V + \beta_{4} I

    6. Calculate values for ``IL, I0, Rs, Rsh,`` and ``nNsVth`` from the
       regression coefficents :math:`\beta_{0}, \beta_{1}, \beta_{3}` and
       :math:`\beta_{4}`.


    References
    ----------
    [1] S.R. Wenham, M.A. Green, M.E. Watt, "Applied Photovoltaics" ISBN
    0 86758 909 4
    [2] C. B. Jones, C. W. Hansen, Single Diode Parameter Extraction from
    In-Field Photovoltaic I-V Curves on a Single Board Computer, 46th IEEE
    Photovoltaic Specialist Conference, Chicago, IL, 2019
    """

    # If not provided, extract v_oc, i_sc, v_mp and i_mp from the IV curve data
    if v_oc is None:
        v_oc = voltage[-1]
    if i_sc is None:
        i_sc = current[0]
    if v_mp_i_mp is not None:
        v_mp, i_mp = v_mp_i_mp
    else:
        v_mp, i_mp = _find_mp(voltage, current)

    # Find beta0 and beta1 from linear portion of the IV curve
    beta0, beta1 = _find_beta0_beta1(voltage, current, vlim, v_oc)

    # Find beta3 and beta4 from the exponential portion of the IV curve
    beta3, beta4 = _find_beta3_beta4(voltage, current, beta0, beta1, ilim,
                                     i_sc)

    # calculate single diode parameters from regression coefficients
    return _calculate_sde_parameters(beta0, beta1, beta3, beta4, v_mp, i_mp,
                                     v_oc)


def _find_mp(voltage, current):
    """
    Finds voltage and current at maximum power point.

    Parameters
    ----------
    voltage : ndarray
        1D array containing voltage at each point on the IV curve, increasing
        from 0 to v_oc inclusive, of `float` type [V]

    current : ndarray
        1D array containing current at each point on the IV curve, decreasing
        from i_sc to 0 inclusive, of `float` type [A]

    Returns
    -------
    v_mp, i_mp : tuple
        voltage ``v_mp`` and current ``i_mp`` at the maximum power point [V],
        [A]
    """
    p = voltage * current
    idx = np.argmax(p)
    return voltage[idx], current[idx]


def _find_beta0_beta1(v, i, vlim, v_oc):
    # Get intercept and slope of linear portion of IV curve.
    # Start with V =< vlim * v_oc, extend by adding points until slope is
    # negative (downward).
    beta0 = np.nan
    beta1 = np.nan
    first_idx = np.searchsorted(v, vlim * v_oc)
    for idx in range(first_idx, len(v)):
        coef = np.polyfit(v[:idx], i[:idx], deg=1)
        if coef[0] < 0:
            # intercept term
            beta0 = coef[1].item()
            # sign change of slope to get positive parameter value
            beta1 = -coef[0].item()
            break
    if any(np.isnan([beta0, beta1])):
        raise RuntimeError("Parameter extraction failed: beta0={}, beta1={}"
                           .format(beta0, beta1))
    else:
        return beta0, beta1


def _find_beta3_beta4(voltage, current, beta0, beta1, ilim, i_sc):
    # Subtract the IV curve from the linear fit.
    y = beta0 - beta1 * voltage - current
    x = np.array([np.ones_like(voltage), voltage, current]).T
    # Select points where y > ilim * i_sc to regress log(y) onto x
    idx = (y > ilim * i_sc)
    result = np.linalg.lstsq(x[idx], np.log(y[idx]), rcond=None)
    coef = result[0]
    beta3 = coef[1].item()
    beta4 = coef[2].item()
    if any(np.isnan([beta3, beta4])):
        raise RuntimeError("Parameter extraction failed: beta3={}, beta4={}"
                           .format(beta3, beta4))
    else:
        return beta3, beta4


def _calculate_sde_parameters(beta0, beta1, beta3, beta4, v_mp, i_mp, v_oc):
    nNsVth = 1.0 / beta3
    Rs = beta4 / beta3
    Gp = beta1 / (1.0 - Rs * beta1)
    Rsh = 1.0 / Gp
    IL = (1 + Gp * Rs) * beta0
    # calculate I0
    I0_vmp = _calc_I0(IL, i_mp, v_mp, Gp, Rs, nNsVth)
    I0_voc = _calc_I0(IL, 0, v_oc, Gp, Rs, nNsVth)
    if any(np.isnan([I0_vmp, I0_voc])) or ((I0_vmp <= 0) and (I0_voc <= 0)):
        raise RuntimeError("Parameter extraction failed: I0 is undetermined.")
    elif (I0_vmp > 0) and (I0_voc > 0):
        I0 = 0.5 * (I0_vmp + I0_voc)
    elif (I0_vmp > 0):
        I0 = I0_vmp
    else:  # I0_voc > 0
        I0 = I0_voc
    return (IL, I0, Rsh, Rs, nNsVth)


def _calc_I0(IL, I, V, Gp, Rs, nNsVth):
    return (IL - I - Gp * V - Gp * Rs * I) / np.exp((V + Rs * I) / nNsVth)


def fit_sde_cocontent(i, v, nsvth):
    """
    Regression technique to fit the single diode equation to data for a single
    IV curve.

    Parameters
    ----------
    i : numeric
        current for the IV curve, the first value is taken as ``Isc``, the last
        value must be 0
    v : numeric
        voltage for the IV curve corresponding to the current values in the
        input vector ``i``, the first value must be 0, the last value is taken
        as ``Voc``
    nsvth : numeric
        the thermal voltage for the module, equal to ``Ns`` (number of cells in
        series) times ``Vth`` (thermal voltage per cell)

    Returns
    -------
    io : numeric
        the dark current value (A) for the IV curve
    iph : numeric
        the light current value (A) for the IV curve
    rs : numeric
        series resistance (ohm) for the IV curve
    rsh : numieric
        shunt resistance (ohm) for the IV curve
    n : numeric
        diode (ideality) factor (unitless) for the IV curve

    This function uses a regression technique based on [2] to fit the single
    diode equation to data for a single IV curve. Although values for each of
    the five parameters are returned, testing has shown only ``Rsh`` to be
    stable. The other parameters, ``Rs``, ``Io`` and ``n`` may be negative or
    imaginary even for IV curve data without obvious flaws. Method coded here
    uses a principal component transformation of ``(V, I)`` prior to regression
    to attempt to overcome effects of strong colinearity between ``v`` and
    ``i`` over much of the I-V curve.

    References
    ----------
    [1] PVLib MATLAB
    [2] A. Ortiz-Conde, F. Garci'a Sa'nchez, J. Murci, "New method to extract
        the model parameters of solar cells from the explicit analytic
        solutions of their illuminated I-V characteristics", Solar Energy
        Materials and Solar Cells 90, pp 352 - 361, 2006.
    [3] C. Hansen, Parameter Estimation for Single Diode Models of
        Photovoltaic Modules, Sandia National Laboratories Report SAND2015-2065
    """

    if len(i) != len(v):
        raise ValueError("Current and Voltage vectors should have the same "
                         "length")
    isc = i[0]  # short circuit current
    voc = v[-1]  # open circuite voltage

    # Fit quadratic spline to IV curve in order to compute the co-content
    # (i.e., integral of Isc - I over V) more accurately

    [a, xk, xi, kflag] = schumaker_qspline(v, i)

    # calculate co-content integral by numerical integration of quadratic
    # spline for (Isc - I) over V
    xn = len(xk)
    xk2 = xk[1:xn]
    xk1 = xk[0:(xn - 1)]
    delx = xk2 - xk1
    tmp = np.array([1. / 3., .5, 1.])
    ss = np.tile(tmp, [xn - 1, 1])
    cc = a * ss
    tmpint = np.sum(cc * np.array([delx ** 3, delx ** 2, delx]).T, 1)
    tmpint = np.append(0., tmpint)

    scc = np.zeros(xn)

    # Use trapezoid rule for the first 5 intervals due to spline being
    # unreliable near the left endpoint
    scc[0:5] = isc * xk[0:5] - np.cumsum(tmpint[0:5])  # by spline
    scc[5:(xn - 5)] = isc * (xk[5:(xn - 5)] - xk[4]) - \
        np.cumsum(tmpint[5:(xn - 5)]) + scc[4]

    # Use trapezoid rule for the last 5 intervals due to spline being
    # unreliable near the right endpoint
    scc[(xn - 5):xn] = isc * (xk[(xn - 5):xn] - xk[xn - 6]) - \
        np.cumsum(tmpint[(xn - 5):xn]) + scc[xn - 6]
    # by spline

    # For estimating diode equation parameters only use original dataa points,
    # not at any knots added by the quadratic spline fit
    # co-content integral, i.e., Int_0^Voc (Isc - I) dV
    cci = scc[~kflag.astype(bool)]

    # predictor variables for regression of CC
    x = np.vstack((v, isc - i, v * (isc - i), v * v, (i - isc) ** 2)).T

    # define principal components transformation to shift, scale and rotate
    # V and I before the regression.
    tmpx = x[:, 0:2]
    tmpx_length = tmpx.shape[0]

    tmpx_mean = np.mean(tmpx, axis=0)
    tmpx_std = np.std(tmpx, axis=0, ddof=1)
    tmpx_zscore = (tmpx - np.tile(tmpx_mean, [tmpx_length, 1])) / \
        np.tile(tmpx_std, [tmpx_length, 1])

    tmpx_d, tmpx_v = np.linalg.eig(np.cov(tmpx_zscore.T))

    idx = np.argsort(tmpx_d)[::-1]

    ev1 = tmpx_v[:, idx[0]]

    # Second component set to be orthogonal and rotated counterclockwise by 90.
    ev2 = np.dot(np.array([[0., -1.], [1., 0.]]), ev1)
    r = np.array([ev1, ev2])  # principal components transformation

    s = np.dot(tmpx_zscore, r)
    # [V, I] shift and scaled by zscore, rotated by r

    scc = cci - np.mean(cci, axis=0)  # center co-content values
    col1 = np.ones(len(scc))

    # predictors. Shifting makes a constant term necessary in the regression
    # model
    sx = np.vstack((s[:, 0], s[:, 1], s[:, 0] * s[:, 1], s[:, 0] * s[:, 0],
                    s[:, 1] * s[:, 1], col1)).T

    gamma = np.linalg.lstsq(sx, scc)[0]
    # coefficients from regression in rotated coordinates

    # Matrix which relates principal components transformation R to the mapping
    # between [V' I' V'I' V'^2 I'^2] and sx, where prime ' indicates shifted
    # and scaled data. Used to translate from regression coefficients in
    # rotated coordinates to coefficients in initial V, I coordinates.
    mb = np.array([[r[0, 0], r[1, 0], 0., 0., 0.], [r[0, 1], r[1, 1], 0., 0.,
                                                    0.],
                   [0., 0., r[0, 0] * r[1, 1] + r[0, 1] * r[1, 0], 2. *
                    r[0, 0] * r[0, 1], 2. * r[1, 0] * r[1, 1]],
                   [0., 0., r[0, 0] * r[1, 0], r[0, 0] ** 2., r[1, 0] ** 2.],
                   [0., 0., r[0, 1] * r[1, 1], r[0, 1] ** 2., r[1, 1] ** 2.]])

    # matrix which is used to undo effect of shifting and scaling on regression
    # coefficients.
    ma = np.array([[np.std(v, ddof=1), 0., np.std(v, ddof=1) *
                    np.mean(isc - i), 2. * np.std(v, ddof=1) * np.mean(v),
                   0.], [0., np.std(isc - i, ddof=1), np.std(isc - i, ddof=1)
                         * np.mean(v), 0.,
                   2. * np.std(isc - i, ddof=1) * np.mean(isc - i)],
                   [0., 0., np.std(v, ddof=1) * np.std(isc - i, ddof=1), 0.,
                    0.],
                   [0., 0., 0., np.std(v, ddof=1) ** 2., 0.],
                   [0., 0., 0., 0., np.std(isc - i, ddof=1) ** 2.]])

    # translate from coefficients in rotated space (gamma) to coefficients in
    # original coordinates (beta)
    beta = np.linalg.lstsq(np.dot(mb, ma), gamma[0:5])[0]

    # Extract five parameter values from coefficients in original coordinates.
    # Equation 11, [2]
    betagp = beta[3] * 2.

    # Equation 12, [2]
    betars = (np.sqrt(1. + 16. * beta[3] * beta[4]) - 1.) / (4. * beta[3])

    # Equation 13, [2]
    betan = (beta[0] * (np.sqrt(1. + 16. * beta[3] * beta[4]) - 1.) + 4. *
             beta[1] * beta[3]) / (4. * beta[3] * nsvth)

    # Single diode equation at Voc, approximating Iph + Io by Isc
    betaio = (isc - voc * betagp) / (np.exp(voc / (betan * nsvth)))

    # Single diode equation at Isc, using Rsh, Rs, n and Io that were
    # determined above
    betaiph = isc - betaio + betaio * np.exp(isc / (betan * nsvth)) + \
        isc * betars * betagp

    iph = betaiph
    rs = betars
    rsh = 1 / betagp
    n = betan
    io = betaio
    return io, iph, rs, rsh, n