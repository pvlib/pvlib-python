import numpy as np
from pvlib.ivtools.Schumaker_QSpline import schumaker_qspline


def estimate_parameters(i, v, nsvth):
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
    to attempt to overcome effects of strong colinearity between ``v`` and ``i``
    over much of the I-V curve.

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
