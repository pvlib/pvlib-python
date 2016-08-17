import numpy as np


def schumaker_qspline(x, y):
    """
    Schumaker_QSpline fits a quadratic spline which preserves monotonicity and
    convexity in the data.

    Syntax
        outa, outxk, outy, kflag = schumaker_qspline(x, y)

    Description
        Calculates coefficients for C1 quadratic spline interpolating data X, Y
        where length(x) = N and length(y) = N, which preserves monotonicity and
        convexity in the data.

    Parameters
    ----------
    x, y - numpy arrays of length N containing (x, y) points between which the
    spline will interpolate.

    Returns
    -------
    outa - a Nx3 matrix of coefficients where the ith row defines the quadratic
           interpolant between xk_i to xk_(i+1), i.e., y = A[i, 0] *
           (x - xk[i]] ** 2 + A[i, 1] * (x - xk[i]) + A[i, 2]
    outxk - an ordered vector of knots, i.e., values xk_i where the spline
            changes coefficients. All values in x are used as knots. However
            the algorithm may insert additional knots between data points in x
            where changes in convexity are indicated by the (numerical)
            derivative. Consequently output outxk has length >= length(x).
    outy - y values corresponding to the knots in outxk. Contains the original
           data points, y, and also y-values estimated from the spline at the
           inserted knots.
    kflag - a vector of length(outxk) of logicals, which are set to true for
            elements of outxk that are knots inserted by the algorithm.

    Sources
    -------
    [1] PVLib MATLAB
    [2] L. L. Schumaker, "On Shape Preserving Quadratic Spline Interpolation",
        SIAM Journal on Numerical Analysis 20(4), August 1983, pp 854 - 864
    [3] M. H. Lam, "Monotone and Convex Quadratic Spline Interpolation",
        Virginia Journal of Science 41(1), Spring 1990
    """

    # A small number used to decide when a slope is equivalent to zero
    eps = 1e-6

    # Make sure vectors are 1D arrays
    if x.ndim != 1.:
        x = x.flatten([range(x.size)])
    if y.ndim != 1.:
        y = y.flatten([range(y.size)])

    n = len(x)

    # compute various values used by the algorithm: differences, length of line
    # segments between data points, and ratios of differences.
    delx = np.diff(x)  # delx[i] = x[i + 1] - x[i]
    dely = np.diff(y)

    delta = dely / delx

    # Calculate first derivative at each x value per [3]

    s = np.zeros(x.shape)

    left = np.append(0., delta)
    right = np.append(delta, 0.)

    pdelta = left * right

    u = pdelta > 0.

    # [3], Eq. 9 for interior points
    # fix tuning parameters in [2], Eq 9 at chi = .5 and eta = .5
    s[u] = pdelta[u] / (.5 * left[u] + .5 * right[u])

    # [3], Eq. 7 for left endpoint
    if delta[0] * (2. * delta[0] - s[1]) > 0.:
        s[0] = 2. * delta[0] - s[1]

    # [3], Eq. 8 for right endpoint
    if delta[n - 2] * (2. * delta[n - 2] - s[n - 2]) > 0.:
        s[n - 1] = 2. * delta[n - 2] - s[n - 2]

    # determine knots. Start with initial pointsx
    # [2], Algorithm 4.1 first 'if' condition of step 5 defines intervals
    # which won't get internal knots
    tests = s[0.:(n - 1)] + s[1:n]
    u = np.abs(tests - 2. * delta[0:(n - 1)]) <= eps
    # u = true for an interval which will not get an internal knot

    k = n + sum(~u)  # total number of knots = original data + inserted knots

    # set up output arrays
    # knot locations, first n - 1 and very last (n + k) are original data
    xk = np.zeros(k)
    yk = np.zeros(k)  # function values at knot locations
    # logicals that will indicate where additional knots are inserted
    flag = np.zeros(k, dtype=bool)
    a = np.zeros((k, 3.))

    # structures needed to compute coefficients, have to be maintained in
    # association with each knot

    tmpx = x[0:(n - 1)]
    tmpy = y[0:(n - 1)]
    tmpx2 = x[1:n]
    tmps = s[0.:(n - 1)]
    tmps2 = s[1:n]
    diffs = np.diff(s)

    # structure to contain information associated with each knot, used to
    # calculate coefficients
    uu = np.zeros((k, 6.))

    uu[0:(n - 1), :] = np.array([tmpx, tmpx2, tmpy, tmps, tmps2, delta]).T

    # [2], Algorithm 4.1 subpart 1 of Step 5
    # original x values that are left points of intervals without internal
    # knots
    xk[u] = tmpx[u]
    yk[u] = tmpy[u]
    # constant term for each polynomial for intervals without knots
    a[u, 2] = tmpy[u]
    a[u, 1] = s[u]
    a[u, 0] = .5 * diffs[u] / delx[u]  # leading coefficients

    # [2], Algorithm 4.1 subpart 2 of Step 5
    # original x values that are left points of intervals with internal knots
    xk[~u] = tmpx[~u]
    yk[~u] = tmpy[~u]

    aa = s[0:(n - 1)] - delta[0:(n - 1)]
    b = s[1:n] - delta[0:(n - 1)]

    sbar = np.zeros(k)
    eta = np.zeros(k)
    # will contain mapping from the left points of intervals containing an
    # added knot to each inverval's internal knot value
    xi = np.zeros(k)

    t0 = aa * b >= 0
    # first 'else' in Algorithm 4.1 Step 5
    v = np.logical_and(~u, t0[0:len(u)])
    q = np.sum(v)  # number of this type of knot to add

    if q > 0.:
        xk[(n - 1):(n + q - 1)] = .5 * (tmpx[v] + tmpx2[v])  # knot location
        uu[(n - 1):(n + q - 1), :] = np.array([tmpx[v], tmpx2[v], tmpy[v],
                                               tmps[v], tmps2[v], delta[v]]).T
        xi[v] = xk[(n - 1):(n + q - 1)]

    t1 = np.abs(aa) > np.abs(b)
    w = np.logical_and(~u, ~v)  # second 'else' in Algorithm 4.1 Step 5
    w = np.logical_and(w, t1)
    r = np.sum(w)

    if r > 0.:
        xk[(n + q - 1):(n + q + r - 1)] = tmpx2[w] + aa[w] * delx[w] / diffs[w]
        uu[(n + q - 1):(n + q + r - 1), :] = np.array([tmpx[w], tmpx2[w],
                                                       tmpy[w], tmps[w],
                                                       tmps2[w], delta[w]]).T
        xi[w] = xk[(n + q - 1):(n + q + r - 1)]

    z = np.logical_and(~u, ~v)  # last 'else' in Algorithm 4.1 Step 5
    z = np.logical_and(z, ~w)
    ss = np.sum(z)

    if ss > 0.:
        xk[(n + q + r - 1):(n + q + r + ss - 1)] = tmpx[z] + b[z] * delx[z] / \
                                                             diffs[z]
        uu[(n + q + r - 1):(n + q + r + ss - 1), :] = \
            np.array([tmpx[z], tmpx2[z], tmpy[z], tmps[z], tmps2[z],
                      delta[z]]).T
        xi[z] = xk[(n + q + r - 1):(n + q + r + ss - 1)]

    # define polynomial coefficients for intervals with added knots
    ff = ~u
    sbar[ff] = (2 * uu[ff, 5] - uu[ff, 4]) + \
               (uu[ff, 4] - uu[ff, 3]) * (xi[ff] - uu[ff, 0]) / (uu[ff, 1] -
                                                                 uu[ff, 0])
    eta[ff] = (sbar[ff] - uu[ff, 3]) / (xi[ff] - uu[ff, 0])

    sbar[(n - 1):(n + q + r + ss - 1)] = \
        (2 * uu[(n - 1):(n + q + r + ss - 1), 5] -
         uu[(n - 1):(n + q + r + ss - 1), 4]) + \
        (uu[(n - 1):(n + q + r + ss - 1), 4] -
         uu[(n - 1):(n + q + r + ss - 1), 3]) * \
        (xk[(n - 1):(n + q + r + ss - 1)] -
         uu[(n - 1):(n + q + r + ss - 1), 0]) / \
        (uu[(n - 1):(n + q + r + ss - 1), 1] -
         uu[(n - 1):(n + q + r + ss - 1), 0])
    eta[(n - 1):(n + q + r + ss - 1)] = \
        (sbar[(n - 1):(n + q + r + ss - 1)] -
         uu[(n - 1):(n + q + r + ss - 1), 3]) / \
        (xk[(n - 1):(n + q + r + ss - 1)] -
         uu[(n - 1):(n + q + r + ss - 1), 0])

    # constant term for polynomial for intervals with internal knots
    a[~u, 2] = uu[~u, 2]
    a[~u, 1] = uu[~u, 3]
    a[~u, 0] = .5 * eta[~u]  # leading coefficient

    a[(n - 1):(n + q + r + ss - 1), 2] = \
        uu[(n - 1):(n + q + r + ss - 1), 2] + \
        uu[(n - 1):(n + q + r + ss - 1), 3] * \
        (xk[(n - 1):(n + q + r + ss - 1)] -
         uu[(n - 1):(n + q + r + ss - 1), 0]) + \
        .5 * eta[(n - 1):(n + q + r + ss - 1)] * \
        (xk[(n - 1):(n + q + r + ss - 1)] -
         uu[(n - 1):(n + q + r + ss - 1), 0]) ** 2.
    a[(n - 1):(n + q + r + ss - 1), 1] = sbar[(n - 1):(n + q + r + ss - 1)]
    a[(n - 1):(n + q + r + ss - 1), 0] = \
        .5 * (uu[(n - 1):(n + q + r + ss - 1), 4] -
              sbar[(n - 1):(n + q + r + ss - 1)]) / \
        (uu[(n - 1):(n + q + r + ss - 1), 1] -
         uu[(n - 1):(n + q + r + ss - 1), 0])

    yk[(n - 1):(n + q + r + ss - 1)] = a[(n - 1):(n + q + r + ss - 1), 2]

    xk[n + q + r + ss - 1] = x[n - 1]
    yk[n + q + r + ss - 1] = y[n - 1]
    flag[(n - 1):(n + q + r + ss - 1)] = True  # these are all inserted knots

    tmp = np.vstack((xk, a.T, yk, flag)).T
    # sort output in terms of increasing x (original plus added knots)
    tmp2 = tmp[tmp[:, 0].argsort(kind='mergesort')]
    outxk = tmp2[:, 0]
    outn = len(outxk)
    outa = tmp2[0:(outn - 1), 1:4]
    outy = tmp2[:, 4]
    kflag = tmp2[:, 5]
    return outa, outxk, outy, kflag
