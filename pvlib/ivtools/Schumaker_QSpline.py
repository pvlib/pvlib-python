import numpy as np

# A small number used to decide when a slope is equivalent to zero
EPS = np.finfo('float').eps**(1/3)


def schumaker_qspline(x, y):
    """
    Schumaker_QSpline fits a quadratic spline which preserves monotonicity and
    convexity in the data.

    Parameters
    ----------
    x : numeric
        independent points between which the spline will interpolate.
    y : numeric
        dependent points between which the spline will interpolate.

    Returns
    -------
    outa : numpy.ndarray
        a Nx3 matrix of coefficients where the ith row defines the quadratic
        interpolant between xk_i to xk_(i+1), i.e., y = A[i, 0] *
        (x - xk[i]] ** 2 + A[i, 1] * (x - xk[i]) + A[i, 2]
    outxk : numpy.ndarray
        an ordered vector of knots, i.e., values xk_i where the spline
        changes coefficients. All values in x are used as knots. However
        the algorithm may insert additional knots between data points in x
        where changes in convexity are indicated by the (numerical)
        derivative. Consequently output outxk has length >= length(x).
    outy : numpy.ndarray
        y values corresponding to the knots in outxk. Contains the original
        data points, y, and also y-values estimated from the spline at the
        inserted knots.
    kflag : numpy.ndarray
        a vector of length(outxk) of logicals, which are set to true for
        elements of outxk that are knots inserted by the algorithm.

    Description
    -----------
    Calculates coefficients for C1 quadratic spline interpolating data X, Y
    where length(x) = N and length(y) = N, which preserves monotonicity and
    convexity in the data.

    References
    ----------
    [1] PVLib MATLAB
    [2] L. L. Schumaker, "On Shape Preserving Quadratic Spline Interpolation",
        SIAM Journal on Numerical Analysis 20(4), August 1983, pp 854 - 864
    [3] M. H. Lam, "Monotone and Convex Quadratic Spline Interpolation",
        Virginia Journal of Science 41(1), Spring 1990
    """
    # Make sure vectors are 1D arrays
    x = x.flatten()
    y = y.flatten()

    n = x.size
    assert n == y.size

    # compute various values used by the algorithm: differences, length of line
    # segments between data points, and ratios of differences.
    delx = np.diff(x)  # delx[i] = x[i + 1] - x[i]
    dely = np.diff(y)

    delta = dely / delx

    # Calculate first derivative at each x value per [3]

    s = np.zeros_like(x)

    left = np.append(0.0, delta)
    right = np.append(delta, 0.0)

    pdelta = left * right

    u = pdelta > 0

    # [3], Eq. 9 for interior points
    # fix tuning parameters in [2], Eq 9 at chi = .5 and eta = .5
    s[u] = pdelta[u] / (0.5*left[u] + 0.5*right[u])

    # [3], Eq. 7 for left endpoint
    left_end = 2.0 * delta[0] - s[1]
    if delta[0] * left_end > 0:
        s[0] = left_end

    # [3], Eq. 8 for right endpoint
    right_end = 2.0 * delta[-1] - s[-2]
    if delta[-1] * right_end > 0:
        s[-1] = right_end

    # determine knots. Start with initial points x
    # [2], Algorithm 4.1 first 'if' condition of step 5 defines intervals
    # which won't get internal knots
    tests = s[:-1] + s[1:]
    u = np.isclose(tests, 2.0 * delta, atol=EPS)
    # u = true for an interval which will not get an internal knot

    k = n + sum(~u)  # total number of knots = original data + inserted knots

    # set up output arrays
    # knot locations, first n - 1 and very last (n + k) are original data
    xk = np.zeros(k)
    yk = np.zeros(k)  # function values at knot locations
    # logicals that will indicate where additional knots are inserted
    flag = np.zeros(k, dtype=bool)
    a = np.zeros((k, 3))

    # structures needed to compute coefficients, have to be maintained in
    # association with each knot

    tmpx = x[:-1]
    tmpy = y[:-1]
    tmpx2 = x[1:]
    tmps = s[:-1]
    tmps2 = s[1:]
    diffs = np.diff(s)

    # structure to contain information associated with each knot, used to
    # calculate coefficients
    uu = np.zeros((k, 6))

    uu[:(n - 1), :] = np.array([tmpx, tmpx2, tmpy, tmps, tmps2, delta]).T

    # [2], Algorithm 4.1 subpart 1 of Step 5
    # original x values that are left points of intervals without internal
    # knots

    # XXX: MATLAB differs from NumPy, boolean indices must be same size as
    # array
    xk[:(n-1)][u] = tmpx[u]
    yk[:(n-1)][u] = tmpy[u]
    # constant term for each polynomial for intervals without knots
    a[:(n-1), 2][u] = tmpy[u]
    a[:(n-1), 1][u] = s[:-1][u]
    a[:(n-1), 0][u] = 0.5 * diffs[u] / delx[u]  # leading coefficients

    # [2], Algorithm 4.1 subpart 2 of Step 5
    # original x values that are left points of intervals with internal knots
    xk[:(n-1)][~u] = tmpx[~u]
    yk[:(n-1)][~u] = tmpy[~u]

    aa = s[:-1] - delta
    b = s[1:] - delta

    sbar = np.zeros(k)
    eta = np.zeros(k)
    # will contain mapping from the left points of intervals containing an
    # added knot to each inverval's internal knot value
    xi = np.zeros(k)

    t0 = aa * b >= 0
    # first 'else' in Algorithm 4.1 Step 5
    v = np.logical_and(~u, t0)  # len(u) == (n - 1) always
    q = np.sum(v)  # number of this type of knot to add

    if q > 0.:
        xk[(n - 1):(n + q - 1)] = .5 * (tmpx[v] + tmpx2[v])  # knot location
        uu[(n - 1):(n + q - 1), :] = np.array([tmpx[v], tmpx2[v], tmpy[v],
                                               tmps[v], tmps2[v], delta[v]]).T
        xi[:(n-1)][v] = xk[(n - 1):(n + q - 1)]

    t1 = np.abs(aa) > np.abs(b)
    w = np.logical_and(~u, ~v)  # second 'else' in Algorithm 4.1 Step 5
    w = np.logical_and(w, t1)
    r = np.sum(w)

    if r > 0.:
        xk[(n + q - 1):(n + q + r - 1)] = tmpx2[w] + aa[w] * delx[w] / diffs[w]
        uu[(n + q - 1):(n + q + r - 1), :] = np.array([tmpx[w], tmpx2[w],
                                                       tmpy[w], tmps[w],
                                                       tmps2[w], delta[w]]).T
        xi[:(n-1)][w] = xk[(n + q - 1):(n + q + r - 1)]

    z = np.logical_and(~u, ~v)  # last 'else' in Algorithm 4.1 Step 5
    z = np.logical_and(z, ~w)
    ss = np.sum(z)

    if ss > 0.:
        xk[(n + q + r - 1):(n + q + r + ss - 1)] = \
            tmpx[z] + b[z] * delx[z] / diffs[z]
        uu[(n + q + r - 1):(n + q + r + ss - 1), :] = \
            np.array([tmpx[z], tmpx2[z], tmpy[z], tmps[z], tmps2[z],
                      delta[z]]).T
        xi[:(n-1)][z] = xk[(n + q + r - 1):(n + q + r + ss - 1)]

    # define polynomial coefficients for intervals with added knots
    ff = ~u
    sbar[:(n-1)][ff] = (
        (2 * uu[:(n-1), 5][ff] - uu[:(n-1), 4][ff])
        + (uu[:(n-1), 4][ff] - uu[:(n-1), 3][ff])
        * (xi[:(n-1)][ff] - uu[:(n-1), 0][ff])
        / (uu[:(n-1), 1][ff] - uu[:(n-1), 0][ff]))
    eta[:(n-1)][ff] = (
        (sbar[:(n-1)][ff] - uu[:(n-1), 3][ff])
        / (xi[:(n-1)][ff] - uu[:(n-1), 0][ff]))

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
    a[:(n-1), 2][~u] = uu[:(n-1), 2][~u]
    a[:(n-1), 1][~u] = uu[:(n-1), 3][~u]
    a[:(n-1), 0][~u] = 0.5 * eta[:(n-1)][~u]  # leading coefficient

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
