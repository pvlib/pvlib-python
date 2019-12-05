import numpy as np


def lambertw(z):
    """
    LAMBERTW computes values for the Lambert W Function W(z).

    Syntax
    ------
    w = lambertw(z)

    Description
    -----------
    w = lambertw(z) computes the principal value of the Lambert W Function, the
    solution of z = w * exp(w). Z may be a complex scalar or array. For real z,
    the result is real on the principal branch for z >= -1/e.

    The algorithm uses series approximations as initializations and Halley's
    method as developed in Corless, Gonnet, Hare, Jeffrey, Knuth, "On the
    Lambert W Function", Advances in Computational Mathematics, volume 5, 1996
    pp. 329-359

    Original code by Pasca Getreuer 2005 - 2006, modified by Didier Clamond,
    2005. Code downloaded from http://www.getreuer.info/home/lambertw and
    modified for inclusion in PVLib.

    Matlab includes a lambertw.m function using a very similar algorithm in the
    Symbolic Math Toolbox.

    Parameters
    ----------
    z: A numpy array of values at which w(z) will be evaluated.

    Returns
    -------
    w: A numpy array of values of w(z) on the principal branch.

    References
    ----------
    [1] PVLib MATLAB
    [2] R.M. Corless, G.H. Gonnet, D.E.G. Hare, G.J. Jeffery, and D.E. Knuth.
        "On the Lambert W Function." Advances in Computational Mathematics,
        vol. 5, 1996
    """

    z = np.atleast_1d(z)
    if any(z < 0):
        f = []
        for i in z:
            if np.isnan(i) or np.isinf(i):
                f.append(float("NaN"))
                continue

            # Use a series expansion when close to the branch point -1/e
            k = (np.abs(i + 0.3678794411714423216) <= 1.5)
            # [2], Eq. 4.22 and text
            tmp = np.sqrt(complex(5.43656365691809047 * i + 2.)) - 1
            if k:
                w = tmp
            else:
                # Use asymptotic expansion w = log(z) - log(log(z)) for most z
                tmp = np.log(complex(i + (i == 0)))
                w = tmp - np.log(complex(tmp + (tmp == 0)))

            for j in range(100):
                # Converge with Halley's method ([2], Eq. 5.9), about 5
                # iterations satisfies the tolerance for most z
                c1 = np.exp(w)
                c2 = w * c1 - i
                w1 = w + (w != -1)
                dw = c2 / (c1 * w1 - ((w + 2) * c2 / (2 * w1)))
                w -= dw

                if np.abs(dw) < 0.7e-16 * (2 + np.abs(w)):
                    f.append(w)
                    break
        w = np.array(f)  # can be either float or complex!
    else:
        # Use asymptotic expansion w = log(z) - log(log(z)) for most z
        tmp = np.log(z + (z == 0))
        w = tmp - np.log(tmp + (tmp == 0))

        # Use a series expansion when close to the branch point -1/e
        k = (np.abs(z + 0.3678794411714423216) <= 1.5)
        # [2], Eq. 4.22 and text
        tmp = np.sqrt(5.43656365691809047 * z + 2.) - 1
        w[k] = tmp[k]

        for i in range(100):
            # Converge with Halley's method ([2], Eq. 5.9), about 5 iterations
            # satisfies the tolerance for most z
            c1 = np.exp(w)
            c2 = w * c1 - z
            w1 = w + (w != -1)
            dw = c2 / (c1 * w1 - ((w + 2) * c2 / (2 * w1)))
            w -= dw

            if all(np.abs(dw) < 0.7e-16 * (2 + np.abs(w))):
                break
    return w
