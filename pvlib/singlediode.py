import numpy as np
from update_io_known_n import v_from_i
from calc_theta_phi_exact import calc_theta_phi_exact
from lambertw import lambertw


def i_from_v(rsh, rs, nnsvth, v, io, iphi):
    # calculates I from V per Eq 2 Jain and Kapoor 2004
    # uses Lambert W implemented in lambertw script
    # Rsh, nVth, V, Io, Iphi can all be vectors
    # Rs can be a numpy array with size greater than 1 but it should have only 1 value

    argw = rs * io * rsh * np.exp(rsh * (rs * (iphi * io) + v) / (nnsvth * (rs + rsh))) / (nnsvth * (rs + rsh))
    inputterm = lambertw(argw)

    # Eqn. 4 in Jain and Kapoor, 2004
    i = -v / (rs + rsh) - (nnsvth / rs) * inputterm + rsh * (iphi + io) / (rs + rsh)
    return i


def g(i, iph, io, a, rs, rsh):
    # calculates dP / dV exactly, using p = I * V = I * V(I), where V = V(I) uses the Lambert's W function W(phi)
    # ([3], Eq. 3)

    x, z = calc_theta_phi_exact(i, iph, 0, io, a, rs, rsh)
    z = np.transpose(z)

    # calculate dP / dV
    y = (iph + io - 2. * i) * rsh - 2. * i * rs - a * z + i * rsh * z / (1. + z)
    return y


def calc_imp_bisect(iph, io, a, rs, rsh):
    # calculates the value of imp (current at maximum power point) for an IV curve with parameters Iph, Io, a, Rs and
    # Rsh. Imp is found as the value of I for which g(I) = dP/dV (I) = 0.

    # Set up lower and upper bounds on Imp
    A = 0. * iph
    B = iph + io

    # Detect when lower and upper bounds are not consistent with finding the zero of dP / dV
    gA = g(A, iph, io, a, rs, rsh)
    gB = g(B, iph, io, a, rs, rsh)

    if any(gA * gB > 0):
        # Where gA * gB > 0, then there is a problem with the IV curve parameters. In the event where gA and gB have the
        # same sign, alert the user with a warning and replace erroneous cases with NaN
        errorvalues = gA * gB > 0
        print "Warning: singlediode has found at least one case where the singlediode parameters are such that dP/dV" \
              "may not have a zero. A NaN value has been reported for all such cases."
        A[errorvalues] = float("NaN")

    # midpoint is initial guess for Imp
    p = (A + B) / 2
    err = g(p, iph, io, a, rs, rsh)  # value of dP / dV at initial guess p

    while max(abs(B - A)) > 1e-6:  # set precision of estimate of Imp to 1e-6 (A)
        gA = g(A, iph, io, a, rs, rsh)  # value of dP / dV at left endpoint
        u = (gA * err) < 0
        B[u] = p[u]
        A[~u] = p[~u]
        p = (A + B) / 2
        err = g(p, iph, io, a, rs, rsh)

    imp = p
    return imp


def calc_pmp_bisect(iph, io, a, rs, rsh):
    # Returns Imp, Vmp, Pmp for the IV curve described by input parameters. Vectorized.

    imp = calc_imp_bisect(iph, io, a, rs, rsh)  # find imp
    x, z = calc_theta_phi_exact(imp, iph, 0, io, a, rs, rsh)  # calculate W(phi) at Imp, where W is Lambert's W function
                                                              # and phi is its argument ([3], Eq. 3)
    z = np.transpose(z)
    vmp = (iph + io - imp) * rsh - imp * rs - a * z  # compute V from Imp and W(phi)
    pmp = vmp * imp
    return imp, vmp, pmp


def singlediode(il, io, rs, rsh, nnsvth, numpoints=0):
    if any([il < 0, io < 0, rsh < 0, nnsvth < 0, numpoints < 0]):
        print "All Input Values Should be Greater Than 0"
        return
    if len(numpoints) != 1:
        print "numpoints should be numpy array of length 1"
        return
