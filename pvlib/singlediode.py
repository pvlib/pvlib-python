import numpy as np
from pvlib.update_io_known_n import v_from_i
from pvlib.calc_theta_phi_exact import calc_theta_phi_exact
from pvlib.lambertw import lambertw


def i_from_v(rsh, rs, nnsvth, v, io, iphi):
    # calculates I from V per Eq 2 Jain and Kapoor 2004
    # uses Lambert W implemented in lambertw script
    # Rsh, nVth, V, Io, Iphi can all be vectors
    # Rs can be a numpy array with size greater than 1 but it should have only 1 value

    argw = rs * io * rsh * np.exp(rsh * (rs * (iphi + io) + v) / (nnsvth * (rs + rsh))) / (nnsvth * (rs + rsh))
    inputterm = lambertw(argw)

    # Eqn. 4 in Jain and Kapoor, 2004
    i = -v / (rs + rsh) - (nnsvth / rs) * inputterm + rsh * (iphi + io) / (rs + rsh)
    return i


def g(i, iph, io, a, rs, rsh):
    # calculates dP / dV exactly, using p = I * V = I * V(I), where V = V(I) uses the Lambert's W function W(phi)
    # ([3], Eq. 3)

    x, z = calc_theta_phi_exact(i, iph, np.array([0.]), io, a, rs, rsh)
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
        print("Warning: singlediode has found at least one case where the singlediode parameters are such that dP/dV "
              "may not have a zero. A NaN value has been reported for all such cases.")
        A[errorvalues] = float("NaN")

    # midpoint is initial guess for Imp
    p = (A + B) / 2.
    err = g(p, iph, io, a, rs, rsh)  # value of dP / dV at initial guess p

    while max(abs(B - A)) > 1e-6:  # set precision of estimate of Imp to 1e-6 (A)
        gA = g(A, iph, io, a, rs, rsh)  # value of dP / dV at left endpoint
        u = (gA * err) < 0
        B[u] = p[u]
        A[~u] = p[~u]
        p = (A + B) / 2.
        err = g(p, iph, io, a, rs, rsh)

    imp = p
    return imp


class Results:
    def __init__(self):
        self.voc = 0.
        self.vmp = 0.
        self.imp = 0.
        self.ix = 0.
        self.ixx = 0.
        self.pmp = 0.
        self.isc = 0.

    def __str__(self):
        return "Results: \n Voc = %s \n Vmp = %s \n Imp = %s \n Ix = %s \n Ixx = %s \n Pmp = %s \n Isc = %s" \
               % (self.voc, self.vmp, self.imp, self.ix, self.ixx, self.pmp, self.isc)

    def __repr__(self):
        return "<\nResults: \n Voc = %s \n Vmp = %s \n Imp = %s \n Ix = %s \n Ixx = %s \n Pmp = %s \n Isc = %s \n>" \
               % (self.voc, self.vmp, self.imp, self.ix, self.ixx, self.pmp, self.isc)


def calc_pmp_bisect(iph, io, a, rs, rsh):
    # Returns Imp, Vmp, Pmp for the IV curve described by input parameters. Vectorized.

    imp = calc_imp_bisect(iph, io, a, rs, rsh)  # find imp
    x, z = calc_theta_phi_exact(imp, iph, np.array([0.]), io, a, rs, rsh)  # calculate W(phi) at Imp, where W is
    # Lambert's W function and phi is its argument ([3], Eq. 3)
    z = np.transpose(z)
    vmp = (iph + io - imp) * rsh - imp * rs - a * z  # compute V from Imp and W(phi)
    pmp = vmp * imp
    return imp, vmp, pmp


def singlediode(il, io, rs, rsh, nnsvth, numpoints=np.array([0])):
    """
    single diode solves the single-diode model to obtain a photovoltaic IV curve

    Syntax
        isc, voc, imp, vmp, pmp, ix, ixx, v, i = singlediode(il, io, rs, rsh, nnsvth)
        isc, voc, imp, vmp, pmp, ix, ixx, v, i = singlediode(il, io, rs, rsh, nnsvth, numpoints)

    Description
        singlediode solves the single diode equation [2]:
        I = IL - I0 * [exp((V + I * Rs) / (nNsVth)) - 1] - (V + I * Rs) / Rsh for I and V when given IL, I0, Rs, Rsh and
        nNsVth (nNsVth = n * Ns * Vth) which are described later. singlediode returns a series of values including the
        5 points on the IV curve specidified in SAND2004-3535 [4], and can optionally provide a full IV curve with a
        user-defined number of points. If all IL, I0, Rs, Rsh and nNsVth are scalar, a single curve will be returned, if
        any are vectors (of the same length), multiple IV curves will be calculated.

    Parameters
    ----------
    il - Light-generated current (photocurrent) in amperes under desired IV curve conditions. Must be a numpy array of
         length 1 or N but all vectors must be of the same length.
    io - diode saturation current in amperes under desired IV curve conditions. Must be a numpy array of length 1 or N
         but all vectors must be of the same length.
    rs - Series resistance in ohms under desired IV curve conditions. Must be a numpy array of length 1 or N but all
         vectors must be of the same length.
    rsh - Shunt resistance in ohms under desired IV curve conditions. Must be a numpy array of length 1 or N but all
          vectors must be of the same length.
    nnsvth - the product of three components. 1) The usual diode ideal factor (n), 2) the number of cells in series (Ns)
             and 3) the cell thermal voltage under the desired IV curve conditions (Vth). The thermal voltage of the
             cell (in volts) may be calculated as k*Tcell/q, where k is the Boltzmann's constant (J/K), Tcell is the
             temperature of p-n junction in Kelvin, and q is the elementary charge of an electron (coulombs). nNsVth
             must be a numpy array of length 1 or N but all vectors must be of the same length.
    numpoints - Number of points in the desired IV curve (optional). Must be a finite, scalar value. Non-integer values
                will be rounded to the next highest integer (ceil). If cel(NumPoints) is < 2, no IV curves will be
                produced (i.e. v and i will not be generated). The default value is 0, resulting in no calculation of
                the IV points other than those specified in [4].

    Returns
    -------
    isc - vector of short circuit currents in amperes.
    voc - vector of open circuit voltages in volts.
    imp - vector of currents at maximum power point in amperes.
    vmp - vector of voltages at maximum power point in volts.
    pmp - vector of powers at maximum power point in watts.
    ix - vector of currents in amperes, at V = 0.5 * Voc.
    ixx - vector of currents in amperes, at V = 0.5 * (Voc + Vmp).
    v - array of voltages in volts. Row n corresponds to IV curve n, with V = 0 in the leftmost column and V = Voc in
        the rightmost column. Thus, voc[n] = v[n, ceil(NumPoints)]. Voltage points are equally spaced (in voltage)
        between 0 and Voc.
    i - array of currents in amperes. Row n corresponds to IV curve n, with I = Isc in the leftmost column and I = 0 in
        the rightmost column. Thus, isc[n] = i[n, 1].

    Notes
    -------
    1) to plot IV curve r, use:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.plot(v[r],i[r])
    2) to plot all of the IV curves on the same plot, use:
        import matplotlib.pyplot as plt
        import numpy as np
        plt.ion()
        plt.plot(np.transpose(v),np.transpose(i))
    3) Generating IV curves using NumPoints will slow down function operation.
    4) The solution employed to solve the implicit diode equation utilizes the Lambert W function to obtain an explicit
    function of V = f(i) and I = f(V) as shown in [3].

    Sources
    -------

    [1] PVLib MATLAB
    [2] S.R. Wenham, M.A. Green, M.E. Watt, "Applied Photovoltaics" ISBN 0 86758 909 4
    [3] A. Jain, A. Kapoor, "Exact analytical solutions of the parameters of real solar cells using Lambert W-function",
        Solar Energy Materials and Solar Cells, 81 (2004) 269 - 277.
    [4] D. King et al, "Sandia Photovoltaic Array Performacne Model", SAND2004-3535, Sandia National Laboratories,
        Albuquergue, NM

    See Also
        PVL_SAPM  PVL_CALCPARAMS_DESOTO
    """

    # Vth is the thermal voltage of the cell in volts. n is the usual diode ideality factor, assumed to be linear.
    # nnsvth is n * Ns * Vth

    t0 = np.append(il < 0, np.append(io < 0, np.append(rs < 0, np.append(rsh < 0, np.append(nnsvth < 0,
                                                                                            numpoints < 0)))))
    if t0.any():
        raise ValueError("All Input Values Should be Greater Than 0")
    if len(numpoints) != 1:
        raise ValueError("numpoints should be numpy array of length 1")
    if ~np.isfinite(numpoints):
        raise ValueError("numpoints should be finite")

    # Ensure that all input values are either numpy arrays of length 1 or are all numpy arrays of the same length.
    vectorsizes = np.array([len(il), len(io), len(rs), len(rsh), len(nnsvth)])
    maxvectorsize = max(vectorsizes)
    t1 = vectorsizes == maxvectorsize
    t2 = vectorsizes == 1

    if not all(np.logical_or(t1, t2)):
        raise ValueError("Input vectors il, io, rs, rsh and nnsvth must be numpy arrays of the same length or of "
                         "length 1")

    if maxvectorsize > 1 and any(vectorsizes == 1.):
        il = il * np.ones(maxvectorsize)
        io = io * np.ones(maxvectorsize)
        rs = rs * np.ones(maxvectorsize)
        rsh = rsh * np.ones(maxvectorsize)
        nnsvth = nnsvth * np.ones(maxvectorsize)

    imax = np.zeros(maxvectorsize)
    pmp = np.zeros(maxvectorsize)
    vmax = np.zeros(maxvectorsize)
    ix = np.zeros(maxvectorsize)
    ixx = np.zeros(maxvectorsize)
    voc = np.zeros(maxvectorsize)
    isc = np.zeros(maxvectorsize)

    u = il > 0.

    # take care of any pesky non-integers by rounding up
    numpoints = np.ceil(numpoints)

    # Find isc using lambert W
    isc[u] = i_from_v(rsh[u], rs[u], nnsvth[u], np.array([0.]), io[u], il[u])

    # Find voc using lambert W
    voc[u] = v_from_i(rsh[u], rs[u], nnsvth[u], np.array([0.]), io[u], il[u])

    # Calculate I, V and P at the maximum power point
    imax[u], vmax[u], pmp[u] = calc_pmp_bisect(il[u], io[u], nnsvth[u], rs[u], rsh[u])

    # Find Ix and Ixx using Lambert W
    ix[u] = i_from_v(rsh[u], rs[u], nnsvth[u], 0.5 * voc[u], io[u], il[u])
    ixx[u] = i_from_v(rsh[u], rs[u], nnsvth[u], 0.5 * (voc[u] + vmax[u]), io[u], il[u])

    # If the user says they want a curve with number of points equal to NumPoints (must be >= 2), then create a voltage
    # array where voltage is zero in the first column, and Voc in the last column. Number of columns must be equal to
    # NumPoints. Each row represents the voltage for one IV curve. Then create a current array where current is Isc in
    # the first column, and zero in the last column, and each row represents the current in one IV curve. Thus the nth
    # (V, I) point of curve m would be found as follows: (v[m, n], i[m, n])

    v = np.array([])
    i = np.array([])
    if numpoints >= 2:
        for j in range(maxvectorsize):
            vc = np.arange(0., voc[j], voc[j] / (numpoints - 1.))
            vc = np.append(vc, voc[j])
            if u[j]:
                ic = i_from_v(rsh[j], rs[j], nnsvth[j], vc, io[j], il[j])
                ic[len(ic) - 1.] = 0.
            else:
                ic = np.zeros(len(vc))
            if j == 0:
                v = vc
                i = ic
            else:
                v = np.vstack((v, vc))
                i = np.vstack((i, ic))

    # Wrap answers in Results Class
    result = Results()

    result.imp = imax
    result.vmp = vmax
    result.isc = isc
    result.voc = voc
    result.pmp = pmp
    result.ix = ix
    result.ixx = ixx

    return result, v, i
