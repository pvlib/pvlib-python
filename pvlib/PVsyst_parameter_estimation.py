import numpy as np


def rectify_iv_curve(ti, tv, voc, isc):
    """
    rectify_IV_curve ensures that Isc and Voc are included in a IV curve and removes duplicate voltage and current
    points.

    Syntax: I, V = rectify_IV_curve(ti, tv, voc, isc)

    Description
        rectify_IV_curve ensures that the IV curve data
            * increases in voltage
            * contain no negative current or voltage values
            * have the first data point as (0, Isc)
            * have the last data point as (Voc, 0)
            * contain no duplicate voltage values. Where voltage values are
              repeated, a single data point is substituted with current equal to the
              average of current at each repeated voltage.
    :param ti: a numpy array of length N containing the current data
    :param tv: a numpy array of length N containing the voltage data
    :param voc: a int or float containing the open circuit voltage
    :param isc: a int or float containing the short circuit current
    :return: I, V: numpy arrays of equal length containing the current and voltage respectively
    """
    # Filter out negative voltage and current values
    data_filter = []
    for n, i in enumerate(ti):
        if i < 0:
            continue
        if tv[n] > voc:
            continue
        if tv[n] < 0:
            continue
        data_filter.append(n)

    current = np.array([isc])
    voltage = np.array([0.])

    for i in data_filter:
        current = np.append(current, ti[i])
        voltage = np.append(voltage, tv[i])

    # Add in Voc and Isc
    current = np.append(current, 0.)
    voltage = np.append(voltage, voc)

    # Remove duplicate Voltage and Current points
    u, index, inverse = np.unique(voltage, return_index=True, return_inverse=True)
    if len(u) != len(voltage):
        v = []
        for i in u:
            fil = []
            for n, j in enumerate(voltage):
                if i == j:
                    fil.append(n)
            t = current[fil]
            v.append(np.average(t))
        voltage = u
        current = np.array(v)
    return current, voltage


def estrsh(x, rshexp, g, go):
    # computes rsh for PVsyst model where the parameters are in vector xL
    # x[0] = Rsh0
    # x[1] = Rshref

    rsho = x[0]
    rshref = x[1]

    rshb = np.max((rshref - rsho * np.exp(-rshexp)) / (1. - np.exp(-rshexp)), 0.)
    prsh = rshb + (rsho - rshb) * np.exp(-rshexp * g / go)
    return prsh


def filter_params(io, rsh, rs, ee, isc):
    # Function filter_params identifies bad parameter sets. A bad set contains Nan, non-positive or imaginary values for
    # parameters; Rs > Rsh; or data where effective irradiance Ee differs by more than 5% from a linear fit to Isc vs.
    # Ee

    badrsh = rsh < 0. or np.isnan(rsh)
    negrs = rs < 0.
    badrs = rs > rsh or np.isnan(rs)
    imagrs = ~(np.isreal(rs))
    badio = ~(np.isreal(rs)) or io <= 0
    goodr = np.logical_and(~badrsh, ~imagrs)
    goodr = np.logical_and(goodr, ~negrs)
    goodr = np.logical_and(goodr, ~badrs)
    goodr = np.logical_and(goodr, ~badio)

    eff = np.linalg.lstsq(ee / 1000, isc)[0]
    pisc = eff * ee / 1000
    pisc_error = np.abs(pisc - isc) / isc
    badiph = pisc_error > .05  # check for departure from linear relation between Isc and Ee

    u = np.logical_and(goodr, ~badiph)
    return u
