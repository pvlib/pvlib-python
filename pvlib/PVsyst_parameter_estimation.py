import numpy as np
import matplotlib.pyplot as plt
from pvlib.est_single_diode_param import est_single_diode_param
from pvlib.update_io_known_n import update_io_known_n
plt.ion()


def numdiff(x, f):
    """
    NUMDIFF: compute first and second order derivative using possibly unequally spaced data.

    Syntax
        df, df2 = numdiff(x,f)

    Description
        numdiff computes first and second order derivatives using a 5th order formula that accounts for possibly
        unequally spaced data. Because a 5th order centered difference formula is used, numdiff returns NaNs for the
        first 2 and last 2 points in the input vector for x.

    Parameters
    ----------
    x - a numpy array of values of x
    f - a numpy array of values of the function f for which derivatives are to be computed. Must be the same length as
        x.

    Returns
    -------
    df - a numpy array of len(x) containing the first derivative of f at each point x except at the first 2 and last 2
         points
    df2 - a numpy array of len(x) containing the second derivative of f at each point x except at the first 2 and last 2
          points.

    Sources
    -------
    [1] PVLib MATLAB
    [2] M. K. Bowen, R. Smith, "Derivative formulae and errors for non-uniformly spaced points", Proceedings of the
        Royal Society A, vol. 461 pp 1975 - 1997, July 2005. DOI: 10.1098/rpsa.2004.1430
    """

    n = len(f)

    df = np.zeros(len(f))
    df2 = np.zeros(len(f))

    # first two points are special
    df[0:2] = float("Nan")
    df2[0:2] = float("Nan")

    # Last two points are special
    df[(n - 2):n] = float("Nan")
    df2[(n - 2):n] = float("Nan")

    # Rest of points. Take reference point to be the middle of each group of 5 points. Calculate displacements
    ff = np.vstack((f[0:(n - 4)], f[1:(n - 3)], f[2:(n - 2)], f[3:(n - 1)], f[4:n])).T

    a = np.vstack((x[0:(n - 4)], x[1:(n - 3)], x[2:(n - 2)], x[3:(n - 1)], x[4:n])).T - np.tile(x[2:(n - 2)], [1, 5])

    u = np.zeros(a.shape)
    l = np.zeros(a.shape)
    u2 = np.zeros(a.shape)

    u[:, 0] = a[:, 1] * a[:, 2] * a[:, 3] + a[:, 1] * a[:, 2] * a[:, 4] + a[:, 1] * a[:, 3] * a[:, 4] + \
              a[:, 2] * a[:, 3] * a[:, 4]
    u[:, 1] = a[:, 0] * a[:, 2] * a[:, 3] + a[:, 0] * a[:, 2] * a[:, 4] + a[:, 0] * a[:, 3] * a[:, 4] + \
              a[:, 2] * a[:, 3] * a[:, 4]
    u[:, 2] = a[:, 0] * a[:, 1] * a[:, 3] + a[:, 0] * a[:, 1] * a[:, 4] + a[:, 0] * a[:, 3] * a[:, 4] + \
              a[:, 1] * a[:, 3] * a[:, 4]
    u[:, 3] = a[:, 0] * a[:, 1] * a[:, 2] + a[:, 0] * a[:, 1] * a[:, 4] + a[:, 0] * a[:, 2] * a[:, 4] + \
              a[:, 1] * a[:, 2] * a[:, 4]
    u[:, 4] = a[:, 0] * a[:, 1] * a[:, 2] + a[:, 0] * a[:, 1] * a[:, 3] + a[:, 0] * a[:, 2] * a[:, 3] + \
              a[:, 1] * a[:, 2] * a[:, 3]

    l[:, 0] = (a[:, 0] - a[:, 1]) * (a[:, 0] - a[:, 2]) * (a[:, 0] - a[:, 3]) * (a[:, 0] - a[:, 4])
    l[:, 1] = (a[:, 1] - a[:, 0]) * (a[:, 1] - a[:, 2]) * (a[:, 1] - a[:, 3]) * (a[:, 1] - a[:, 4])
    l[:, 2] = (a[:, 2] - a[:, 0]) * (a[:, 2] - a[:, 1]) * (a[:, 2] - a[:, 3]) * (a[:, 2] - a[:, 4])
    l[:, 3] = (a[:, 3] - a[:, 0]) * (a[:, 3] - a[:, 1]) * (a[:, 3] - a[:, 2]) * (a[:, 3] - a[:, 4])
    l[:, 4] = (a[:, 4] - a[:, 0]) * (a[:, 4] - a[:, 1]) * (a[:, 4] - a[:, 2]) * (a[:, 4] - a[:, 3])

    df[2:(n - 2)] = np.sum(-(u / l) * ff)

    # second derivative
    u2[:, 0] = a[:, 1] * a[:, 2]+a[:, 1] * a[:, 3]+a[:, 1] * a[:, 4] + a[:, 2] * a[:, 3] + a[:, 2] * a[:, 4] + \
               a[:, 3] * a[:, 4]
    u2[:, 1] = a[:, 0] * a[:, 2]+a[:, 0] * a[:, 3]+a[:, 0] * a[:, 4] + a[:, 2] * a[:, 3] + a[:, 2] * a[:, 4] + \
               a[:, 3] * a[:, 4]
    u2[:, 2] = a[:, 0] * a[:, 1]+a[:, 0] * a[:, 3]+a[:, 0] * a[:, 4] + a[:, 1] * a[:, 3] + a[:, 1] * a[:, 3] + \
               a[:, 3] * a[:, 4]
    u2[:, 3] = a[:, 0] * a[:, 1]+a[:, 0] * a[:, 2]+a[:, 0] * a[:, 4] + a[:, 1] * a[:, 2] + a[:, 1] * a[:, 4] + \
               a[:, 2] * a[:, 4]
    u2[:, 4] = a[:, 0] * a[:, 1]+a[:, 0] * a[:, 2]+a[:, 0] * a[:, 3] + a[:, 1] * a[:, 2] + a[:, 1] * a[:, 4] + \
               a[:, 2] * a[:, 3]

    df2[2:(n - 2)] = 2. * np.sum(u2 * ff)
    return df, df2


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


class ConvergeParam:
    def __init__(self):
        self.imperrmax = 0.
        self.imperrmin = 0.
        self.imperrabsmax = 0.
        self.imperrmean = 0.
        self.imperrstd = 0.

        self.vmperrmax = 0.
        self.vmperrmin = 0.
        self.vmperrabsmax = 0.
        self.vmperrmean = 0.
        self.vmperrstd = 0.

        self.pmperrmax = 0.
        self.pmperrmin = 0.
        self.pmperrabsmax = 0.
        self.pmperrmean = 0.
        self.pmperrstd = 0.

        self.imperrabsmaxchange = 0.
        self.vmperrabsmaxchange = 0.
        self.pmperrabsmaxchange = 0.
        self.imperrmeanchange = 0.
        self.vmperrmeanchange = 0.
        self.pmperrmeanchange = 0.
        self.imperrstdchange = 0.
        self.vmperrstdchange = 0.
        self.pmperrstdchange = 0.

        self.state = 0.

    def __str__(self):
        return "ConvergeParam: \n Max Imp Error = %s \n Min Imp Error = %s \n Absolute Max Imp Error = %s \n " \
               "Mean Imp Error = %s \n Standard Deviation of Imp Error = %s \n\n Max Vmp Error = %s \n Min Vmp " \
               "Error = %s \n Absolute Max Vmp Error = %s \n Mean Vmp Error = %s \n Standard Deviation of Vmp Error " \
               "= %s \n\n Max Pmp Error = %s \n Min Pmp Error = %s \n Absolute Max Pmp Error = %s \n Mean Pmp " \
               "Error = %s \n Standard Deviation of Pmp Error = %s \n\n Imp Changes: \n Absolute Max Error: %s \n " \
               "Mean Error: %s \n Standard Deviation: %s \n\n Vmp Changes: \n Absolute Max Error: %s \n Mean " \
               "Error: %s \n Standard Deviation: %s \n\n Pmp Changes: \n Absolute Max Error: %s \n Mean Error: %s \n " \
               "Standard Deviation: %s \n State = %s" \
               % (self.imperrmax, self.imperrmin, self.imperrabsmax, self.imperrmean, self.imperrstd, self.vmperrmax,
                  self.vmperrmin, self.vmperrabsmax, self.vmperrmean, self.vmperrstd, self.pmperrmax, self.pmperrmin,
                  self.pmperrabsmax, self.pmperrmean, self.pmperrstd, self.imperrabsmaxchange, self. imperrmeanchange,
                  self.imperrstdchange, self.vmperrabsmaxchange, self.vmperrmeanchange, self.vmperrstdchange,
                  self.pmperrabsmaxchange, self.pmperrmeanchange, self.pmperrstdchange, self.state)

    def __repr__(self):
        return "<\nConvergeParam: \n Max Imp Error = %s \n Min Imp Error = %s \n Absolute Max Imp Error = %s \n " \
               "Mean Imp Error = %s \n Standard Deviation of Imp Error = %s \n\n Max Vmp Error = %s \n Min Vmp " \
               "Error = %s \n Absolute Max Vmp Error = %s \n Mean Vmp Error = %s \n Standard Deviation of Vmp Error " \
               "= %s \n\n Max Pmp Error = %s \n Min Pmp Error = %s \n Absolute Max Pmp Error = %s \n Mean Pmp " \
               "Error = %s \n Standard Deviation of Pmp Error = %s \n\n Imp Changes: \n Absolute Max Error: %s \n " \
               "Mean Error: %s \n Standard Deviation: %s \n\n Vmp Changes: \n Absolute Max Error: %s \n Mean " \
               "Error: %s \n Standard Deviation: %s \n\n Pmp Changes: \n Absolute Max Error: %s \n Mean Error: %s \n " \
               "Standard Deviation: %s \n State = %s \n>" \
               % (self.imperrmax, self.imperrmin, self.imperrabsmax, self.imperrmean, self.imperrstd, self.vmperrmax,
                  self.vmperrmin, self.vmperrabsmax, self.vmperrmean, self.vmperrstd, self.pmperrmax, self.pmperrmin,
                  self.pmperrabsmax, self.pmperrmean, self.pmperrstd, self.imperrabsmaxchange, self. imperrmeanchange,
                  self.imperrstdchange, self.vmperrabsmaxchange, self.vmperrmeanchange, self.vmperrstdchange,
                  self.pmperrabsmaxchange, self.pmperrmeanchange, self.pmperrstdchange, self.state)


def check_converge(prevparams, result, vmp, imp, graphic, convergeparamsfig, i):
    """
    Function check_converge computes convergence metrics for all IV curves.

    Parameters
    ----------
    prevparams: Convergence Parameters from the previous Iteration (used to determine Percent Change in values between
                iterations)
    result: performacne paramters of the (predicted) single diode fitting, which includes Voc, Vmp, Imp, Pmp and Isc
    vmp: measured values for each IV curve
    imp: measured values for each IV curve
    graphic: argument to determine whether to display Figures
    convergeparamsfig: Hangle to the ConvergeParam Plot
    i: Index of current iteration in cec_parameter_estimation

    Returns
    -------
    convergeparam - a class containing the following for Imp, Vmp and Pmp:
        - maximum percent difference between measured and modeled values
        - minimum percent difference between measured and modeled values
        - maximum absolute percent difference between measured and modeled values
        - mean percent difference between measured and modeled values
        - standard deviation of percent difference between measured and modeled values
        - absolute difference for previous and current values of maximum absolute percent difference
          (measured vs. modeled)
        - absolute difference for previous and current values of mean percent difference (measured vs. modeled)
        - absolute difference for previous and current values of standard deviation of percent difference
          (measured vs. modeled)

    """
    convergeparam = ConvergeParam()

    imperror = (result.imp - imp) / imp * 100.
    vmperror = (result.vmp - vmp) / vmp * 100.
    pmperror = (result.pmp - (imp * vmp)) / (imp * vmp) * 100.

    convergeparam.imperrmax = max(imperror)  # max of the error in Imp
    convergeparam.imperrmin = min(imperror)  # min of the error in Imp
    convergeparam.imperrabsmax = max(abs(imperror))  # max of the absolute error in Imp
    convergeparam.imperrmean = np.mean(imperror, axis=0)  # mean of the error in Imp
    convergeparam.imperrorstd = np.std(imperror, axis=0, ddof=1)  # std of the error in Imp

    convergeparam.vmperrmax = max(vmperror)  # max of the error in Vmp
    convergeparam.vmperrmin = min(vmperror)  # min of the error in Vmp
    convergeparam.vmperrabsmax = max(abs(vmperror))  # max of the absolute error in Vmp
    convergeparam.vmperrmean = np.mean(vmperror, axis=0)  # mean of the error in Vmp
    convergeparam.vmperrorstd = np.std(vmperror, axis=0, ddof=1)  # std of the error in Vmp

    convergeparam.pmperrmax = max(pmperror)  # max of the error in Pmp
    convergeparam.pmperrmin = min(pmperror)  # min of the error in Pmp
    convergeparam.pmperrabsmax = max(abs(pmperror))  # max of the abs err. in Pmp
    convergeparam.pmperrmean = np.mean(pmperror, axis=0)  # mean error in Pmp
    convergeparam.pmperrorstd = np.std(pmperror, axis=0, ddof=1)  # std error Pmp

    if prevparams.state != 0.:
        convergeparam.imperrstdchange = np.abs((convergeparam.imperrstd - prevparams.imperrstd) / prevparams.imperrstd)
        convergeparam.vmperrstdchange = np.abs((convergeparam.vmperrstd - prevparams.vmperrstd) / prevparams.vmperrstd)
        convergeparam.pmperrstdchange = np.abs((convergeparam.pmperrstd - prevparams.pmperrstd) / prevparams.PmpErrStd)
        convergeparam.imperrmeanchange = np.abs((convergeparam.imperrmean - prevparams.imperrmean) /
                                                prevparams.imperrmean)
        convergeparam.vmperrmeanchange = np.abs((convergeparam.vmperrmean - prevparams.vmperrmean) /
                                                prevparams.vmperrmean)
        convergeparam.pmperrmeanchange = np.abs((convergeparam.pmperrmean - prevparams.pmperrmean) /
                                                prevparams.pmperrmean)
        convergeparam.imperrabsmaxchange = np.abs((convergeparam.imperrabsmax - prevparams.imperrabsmax) /
                                                  prevparams.imperrabsmax)
        convergeparam.vmperrabsmaxchange = np.abs((convergeparam.vmperrabsmax - prevparams.vmperrabsmax) /
                                                  prevparams.vmperrabsmax)
        convergeparam.pmperrabsmaxchange = np.abs((convergeparam.pmperrabsmax - prevparams.pmperrabsmax) /
                                                  prevparams.pmperrabsmax)
        convergeparam.state = 1.
    else:
        convergeparam.imperrstdchange = float("Inf")
        convergeparam.vmperrstdchange = float("Inf")
        convergeparam.pmperrstdchange = float("Inf")
        convergeparam.imperrmeanchange = float("Inf")
        convergeparam.vmperrmeanchange = float("Inf")
        convergeparam.pmperrmeanchange = float("Inf")
        convergeparam.imperrabsmaxchange = float("Inf")
        convergeparam.vmperrabsmaxchange = float("Inf")
        convergeparam.pmperrabsmaxchange = float("Inf")
        convergeparam.state = 1.

    if graphic:
        ax1 = convergeparamsfig.add_subplot(331)
        ax2 = convergeparamsfig.add_subplot(332)
        ax3 = convergeparamsfig.add_subplot(333)
        ax4 = convergeparamsfig.add_subplot(334)
        ax5 = convergeparamsfig.add_subplot(335)
        ax6 = convergeparamsfig.add_subplot(336)
        ax7 = convergeparamsfig.add_subplot(337)
        ax8 = convergeparamsfig.add_subplot(338)
        ax9 = convergeparamsfig.add_subplot(339)
        ax1.plot(i, convergeparam.pmperrmean, 'x-')
        ax1.set_ylabel('mean((pPmp-Pmp)/Pmp*100)')
        ax1.set_title('Mean of Err in Pmp')
        ax1.hold(True)
        ax2.plot(i, convergeparam.vmperrmean, 'x-')
        ax2.set_ylabel('mean((pVmp-Vmp)/Vmp*100)')
        ax2.set_title('Mean of Err in Vmp')
        ax2.hold(True)
        ax3.plot(i, convergeparam.imperrmean, 'x-')
        ax3.set_ylabel('mean((pImp-Imp)/Imp*100)')
        ax3.set_title('Mean of Err in Imp')
        ax3.hold(True)
        ax4.plot(i, convergeparam.pmperrstd, 'x-')
        ax4.set_ylabel('std((pPmp-Pmp)/Pmp*100)')
        ax4.set_title('Std of Err in Pmp')
        ax4.hold(True)
        ax5.plot(i, convergeparam.vmperrstd, 'x-')
        ax5.set_ylabel('std((pVmp-Vmp)/Vmp*100)')
        ax5.set_title('Std of Err in Vmp')
        ax5.hold(True)
        ax6.plot(i, convergeparam.imperrstd, 'x-')
        ax6.set_ylabel('std((pImp-Imp)/Imp*100)')
        ax6.set_title('Std of Err in Imp')
        ax6.hold(True)
        ax7.plot(i, convergeparam.pmperrabsmax, 'x-')
        ax7.set_xlabel('Iteration')
        ax7.set_ylabel('max(abs((pPmp-Pmp)/Pmp*100))')
        ax7.set_title('AbsMax of Err in Pmp')
        ax7.hold(True)
        ax8.plot(i, convergeparam.vmperrabsmax, 'x-')
        ax8.set_xlabel('Iteration')
        ax8.set_ylabel('max(abs((pVmp-Vmp)/Vmp*100))')
        ax8.set_title('AbsMax of Err in Vmp')
        ax8.hold(True)
        ax9.plot(i, convergeparam.imperrabsmax, 'x-')
        ax9.set_xlabel('Iteration')
        ax9.set_ylabel('max(abs((pImp-Imp)/Imp*100))')
        ax9.set_title('AbsMax of Err in Imp')
        ax9.hold(True)
    return convergeparam


class CONST:
    def __init__(self):
        self.eo = 1000.
        self.to = 25.
        self.k = 1.38066e-23
        self.q = 1.60218e-19

    def __str__(self):
        return "Const: \n E0 = %s \n T0 = %s \n k = %s \n q = %s" % (self.eo, self.to, self.k, self.q)

    def __repr__(self):
        return "<\nConst: \n E0 = %s \n T0 = %s \n k = %s \n q = %s\n>" % (self.eo, self.to, self.k, self.q)

const_default = CONST()


class PVSYST:
    def __init__(self):
        self.il_ref = 0.
        self.io_ref = 0.
        self.eg = 0.
        self.rsh_ref = 0.
        self.rsho = 0.
        self.rshexp = 0.
        self.rs_ref = 0.
        self.gamma_ref = 0.
        self.mugamma = 0.
        self.iph = 0.
        self.io = 0.
        self.rsh = 0.
        self.rs = 0.
        self.ns = 0.
        self.u = 0.

    def __str__(self):
        return "PVsyst Parameters: \n IL_ref = %s \n Io_ref = %s \n eG = %s \n Rsh_ref = %s \n Rsh0 = %s \n " \
               "Rshexp = %s \n Rs_ref = %s \n gamma_ref = %s \n mugamma = %s \n Ns = %s" % \
               (self.il_ref, self.io_ref, self.eg, self.rsh_ref, self.rsho, self.rshexp, self.rs_ref, self.gamma_ref,
                self.mugamma, self.ns)

    def __repr__(self):
        return "<\nPVsyst Parameters: \n IL_ref = %s \n Io_ref = %s \n eG = %s \n Rsh_ref = %s \n Rsh0 = %s \n " \
               "Rshexp = %s \n Rs_ref = %s \n gamma_ref = %s \n mugamma = %s \n Ns = %s\n>" % \
               (self.il_ref, self.io_ref, self.eg, self.rsh_ref, self.rsho, self.rshexp, self.rs_ref, self.gamma_ref,
                self.mugamma, self.ns)


def pvsyst_parameter_estimation(ivcurves, specs, const=const_default, maxiter=5, eps1=1e-3, graphic=False):
    """
    pvsyst_parameter_estimation estimates parameters fro the PVsyst module performance model

    Syntax
        PVsyst, oflag = pvsyst_paramter_estimation(ivcurves, specs, const, maxiter, eps1, graphic)

    Description
        pvsyst_paramter_estimation estimates parameters for the PVsyst module performance model [2,3,4]. Estimation
        methods are documented in [5,6,7].

    Parameters
    ----------
    ivcurves - a class containing IV curve data in the following fields where j denotes the jth data set
        ivcurves.i[j] - a numpy array of current (A) (same length as v)
        ivcurves.v[j] - a numpy array of voltage (V) (same length as i)
        ivcurves.ee[j] - effective irradiance (W / m^2), i.e., POA broadband irradiance adjusted by solar spectrum
                         modifier
        ivcurves.tc[j] - cell temperature (C)
        ivcurves.isc - short circuit current of IV curve (A)
        ivcurves.voc - open circuit voltage of IV curve (V)
        ivcurves.imp - current at max power point of IV curve (A)
        ivcurves.vmp - voltage at max power point of IV curve (V)

    specs - a class containing module-level values
        specs.ns - number of cells in series
        specs.aisc - temperature coefficeint of isc (A/C)

    const - an optional class containing physical and other constants
        const.eo - effective irradiance at STC, normally 1000 W/m2
        const.to - cell temperature at STC, normally 25 C
        const.k - 1.38066E-23 J/K (Boltzmann's constant)
        const.q - 1.60218E-19 Coulomb (elementary charge)

    maxiter - an optional numpy array input that sets the maximum number of iterations for the parameter updating part
              of the algorithm. Default value is 5.

    eps1 - the desired tolerance for the IV curve fitting. The iterative parameter updating stops when absolute values
           of the percent change in mean, max and standard deviation of Imp, Vmp and Pmp between iterations are all less
           than eps1, or when the number of iterations exceeds maxiter. Default value is 1e-3 (.0001%).

    graphic - a boolean, if true then plots are produced during the parameter estimation process. Default is false

    Returns
    -------
    pvsyst - a class containing the model parameters
        pvsyst.il_ref - light current (A) at STC
        pvsyst.io_ref - dark current (A) at STC
        pvsyst.eg - effective band gap (eV) at STC
        pvsyst.rsh_ref - shunt resistance (ohms) at STC
        pvsyst.rsho - shunt resistance (ohms) at zero irradiance
        pvsyst.rshexp - exponential factor defining decrease in rsh with increasing effective irradiance
        pvsyst.rs_ref - series resistance (ohms) at STC
        pvsyst.gamma_ref - diode (ideality) factor at STC
        pvsyst.mugamma - temperature coefficient for diode (ideality) factor
        pvsyst.iph - numpy array of values of light current Iph estimated for each IV curve
        pvsyst.io - numpy array of values of dark current Io estimated for each IV curve
        pvsyst.rsh - numpy array of values of shunt resistance Rsh estimated for each IV curve
        pvsyst.rs - numpy array of values of series resistance Rs estimated for each IV curve
        pvsyst.u - filter indicating IV curves with parameter values deemed reasonable by the private function
                   filter_params

    oflag - Boolean indicating success or failure of estimation of the diode (ideality) factor parameter. If failur,
            then no parameter values are returned

    Sources:
    [1] PVLib MATLAB
    [2] K. Sauer, T. Roessler, C. W. Hansen, Modeling the Irradiance and Temperature Dependence of Photovoltaic Modules
        in PVsyst, IEEE Journal of Photovoltaics v5(1), January 2015.
    [3] A. Mermoud, PV Modules modeling, Presentation at the 2nd PV Performance Modeling Workshop, Santa Clara, CA, May
        2013
    [4] A. Mermoud, T. Lejeuene, Performance Assessment of a Simulation Model for PV modules of any available
        technology, 25th European Photovoltaic Solar Energy Conference, Valencia, Spain, Sept. 2010
    [5] C. Hansen, Estimating Parameters for the PVsyst Version 6 Photovoltaic Module Performance Model, Sandia National
        Laboratories Report SAND2015-8598
    [6] C. Hansen, Parameter Estimation for Single Diode Models of Photovoltaic Modules, Sandia National Laboratories
        Report SAND2015-2065
    [7] C. Hansen, Estimation of Parameters for Single Diode Models using Measured IV Curves, Proc. of the 39th IEEE
        PVSC, June 2013.
    """
    ee = ivcurves.ee
    tc = ivcurves.tc
    tck = tc + 273.15
    isc = ivcurves.isc
    voc = ivcurves.voc
    imp = ivcurves.imp
    vmp = ivcurves.vmp

    # Cell Thermal Voltage
    vth = const.k / const.q * tck

    n = len(ivcurves.voc)

    # Initial estimate of Rsh used to obtain the diode factor gamma0 and diode temperature coefficient mugamma. Rsh is
    # estimated using the co-content integral method.

    pio = np.ones(n)
    piph = np.ones(n)
    prsh = np.ones(n)
    prs = np.ones(n)
    pn = np.ones(n)

    for j in range(n):
        current, voltage = rectify_iv_curve(ivcurves.i[j], ivcurves.v[j], voc[j], isc[j])
        # initial estimate of Rsh, from integral over voltage regression
        # [5] Step 3a; [6] Step 3a
        pio[j], piph[j], prs[j], prsh[j], pn[j] = est_single_diode_param(current, voltage, vth[j] * specs.ns)

    # Estimate the diode factor gamma from Isc-Voc data. Method incorporates temperature dependence by means of the
    # equation for Io

    y = np.log(isc - voc / prsh) - 3. * np.log(tck / (const.to + 273.15))
    x1 = const.q / const.k * (1. / (const.to + 273.15) - 1. / tck)
    x2 = voc / (vth * specs.ns)
    t0 = np.isnan(y)
    t1 = np.isnan(x1)
    t2 = np.isnan(x2)
    uu = np.logical_or(t0, t1)
    uu = np.logical_or(uu, t2)

    x = np.vstack((np.ones(len(x1[~uu])), x1[~uu], -x1[~uu] * (tck[~uu] - (const.to + 273.15)), x2[~uu],
                   -x2[~uu] * (tck[~uu] - (const.to + 273.15)))).T
    alpha = np.linalg.lstsq(x, y[~uu])[0]

    gamma_ref = 1. / alpha[3]
    mugamma = alpha[4] / alpha[3] ** 2

    if np.isnan(gamma_ref) or np.isnan(mugamma) or not np.isreal(gamma_ref) or not np.isreal(mugamma):
        badgamma = True
    else:
        badgamma = False

    if ~badgamma:
        pvsyst = PVSYST()

        gamma = gamma_ref + mugamma * (tc - const.to)

        if graphic:
            f1 = plt.figure()
            ax1 = f1.add_subplot(111)
            ax1.plot(x2, y, 'b+', x2, x * alpha, 'r.')
            ax1.set_xlabel('X = Voc / Ns * Vth')
            ax1.set_ylabel('Y = log(Isc - Voc/Rsh)')
            ax1.legend(['I-V Data', 'Regression Model'], loc=2)
            ax1.text(np.min(x2) + 0.85 * (np.max(x2) - np.min(x2)), 1.05 * np.max(y), ['\gamma_0 = %s' % gamma_ref])
            ax1.text(np.min(x2) + 0.85 * (np.max(x2) - np.min(x2)), 0.98 * np.max(y), ['\mu_\gamma = %s' % mugamma])

        nnsvth = gamma * (vth * specs.ns)

        # For each IV curve, sequentially determine initial values for Io, Rs, and Iph
        # [5] Step 3a; [6] Step 3

        io = np.ones(n)
        iph = np.ones(n)
        rs = np.ones(n)
        rsh = prsh

        for j in range(n):
            curr, volt = rectify_iv_curve(ivcurves.i[j], ivcurves.v[j], voc[j], isc[j])

            if rsh[j] > 0:
                # Initial estimate of Io, evaluate the single diode model at voc and approximate Iph + Io = Isc
                # [5] Step 3a; [6] Step 3b
                io[j] = (isc[j] - voc[j] / rsh[j]) * np.exp(-voc[j] / nnsvth[j])

                # initial estimate of rs from dI/dV near Voc
                # [5] Step 3a; [6] Step 3c
                [didv, d2id2v] = numdiff(volt, curr)
                t3 = volt > .5 * voc[j]
                t4 = volt < .9 * voc[j]
                u = np.logical_and(t3, t4)
                tmp = -rsh[j] * didv - 1.
                v = np.logical_and(u, tmp > 0)
                if np.sum(v) > 0:
                    vtrs = nnsvth[j] / isc[j] * (np.log(tmp[v] * nnsvth[j] / (rsh[j] * io[j])) - volt[v] / nnsvth[j])
                    rs[j] = np.mean(vtrs[vtrs > 0], axis=0)
                else:
                    rs[j] = 0.

                # Initial estimate of Iph, evaluate the single diode model at Isc
                # [5] Step 3a; [6] Step 3d

                iph[j] = isc[j] - io[j] + io[j] * np.exp(isc[j] / nnsvth[j]) + isc[j] * rs[j] / rsh[j]
            else:
                io[j] = float("Nan")
                rs[j] = float("Nan")
                iph[j] = float("Nan")

        # Filter IV curves for good initial values
        # [5] Step 3b
        u = filter_params(io, rsh, rs, ee, isc)

        # Refine Io to match Voc
        # [5] Step 3c
        tmpiph = iph
        tmpio = update_io_known_n(rsh[u], rs[u], nnsvth[u], io[u], tmpiph[u], voc[u])
        io[u] = tmpio

        # Calculate Iph to be consistent with Isc and current values of other parameters
        # [6], Step 3c
        iph = isc - io + io * np.exp(rs * isc / nnsvth) + isc * rs / rsh

        # Refine Rsh, Rs, Io and Iph in that order.
    else:
        oflag = False
        pvsyst = PVSYST()

        pvsyst.il_ref = float("Nan")
        pvsyst.io_ref = float("Nan")
        pvsyst.eg = float("Nan")
        pvsyst.rs_ref = float("Nan")
        pvsyst.gamma_ref = float("Nan")
        pvsyst.mugamma = float("Nan")
        pvsyst.iph = float("Nan")
        pvsyst.io = float("Nan")
        pvsyst.rsho = float("Nan")
        pvsyst.rsh_ref = float("Nan")
        pvsyst.rshexp = float("Nan")
        pvsyst.rs = float("Nan")
        pvsyst.rsh = float("Nan")
        pvsyst.ns = specs.ns
        pvsyst.u = np.zeros(n)
    return pvsyst, oflag
