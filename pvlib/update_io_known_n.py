import numpy as np
import lambertw


def v_from_i(rsh, rs, nnsvth, i, io, iphi):
    """
    v_from_i calculates voltage v from current i for a single diode equation.

    Syntax
        v = v_from_i(rsh, rs, nnsvth, i, io, iphi)

    Description
        Calculate voltage V from current I using the single diode equation and parameter values Rsh, Rs, nNsVth, I, Io,
        Iphi.

    :param rsh: a numpy array. The shunt resistance (ohm)
    :param rs: a numpy array. The series resistance (ohm)
    :param nnsvth: a numpy array. product of diode factor n, number of cells in series Ns, and cell thermal voltage Vth
    :param i: a numpy array. Current at which corresponding voltage will be computed (A)
    :param io: a numpy array. dark current (A)
    :param iphi: a numpy array. light current (A)
    All numpy arrays must be of the same length
    :return:
        v - numpy array of corresponding voltages for current(s) i

    Sources:
        [1] PVLib MATLAB
        [2] A. Jain, A. Kapoor, "Exact analytical solutions of the parameters of real solar cells using Lambert
            W-function", Solar Energy Materials and Solar Cells, 81 (2004) 269 - 277.

    See also lambertw
    """

    # Generate the argument of the LambertW function
    argw = (io * rsh / nnsvth) * np.exp(rsh * (-i + iphi + io) / nnsvth)
    inputterm = lambertw.lambertw(argw)  # Get the LambertW output
    f = np.isnan(inputterm)  # If argw is too big, the LambertW result will be NaN and we have to go to logspace

    # Where f = NaN then the input argument (argW) is too large. It is necessary to go to logspace
    if any(f):
        # Calculate the log(argw) if argw is really big
        logargw = np.log(io) + np.log(rsh) + rsh * (iphi + io - i) / nnsvth - np.log(nnsvth)
        # Three iterations of Newton-Raphson method to solve w+log(w)=logargw. The initial guess is w=logargw. Where
        # direct evaluation (above) results in NaN from overflow, 3 iterations of Newton's method gives approximately
        # 8 digits of precision.
        w = logargw
        for j in range(3):
            w *= ((1. - np.log(w) + logargw) / (1. + w))
        inputterm[f] = w[f]

    # Eqn. 3 in Jain and Kapoor, 2004
    v = -i * (rs + rsh) + iphi * rsh - nnsvth * inputterm + io * rsh
    return v


def update_io_known_n(rsh, rs, nnsvth, io, il, voc):
    """
    update_io_known_n adjusts io to match voc using other parameter values.

    Syntax
        outio = update_io_known_n(rsh, rs, nnsvth, io, il, voc)

    Description
        update_io_known_n adjusts io to match voc using other parameter values, i.e., Rsh (shunt resistance), Rs
        (Series Resistance), n (diode factor), and IL (Light Current). Io is updated iteratively 10 times or until
        successive values are less than 0.000001 % different. The updating is similar to Newton's method.

    :param rsh: a numpy array of length N of values for the shunt resistance (ohm)
    :param rs: a numpy array of length N of values for the series resistance (ohm)
    :param nnsvth: a numpy array of length N of values for the diode factor x thermal voltage for the module, equal to
                   Ns (number of cells in series) x Vth (thermal voltage per cell).
    :param io: a numpy array of length N of initial values for Io (A)
    :param il: a numpy array of length N of values for lighbt current IL (A)
    :param voc: a numpy array of length N of values for Voc (V)
    :return:
        outio - a numpy array of lenght N of updated values for Io

    Sources:
        [1] PVLib MATLAB
        [2] C. Hansen, Parameter Estimation for Single Diode Models of Photovoltaic Modules, Sandia National
            Laboratories Report SAND2015-XXXX
        [3] C. Hansen, Estimation of Parameteres for Single Diode Models using Measured IV Curves, Proc. of the 39th
            IEEE PVSC, June 2013.
    """

    eps = 1e-6
    niter = 10
    k = 1
    maxerr = 1

    tio = np.transpose(io)  # Current Estimate of Io

    while maxerr > eps and k < niter:
        # Predict Voc
        pvoc = v_from_i(rsh, rs, nnsvth, 0, tio, il)

        # Difference in Voc
        dvoc = pvoc - voc

        # Update Io
        next_io = tio * (1. + (2. * dvoc) / (2. * nnsvth - dvoc))

        # Calculate Maximum Percent Difference
        maxerr = max(abs(next_io - tio) / tio) * 100
        tio = next_io
        k += 1.

    outio = tio
    return outio
