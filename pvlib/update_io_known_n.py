import numpy as np
from pvlib.pvsystem import v_from_i


def update_io_known_n(rsh, rs, nnsvth, io, il, voc):
    """
    update_io_known_n adjusts io to match voc using other parameter values.

    Syntax
    ------
    outio = update_io_known_n(rsh, rs, nnsvth, io, il, voc)

    Description
    -----------
    update_io_known_n adjusts io to match voc using other parameter values,
    i.e., Rsh (shunt resistance), Rs (Series Resistance), n (diode factor), and
    IL (Light Current). Io is updated iteratively 10 times or until successive
    values are less than 0.000001 % different. The updating is similar to
    Newton's method.

    Parameters
    ----------
    rsh: a numpy array of length N of values for the shunt resistance (ohm)
    rs: a numpy array of length N of values for the series resistance (ohm)
    nnsvth: a numpy array of length N of values for the diode factor x thermal
            voltage for the module, equal to Ns (number of cells in series) x
            Vth (thermal voltage per cell).
    io: a numpy array of length N of initial values for Io (A)
    il: a numpy array of length N of values for lighbt current IL (A)
    voc: a numpy array of length N of values for Voc (V)

    Returns
    -------
    outio - a numpy array of lenght N of updated values for Io

    References
    ----------
    [1] PVLib MATLAB
    [2] C. Hansen, Parameter Estimation for Single Diode Models of Photovoltaic
        Modules, Sandia National Laboratories Report SAND2015-XXXX
    [3] C. Hansen, Estimation of Parameteres for Single Diode Models using
        Measured IV Curves, Proc. of the 39th IEEE PVSC, June 2013.
    """

    eps = 1e-6
    niter = 10
    k = 1
    maxerr = 1

    tio = io  # Current Estimate of Io

    while maxerr > eps and k < niter:
        # Predict Voc
        pvoc = v_from_i(rsh, rs, nnsvth, 0., tio, il)

        # Difference in Voc
        dvoc = pvoc - voc

        # Update Io
        next_io = tio * (1. + (2. * dvoc) / (2. * nnsvth - dvoc))

        # Calculate Maximum Percent Difference
        maxerr = np.max(np.abs(next_io - tio) / tio) * 100.
        tio = next_io
        k += 1.

    outio = tio
    return outio
