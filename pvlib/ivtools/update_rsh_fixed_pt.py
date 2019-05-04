from pvlib.ivtools.calc_theta_phi_exact import calc_theta_phi_exact


def update_rsh_fixed_pt(rsh, rs, io, il, nnsvth, imp, vmp):
    """
    UPDATE_RSH_FIXED_PT adjusts Rsh to match Vmp using other paramter values

    Syntax
    ------
    outrsh = update_rsh_fixed_pt(rsh, rs, io, il, nnsvth, imp, vmp)

    Description
    -----------
    update_rsh_fixed_pt adjusts rsh to match vmp using other parameter values,
    i.e., Rs (series resistance), n (diode factor), Io (dark current), and IL
    (light current). Rsh is updated iteratively using a fixed point expression
    obtained from combining Vmp = Vmp(Imp) (using the analytic solution to the
    single diode equation) and dP / dI = 0 at Imp. 500 iterations are performed
    because convergence can be very slow.

    Parameters
    ----------
    rsh: a numpy array of length N of initial values for shunt resistance (ohm)
    rs: a numpy array of length N of values for series resistance (ohm)
    io: a numpy array of length N of values for Io (A)
    il: a numpy array of length N of values for light current IL (A)
    nnsvth: a numpy array length N of values for the diode factor x thermal
            voltage for the module, equal to Ns (number of cells in series) x
            Vth (thermal voltage per cell).
    imp: a numpy array of length N of values for Imp (A)
    vmp: a numpy array of length N of values for Vmp (V)

    Returns
    -------
    outrsh: a numpy array of length N of updated values for Rsh

    References
    ----------
    [1] PVL MATLAB
    [2] C. Hansen, Parameter Estimation for Single Diode Models of Photovoltaic
        Modules, Sandia National Laboratories Report SAND2015-XXXX
    """
    niter = 500
    x1 = rsh

    for i in range(niter):
        y, z = calc_theta_phi_exact(imp, il, vmp, io, nnsvth, rs, x1)
        next_x1 = (1 + z) / z * ((il + io) * x1 / imp - nnsvth * z / imp - 2 *
                                 vmp / imp)
        x1 = next_x1

    outrsh = x1
    return outrsh
