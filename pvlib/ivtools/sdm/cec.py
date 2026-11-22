
def fit_cec_sam(celltype, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc,
                gamma_pmp, cells_in_series, temp_ref=25):
    """
    Estimates parameters for the CEC single diode model (SDM) using the SAM
    SDK.

    Parameters
    ----------
    celltype : str
        Value is one of 'monoSi', 'multiSi', 'polySi', 'cis', 'cigs', 'cdte',
        'amorphous'
    v_mp : float
        Voltage at maximum power point [V]
    i_mp : float
        Current at maximum power point [A]
    v_oc : float
        Open circuit voltage [V]
    i_sc : float
        Short circuit current [A]
    alpha_sc : float
        Temperature coefficient of short circuit current [A/C]
    beta_voc : float
        Temperature coefficient of open circuit voltage [V/C]
    gamma_pmp : float
        Temperature coefficient of power at maximum power point [%/C]
    cells_in_series : int
        Number of cells in series
    temp_ref : float, default 25
        Reference temperature condition, [°C]

    Returns
    -------
    I_L_ref : float
        The light-generated current (or photocurrent) at reference
        conditions [A]
    I_o_ref : float
        The dark or diode reverse saturation current at reference
        conditions [A]
    R_s : float
        The series resistance at reference conditions, in ohms.
    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.
    a_ref : float
        The product of the usual diode ideality factor ``n`` (unitless),
        number of cells in series ``Ns``, and cell thermal voltage at
        reference conditions [V]
    Adjust : float
        The adjustment to the temperature coefficient for short circuit
        current, in percent.

    Raises
    ------
    ImportError
        if NREL-PySAM is not installed.
    RuntimeError
        if parameter extraction is not successful.

    Notes
    -----
    The CEC model and estimation method  are described in [1]_.
    Inputs ``v_mp``, ``i_mp``, ``v_oc`` and ``i_sc`` are assumed to be from a
    single IV curve at constant irradiance and cell temperature. Irradiance is
    not explicitly used by the fitting procedure. The irradiance level at which
    the input IV curve is determined and the specified cell temperature
    ``temp_ref`` are the reference conditions for the output parameters
    ``I_L_ref``, ``I_o_ref``, ``R_s``, ``R_sh_ref``, ``a_ref`` and ``Adjust``.

    References
    ----------
    .. [1] A. Dobos, "An Improved Coefficient Calculator for the California
       Energy Commission 6 Parameter Photovoltaic Module Model", Journal of
       Solar Energy Engineering, vol 134, 2012. :doi:`10.1115/1.4005759`
    """

    try:
        from PySAM import PySSC
    except ImportError:
        raise ImportError("Requires NREL's PySAM package at "
                          "https://pypi.org/project/NREL-PySAM/.")

    datadict = {'tech_model': '6parsolve', 'financial_model': None,
                'celltype': celltype, 'Vmp': v_mp,
                'Imp': i_mp, 'Voc': v_oc, 'Isc': i_sc, 'alpha_isc': alpha_sc,
                'beta_voc': beta_voc, 'gamma_pmp': gamma_pmp,
                'Nser': cells_in_series, 'Tref': temp_ref}

    result = PySSC.ssc_sim_from_dict(datadict)
    if result['cmod_success'] == 1:
        return tuple([result[k] for k in ['Il', 'Io', 'Rs', 'Rsh', 'a',
                      'Adj']])
    else:
        raise RuntimeError('Parameter estimation failed')
