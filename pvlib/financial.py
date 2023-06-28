import numpy as np


def lcoe(production=None, cap_cost=None, fixed_om=None):
    """
    Real levelized cost of electricity (LCOE).

    Includes cost of capital and fixed operations and maintenance (O&M).
    Described in [1]_, pp. 43 and 47-48, and defined here as
    .. math::
        \frac{\text{total capital cost} + \text{total fixed O&M costs}}
        {\text{lifetime energy production}}

    Parameters
    ----------
    production : np.array, or pd.Series, default None
        Annual production [kWh/kW installed]
    cap_cost : np.array, or pd.Series, default None
        Initial and annual payments on capital costs using real discount rates
        [$/kW installed]
    fixed_om : np.array, or pd.Series, default None
        Annual payments on operations and maintenance costs [$/kW installed]

    Returns
    ----------
    LCOE [cents/kWh]

    References
    ----------
    .. [1] W. Short, D. J. Packey, and T. Holt, "A Manual for the Economic
      Evaluation of Energy Efficiency and Renewable Energy Technologies",
      NREL/TP-462-5173, 1995.
    """

    cost = cap_cost + fixed_om
    return np.nansum(cost)*100/np.nansum(production)


def crf(rate, n_years):
    """
    Capital recovery factor.

    Described in [1]_, pp. 23.

    Parameters
    ----------
    rate : float
        Real (as opposed to nominal) rate at which CRF is calculated
    n_years: int
        Number of years over which CRF is calculated

    Returns
    ----------
    crf : float
        Real capital recovery factor

    References
    ----------
    .. [1] W. Short, D. J. Packey, and T. Holt, "A Manual for the Economic
       Evaluation of Energy Efficiency and Renewable Energy Technologies",
       NREL/TP-462-5173, 1995.
    """

    return (rate*(1+rate)**n_years)/((1+rate)**n_years-1)


def nominal_to_real(nominal, rate):

    """
    Convert nominal to real (inflation-adjusted) rate.

    Described in [1]_, pp. 6.
    Parameters
    ----------
    nominal : float
        Nominal rate (does not include adjustments for inflation)
    rate : float
        Inflation rate

    Returns
    ----------
    Real rate (includes adjustments for inflation)

    References
    ----------
    .. [1] W. Short, D. J. Packey, and T. Holt, "A Manual for the Economic
       Evaluation of Energy Efficiency and Renewable Energy Technologies",
       NREL/TP-462-5173, 1995.
    """

    return (1+nominal)/(1+rate)-1


def real_to_nominal(real, rate):

    """
    Convert real (inflation-adjusted) rate to nominal rate.

    Described by [1]_, pp. 6.

    Parameters
    ----------
    real : float
        Real rate (includes adjustments for inflation)
    rate : float
        Inflation rate

    Returns
    ----------
    Nominal rate (does not include adjustments for inflation)

    References
    ----------
    .. [1] W. Short, D. J. Packey, and T. Holt, "A Manual for the Economic
       Evaluation of Energy Efficiency and Renewable Energy Technologies",
       NREL/TP-462-5173, 1995.
    """

    return (real+1)*(1+rate)-1


def wacc(loan_frac, rroi, rint, inflation_rate, tax_rate):

    """
    Weighted average cost of capital (WACC).

    The average expected rate that is paid to finance assets [1]_.

    Parameters
    ----------
    loan_frac : float
        Fraction of capital cost paid for with a loan
    rroi : float
        Real internal rate of return on investment
    rint : float
        Real interest rate
    inflation_rate : float
        Effective inflation rate
    tax_rate : float
        Tax rate

    Returns
    ----------
    wacc : float
        Weighted average cost of capital

    References
    ----------
    .. [1] NREL, "Equations and Variables in the ATB", Available:
       https://atb.nrel.gov/electricity/2022/index
    """

    numerator = (1 + ((1 - loan_frac)*((1 + rroi)*(1 + inflation_rate)-1))
                 + loan_frac*((1 + rint)*(1 + inflation_rate)
                              - 1)*(1 - tax_rate))
    denominator = 1 + inflation_rate
    return numerator/denominator - 1
