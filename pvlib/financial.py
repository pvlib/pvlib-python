import numpy as np

def lcoe(production=None, cap_cost=None, fixed_om=None):
        """
        Levelized cost of electricity as described on pp.43 and 47-48 by [1]   
        Parameters
        ----------
        production : np.array, or pd.Series, default None
            Annual production [kWh/kW installed]
        cap_cost : np.array, or pd.Series, default None
            Initial and annual payments on capital costs [$/kW installed]
        fixed_om : np.array, or pd.Series, default None
            Annual payments on operations and maintenance costs [$/kW installed]
            
        Returns
        ----------
        LCOE [cents/kWh]

        References
        ----------
        .. [1] W. Short, D. J. Packey, and T. Holt, "A Manual for the Economic Evaluation of Energy Efficiency and Renewable Energy Technologies", NREL/TP-462-5173, 1995.
        """        
        if len(production)!=len(cap_cost) or len(cap_cost)!=len(fixed_om):
            raise ValueError("Unequal input array lengths")      
        cost = cap_cost + fixed_om            
        return np.round(np.nansum(cost)*100/np.nansum(production),2)

def crf(rate, n_years):
    """
    Capital recovery factor as described on pp. 23 by [1]    
    Parameters
    ----------
    rate : float
        Rate at which CRF is calculated
    n_years: int
        Number of years over which CRF is calculated

    Returns
    ----------
    crf : float
        Capital recovery factor
    References
    ----------
    .. [1] W. Short, D. J. Packey, and T. Holt, "A Manual for the Economic Evaluation of Energy Efficiency and Renewable Energy Technologies", NREL/TP-462-5173, 1995.
    """ 
    return np.round((rate*(1+rate)**n_years)/((1+rate)**n_years-1),8)

def nominal_to_real(nominal, rate):
    """
    Inflation-adjusted rate described on pp. 6 by [1]
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
    .. [1] W. Short, D. J. Packey, and T. Holt, "A Manual for the Economic Evaluation of Energy Efficiency and Renewable Energy Technologies", NREL/TP-462-5173, 1995.
    """ 
    return np.round((1+nominal)/(1+rate)-1, 8)

def real_to_nominal(real, rate):
    """
    Rate without adjusting for inflation as described on pp. 6 by [1]
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
    .. [1] W. Short, D. J. Packey, and T. Holt, "A Manual for the Economic Evaluation of Energy Efficiency and Renewable Energy Technologies", NREL/TP-462-5173, 1995.
    """ 

    return np.round((real+1)*(1+rate)-1, 8)

def wacc(loan_frac, rroi, rint, inflation_rate, tax_rate):
    """        
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
    .. [1] S. Blumsack, "Weighted Average Cost of Capital", The Pennsylvania State University, Available: http://www.e-education.psu.edu/eme801/node/585
    """ 
    numerator = (1 + ((1 - loan_frac)*((1 + rroi)*(1 + inflation_rate)-1))
                + loan_frac*((1 + rint)*(1 + inflation_rate) - 1)*(1 - tax_rate))
    denominator = 1 + inflation_rate
    return np.round(numerator/denominator -1, 8)