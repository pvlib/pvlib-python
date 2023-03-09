import pandas as pd
import numpy as np
from pvlib import financial
from numpy.testing import assert_allclose


def test_lcoe_series():
    n, cf = 20, 0.5
    production = pd.Series(data=[cf*8670 for j in range(n)])
    capex, loan_frac, my_crf, debt_tenor = 1000, 0.5, 0.05, n
    cap_cost = pd.Series(data=[capex*loan_frac*my_crf for i in
                               range(debt_tenor)]
                         + [0 for j in range(debt_tenor, n)])
    base_om = 25
    fixed_om = pd.Series(data=[base_om for j in range(n)])

    expected = 1.1534025374855825
    out = financial.lcoe(production=production, cap_cost=cap_cost,
                         fixed_om=fixed_om)

    assert_allclose(expected, out)


def test_lcoe_arrays():
    n, cf = 20, 0.5
    production = np.full(n, cf*8670)
    capex, loan_frac, my_crf, debt_tenor = 1000, 0.5, 0.05, n
    cap_cost = np.array([capex*loan_frac*my_crf for i in range(debt_tenor)]
                        + [0 for j in range(debt_tenor, n)])
    base_om = 25
    fixed_om = np.full(n, base_om)
    expected = 1.1534025374855825
    out = financial.lcoe(production=production, cap_cost=cap_cost,
                         fixed_om=fixed_om)
    assert_allclose(expected, out)


def test_lcoe_nans():
    n, cf = 20, 0.5
    production = np.full(n, cf*8670)
    capex, loan_frac, my_crf, debt_tenor = 1000, 0.5, 0.05, n
    cap_cost = np.array([capex*loan_frac*my_crf for i in range(debt_tenor)]
                        + [0 for j in range(debt_tenor, n)])
    base_om = 25.
    fixed_om = np.full(n, base_om)
    cap_cost[1] = np.nan
    fixed_om[2] = np.nan
    production[3] = np.nan
    expected = 1.092697140775815
    out = financial.lcoe(production=production, cap_cost=cap_cost,
                         fixed_om=fixed_om)
    assert_allclose(expected, out)


def test_crf():
    rate, n = 0.05, 20
    expected = 0.08024258719069129
    out = financial.crf(rate, n)
    assert_allclose(expected, out)


def test_nominal_to_real():
    nominal, rate = 0.04, 0.025
    expected = 0.014634146341463428
    out = financial.nominal_to_real(nominal, rate)
    assert_allclose(expected, out)


def test_real_to_nominal():
    real, rate = 0.04, 0.025
    expected = 0.066
    out = financial.real_to_nominal(real, rate)
    assert_allclose(expected, out)


def test_wacc():
    loan_frac = 0.5
    rroi = 0.12
    rint = 0.04
    inflation_rate = 0.025
    tax_rate = 0.28
    expected = 0.07098536585365856
    out = financial.wacc(loan_frac, rroi, rint, inflation_rate, tax_rate)
    assert_allclose(expected, out)
