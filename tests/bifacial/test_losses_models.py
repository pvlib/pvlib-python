from pvlib import bifacial

import pandas as pd
import numpy as np
from numpy.testing import assert_allclose


def test_power_mismatch_deline():
    """tests bifacial.power_mismatch_deline"""
    premise_rmads = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
    # test default model is for fixed tilt
    expected_ft_mms = np.array([0.0, 0.0151, 0.0462, 0.0933, 0.1564, 0.2355])
    result_def_mms = bifacial.power_mismatch_deline(premise_rmads)
    assert_allclose(result_def_mms, expected_ft_mms, atol=1e-5)
    assert np.all(np.diff(result_def_mms) > 0)  # higher RMADs => higher losses

    # test custom coefficients, set model to 1+1*RMAD
    # as Polynomial class
    polynomial = np.polynomial.Polynomial([1, 1, 0])
    result_custom_mms = bifacial.power_mismatch_deline(
        premise_rmads, coefficients=polynomial
    )
    assert_allclose(result_custom_mms, 1 + premise_rmads)
    # as list
    result_custom_mms = bifacial.power_mismatch_deline(
        premise_rmads, coefficients=[1, 1, 0]
    )
    assert_allclose(result_custom_mms, 1 + premise_rmads)

    # test datatypes IO with Series
    result_mms = bifacial.power_mismatch_deline(pd.Series(premise_rmads))
    assert isinstance(result_mms, pd.Series)

    # test fill_factor, fill_factor_reference
    # default model + default fill_factor_reference
    ff_ref_default = 0.79
    ff_of_interest = 0.65
    result_mms = bifacial.power_mismatch_deline(
        premise_rmads, fill_factor=ff_of_interest
    )
    assert_allclose(
        result_mms,
        expected_ft_mms * ff_of_interest / ff_ref_default,
        atol=1e-5,
    )
    # default model + custom fill_factor_reference
    ff_of_interest = 0.65
    ff_ref = 0.75
    result_mms = bifacial.power_mismatch_deline(
        premise_rmads, fill_factor=ff_of_interest, fill_factor_reference=ff_ref
    )
    assert_allclose(
        result_mms, expected_ft_mms * ff_of_interest / ff_ref, atol=1e-5
    )
