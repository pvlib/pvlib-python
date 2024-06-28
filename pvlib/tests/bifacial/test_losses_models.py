from pvlib import bifacial

import pandas as pd
import numpy as np
from numpy.testing import assert_allclose


def test_power_mismatch_deline():
    """tests bifacial.power_mismatch_deline"""
    premise_rmads = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
    # test default model is for fixed tilt
    expected_ft_mms = np.array(
        [0.0, 0.00718, 0.01452, 0.02202, 0.02968, 0.0375]
    )
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
    result_mms = bifacial.power_mismatch_deline(
        premise_rmads, fill_factor=0.65
    )
    assert_allclose(result_mms, expected_ft_mms * 0.65 / 0.79, atol=1e-5)
    # default model + custom fill_factor_reference
    ff_ref = 0.75
    result_mms = bifacial.power_mismatch_deline(
        premise_rmads, fill_factor=0.65, fill_factor_reference=ff_ref
    )
    assert_allclose(result_mms, expected_ft_mms * 0.65 / ff_ref, atol=1e-5)
