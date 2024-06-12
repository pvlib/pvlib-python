from pvlib import bifacial

import pandas as pd
import numpy as np
from numpy.testing import assert_allclose

import pytest


def test_power_mismatch_deline():
    """tests bifacial.power_mismatch_deline"""
    premise_rmads = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
    # default model result values
    expected_sat_mms = np.array(
        [0.0, 0.00287, 0.00608, 0.00963, 0.01352, 0.01775]
    )
    result_def_mms = bifacial.power_mismatch_deline(premise_rmads)
    assert_allclose(result_def_mms, expected_sat_mms, atol=1e-5)
    assert np.all(np.diff(result_def_mms) > 0)  # higher RMADs => higher losses
    # default model matches single-axis tracker
    result_sat_mms = bifacial.power_mismatch_deline(
        premise_rmads, model="single-axis-tracking"
    )
    assert_allclose(result_sat_mms, expected_sat_mms)
    # fixed-tilt model result values
    expected_ft_mms = np.array(
        [0.0, 0.00718, 0.01452, 0.02202, 0.02968, 0.0375]
    )
    result_ft_mms = bifacial.power_mismatch_deline(
        premise_rmads, model="fixed-tilt"
    )
    assert_allclose(result_ft_mms, expected_ft_mms)
    assert np.all(np.diff(result_ft_mms) > 0)  # higher RMADs => higher losses
    # test custom coefficients, set model to 1+1*RMAD
    # as Polynomial class
    polynomial = np.polynomial.Polynomial([1, 1, 0])
    result_custom_mms = bifacial.power_mismatch_deline(
        premise_rmads, model=polynomial
    )
    assert_allclose(result_custom_mms, 1 + premise_rmads)
    # as list
    result_custom_mms = bifacial.power_mismatch_deline(
        premise_rmads, model=[1, 1, 0]
    )
    assert_allclose(result_custom_mms, 1 + premise_rmads)

    # test datatypes IO with Series
    result_mms = bifacial.power_mismatch_deline(pd.Series(premise_rmads))
    assert isinstance(result_mms, pd.Series)

    # test fillfactor
    # with an internal model
    result_mms = bifacial.power_mismatch_deline(
        premise_rmads, fillfactor=0.65
    )
    assert_allclose(result_mms, expected_sat_mms * 0.65 / 0.79, atol=1e-5)
    # fails for a custom polynomial
    with pytest.raises(ValueError, match="Fill factor can only be used"):
        bifacial.power_mismatch_deline(
            premise_rmads, model=polynomial, fillfactor=0.24
        )

    # test raises error on inexistent model
    with pytest.raises(ValueError, match="Invalid model 'foo'"):
        bifacial.power_mismatch_deline(premise_rmads, model="foo")