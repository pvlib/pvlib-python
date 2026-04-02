import pandas as pd
import pytest

from numpy.testing import assert_allclose

from pvlib import transformer

import numpy as np


def test_simple_efficiency():

    # define test inputs
    input_power = pd.Series([
        -800.0,
        436016.609823837,
        1511820.16603752,
        1580687.44677249,
        1616441.79660171
    ])
    no_load_loss = 0.002
    load_loss = 0.007
    transformer_rating = 2750000

    # define expected test results
    expected_output_power = pd.Series([
        -6300.10103234071,
        430045.854892526,
        1500588.39919874,
        1568921.77089526,
        1604389.62839879
    ])

    # run test function with test inputs
    calculated_output_power = transformer.simple_efficiency(
        input_power=input_power,
        no_load_loss=no_load_loss,
        load_loss=load_loss,
        transformer_rating=transformer_rating
    )

    # determine if expected results are obtained
    assert_allclose(calculated_output_power, expected_output_power)


@pytest.mark.parametrize(
    "input_power, no_load_loss, load_loss, rating, expected",
    [
        # no-load condition
        (0.005 * 1000, 0.005, 0.01, 1000, 0.0),

        # rated condition
        (1000 * (1 + 0.005 + 0.01), 0.005, 0.01, 1000, 1000),

        # zero load_loss case
        # for load_loss = 0, the model reduces to:
        # P_out = P_in - L_no_load * P_nom
        (1000.0, 0.01, 0.0, 1000.0, 990.0),
    ],
)
def test_simple_efficiency_numeric_cases(
    input_power, no_load_loss, load_loss, rating, expected
):
    result = transformer.simple_efficiency(
        input_power=input_power,
        no_load_loss=no_load_loss,
        load_loss=load_loss,
        transformer_rating=rating,
    )

    assert_allclose(result, expected)


def test_simple_efficiency_vector_equals_scalar():
    input_power = np.array([200.0, 600.0, 900.0])
    no_load_loss = 0.005
    load_loss = 0.01
    rating = 1000.0

    vector_result = transformer.simple_efficiency(
        input_power=input_power,
        no_load_loss=no_load_loss,
        load_loss=load_loss,
        transformer_rating=rating,
    )

    scalar_result = np.array([
        transformer.simple_efficiency(p, no_load_loss, load_loss, rating)
        for p in input_power
    ])

    assert_allclose(vector_result, scalar_result)
