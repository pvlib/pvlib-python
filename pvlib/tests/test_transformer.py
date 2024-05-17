import numpy as np
import pandas as pd

from numpy.testing import assert_allclose

from pvlib import transformer



def test_simple_transformer():

    # define test inputs
    input_power = pd.Series([
        -800.0,
        436016.609823837,
        1511820.16603752,
        1580687.44677249,
        1616441.79660171
        ])
    no_load_loss_fraction = 0.002
    load_loss_fraction = 0.007
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
        no_load_loss_fraction=no_load_loss_fraction,
        load_loss_fraction=load_loss_fraction,
        transformer_rating=transformer_rating
        )
    
    # determine if expected results are obtained
    assert_allclose(calculated_output_power, expected_output_power)