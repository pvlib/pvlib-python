import pandas as pd

from numpy.testing import assert_allclose

from pvlib import transformer


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


def test_simple_efficiency_known_values():
    no_load_loss = 0.005
    load_loss = 0.01
    rating = 1000
    args = (no_load_loss, load_loss, rating)

    # verify correct behavior at no-load condition
    assert_allclose(
        transformer.simple_efficiency(no_load_loss*rating, *args),
        0.0
    )

    # verify correct behavior at rated condition
    assert_allclose(
        transformer.simple_efficiency(rating*(1 + no_load_loss + load_loss),
                                      *args),
        rating,
    )
