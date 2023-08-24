
import pytest
import numpy as np
import pandas as pd
from pvlib.ivtools.mismatch import prepare_curves, combine_curves


# inputs to test combine_curves
@pytest.mark.parametrize('currents, voltages, expected', [
        # different iscs and vocs
        (
            np.array([2, 1, 0.]),
            np.array([[0., 7.5, 8], [-1, 9.5, 10]]),
            {'i_sc': 1.944444444444, 'v_oc': 18, 'i_mp': 1, 'v_mp': 17,
             'p_mp': 17,
             'i': np.array([2, 1, 0.]),
             'v': np.array([-1, 17, 18])
            }
        ),

        # pandas inputs
        (
            pd.Series([2, 1, 0.]),
            pd.DataFrame([[0., 7.5, 8], [-1, 9.5, 10]]),
            {'i_sc': 1.944444444444, 'v_oc': 18, 'i_mp': 1, 'v_mp': 17,
             'p_mp': 17,
             'i': np.array([2, 1, 0.]),
             'v': np.array([-1, 17, 18])
            }
        ),

        # check that isc is actually largest current if curve doesn't
        # cross y-axis
        (
            np.array([1, 0.5, 0]),
            np.array([[0., 1.75, 2], [0.5, 1, 1.5], [0.25, 0.75, 1]]),
            {'i_sc': 1, 'v_oc': 4.5, 'i_mp': 0.5, 'v_mp': 3.5, 'p_mp': 1.75,
             'i': np.array([1, 0.5, 0.]),
             'v': np.array([0.75, 3.5, 4.5])
            }
        ),

        # same curve twice
        (
            np.array([1, 0.9, 0.]),
            np.array([[0., 0.5, 1], [0, 0.5, 1]]),
            {'i_sc': 1, 'v_oc': 2, 'i_mp': 0.9, 'v_mp': 1, 'p_mp': 0.9,
             'i': np.array([1, 0.9, 0.]),
             'v': np.array([0, 1, 2])
            }
        )
    ])
def test_combine(currents, voltages, expected):
    out = combine_curves(currents, voltages)

    # check that outputted dictionary is close to expected
    for k, v in expected.items():
        assert np.all(np.isclose(out[k], v))


def test_combine_curves_args_fail():
    # voltages are not increasing
    with pytest.raises(ValueError):
        combine_curves([1,0], [[2,1],[0.1, 1]])

    # currents don't end at zero
    with pytest.raises(ValueError):
        combine_curves([1.1,0.1], [[1,2],[3,4]])


# inputs to test prepare_curves
@pytest.mark.parametrize('num_pts, breakdown_voltage, params, expected', [
        # standard use
        # also tests vectorization of both the inputted currents and the
        # curve parameters (when calling bishop88_v_from_i)
        (
            3, -0.5,
            np.array([[1, 3e-08, 1., 300, 1.868364353685363],
                      [5, 1e-09, 0.1, 3000, 2.404825405733636]]),
            (
                np.array([4.99983334, 2.49991667, 0.]),
                np.array([[-0.5, -0.5, 3.21521313e+01],
                          [-2.52464716e-13, 5.17727056e+01, 5.36976290e+01]])
            )
        ),

        # different breakdown_voltage from default
        (
            3, -2,
            np.array([[1, 3e-08, 1., 300, 1.868364353685363],
                      [5, 1e-09, 0.1, 3000, 2.404825405733636]]),
            (
                np.array([4.99983334, 2.49991667, 0.]),
                np.array([[-2, -2, 3.21521313e+01],
                          [-2.52464716e-13, 5.17727056e+01, 5.36976290e+01]])
            )
        ),

        # pandas input
        (
            3, -0.5,
            pd.DataFrame([[1, 3e-08, 1., 300, 1.868364353685363],
                          [5, 1e-09, 0.1, 3000, 2.404825405733636]]),
            (
                np.array([4.99983334, 2.49991667, 0.]),
                np.array([[-0.5, -0.5, 3.21521313e+01],
                          [-2.52464716e-13, 5.17727056e+01, 5.36976290e+01]])
            )
        ),

        # params is just a list (no dimension)
        (
            5, -0.5,
            [1, 3e-08, 1., 300, 1.868364353685363],
            (
                np.array([0.99667772, 0.74750829, 0.49833886, 0.24916943, 0.]),
                np.array([2.42028619e-14, 2.81472975e+01, 3.01512748e+01,
                          3.12974337e+01, 3.21521313e+01])
            )
        )
    ])
def test_prepare_curves(params, num_pts, breakdown_voltage, expected):
    out = prepare_curves(params, num_pts, breakdown_voltage)

    # check that outputted currents array is close to expected
    assert np.all(np.isclose(out[0], expected[0]))

    # check that outputted voltages array is close to expected
    assert np.all(np.isclose(out[1], expected[1]))


