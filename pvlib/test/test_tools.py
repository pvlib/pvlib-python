import numpy as np
import pandas as pd
import pytest

import pvlib.tools
from pvlib import tools
from pvlib.test.test_solarposition import times


@pytest.mark.parametrize('keys, input_dict, expected', [
    (['a', 'b'], {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2}),
    (['a', 'b', 'd'], {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2}),
    (['a'], {}, {}),
    (['a'], {'b': 2}, {})
])
def test_build_kwargs(keys, input_dict, expected):
    kwargs = tools._build_kwargs(keys, input_dict)
    assert kwargs == expected


def test_datetime_to_julian():
    """ test transformation from datetime to julians """
    julians = pvlib.tools.datetime_to_julian(pd.to_datetime(times))
    np.testing.assert_array_almost_equal(np.array(julians[:10]),
                                         np.array([
                                             2456832.5,
                                             2456832.5104166665,
                                             2456832.5208333335,
                                             2456832.53125,
                                             2456832.5416666665,
                                             2456832.5520833335,
                                             2456832.5625,
                                             2456832.5729166665,
                                             2456832.5833333335,
                                             2456832.59375])
                                         )
