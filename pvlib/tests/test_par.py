"""
Test Photosynthetically Active Radiation (PAR) submodule.
"""

from pvlib import par

import numpy as np
from numpy.testing import assert_allclose


def test_spitters_relationship():
    solar_zenith, global_diffuse_fraction = np.meshgrid(
        [90, 85, 75, 60, 40, 30, 10, 0], [0.01, 0.1, 0.3, 0.6, 0.8, 0.99]
    )
    solar_zenith = solar_zenith.ravel()
    global_diffuse_fraction = global_diffuse_fraction.ravel()
    result = par.spitters_relationship(solar_zenith, global_diffuse_fraction)
    expected = np.array([
        0.00650018, 0.00656213, 0.00706211, 0.0087417 , 0.01171437, 0.01260581,
        0.01299765, 0.0129997 , 0.06517588, 0.06579393, 0.07077986, 0.08750105,
        0.11699064, 0.12580782, 0.12967973, 0.12970000, 0.19994764, 0.20176275,
        0.21635259, 0.26460255, 0.34722693, 0.37134002, 0.38184514, 0.38190000,
        0.43609756, 0.43933488, 0.46497584, 0.54521789, 0.66826809, 0.70117647,
        0.71512774, 0.71520000, 0.65176471, 0.65503875, 0.68042968, 0.75414541,
        0.85271445, 0.87653894, 0.88634962, 0.88640000, 0.97647838, 0.97683827,
        0.97952006, 0.98634857, 0.99374028, 0.99529135, 0.99590717, 0.9959103
    ])
    assert_allclose(result, expected, atol=1e-8)
