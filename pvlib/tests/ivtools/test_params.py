import numpy as np
import pytest
from pvlib.ivtools import params


@pytest.fixture
def i_array():
    i = np.array([8.09403993, 8.09382549, 8.09361103, 8.09339656, 8.09318205,
                  8.09296748, 8.09275275, 8.09253771, 8.09232204, 8.09210506,
                  8.09188538, 8.09166014, 8.09142342, 8.09116305, 8.09085392,
                  8.09044425, 8.08982734, 8.08878333, 8.08685945, 8.08312463,
                  8.07566926, 8.06059856, 8.03005836, 7.96856869, 7.8469714,
                  7.61489584, 7.19789314, 6.51138396, 5.49373476, 4.13267172,
                  2.46021487, 0.52838624, -1.61055289])
    return i


@pytest.fixture
def v_array():
    v = np.array([-0.005, 0.015, 0.035, 0.055, 0.075, 0.095, 0.115, 0.135,
                  0.155, 0.175, 0.195, 0.215, 0.235, 0.255, 0.275, 0.295,
                  0.315, 0.335, 0.355, 0.375, 0.395, 0.415, 0.435, 0.455,
                  0.475, 0.495, 0.515, 0.535, 0.555, 0.575, 0.595, 0.615,
                  0.635])
    return v


def test_astm_e1036(v_array, i_array):
    result = params.astm_e1036(v_array, i_array)
    expected = {'voc': 0.6195097477985162,
                'isc': 8.093986320386227,
                'vmp': 0.494283417170082,
                'imp': 7.626088301548568,
                'pmp': 3.7694489853302127,
                'ff': 0.7517393078504361}
    fit = result.pop('mp_fit')
    expected_fit = np.array(
        [3.6260726, 0.49124176, -0.24644747, -0.26442383, -0.1223237])
    assert fit.coef == pytest.approx(expected_fit)
    assert result == pytest.approx(expected)


def test_astm_e1036_est_isc_voc(v_array, i_array):
    '''
    Test the case in which Isc and Voc estimates are
    valid without a linear fit
    '''
    v = v_array
    i = i_array
    v = np.append(v, [0.001, 0.6201])
    i = np.append(i, [8.09397560e+00, 7.10653445e-04])
    result = params.astm_e1036(v, i)
    expected = {'voc': 0.6201,
                'isc': 8.093975598317805,
                'vmp': 0.494283417170082,
                'imp': 7.626088301548568,
                'pmp': 3.7694489853302127,
                'ff': 0.751024747526615}
    result.pop('mp_fit')
    assert result == pytest.approx(expected)


def test_astm_e1036_mpfit_limits(v_array, i_array):
    result = params.astm_e1036(v_array,
                               i_array,
                               imax_limits=(0.85, 1.1),
                               vmax_limits=(0.85, 1.1))
    expected = {'voc': 0.6195097477985162,
                'isc': 8.093986320386227,
                'vmp': 0.49464214190725303,
                'imp': 7.620032530519718,
                'pmp': 3.769189212299219,
                'ff': 0.7516875014460312}
    result.pop('mp_fit')
    assert result == pytest.approx(expected)


def test_astm_e1036_fit_points(v_array, i_array):
    i = i_array
    i[3] = 8.1  # ensure an interesting change happens
    result = params.astm_e1036(v_array, i, voc_points=4, isc_points=4)
    expected = {'voc': 0.619337073271274,
                'isc': 8.093160893325297,
                'vmp': 0.494283417170082,
                'imp': 7.626088301548568,
                'pmp': 3.7694489853302127,
                'ff': 0.7520255886236707}
    result.pop('mp_fit')
    assert result == pytest.approx(expected)
