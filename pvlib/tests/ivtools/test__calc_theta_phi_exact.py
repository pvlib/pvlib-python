import numpy as np
import pytest
from pvlib.tests.conftest import requires_scipy
from pvlib.ivtools.sdm import _calc_theta_phi_exact

# old (imp, il, vmp, io, nnsvth, rs, rsh)
# new (vmp, imp, iph, io, rs, rsh, nnsvth)

@requires_scipy
@pytest.mark.parametrize('vmp, imp, iph, io, rs, rsh, nnsvth, expected', [
    (2., 2., 2., 2., 2., 2., 2., (1.8726, 2.)),
    (2., 0., 2., 2., 2., 2., 2., (1.8726, 3.4537)),
    (2., 2., 0., 2., 2., 2., 2., (1.2650, 0.8526)),
    (0., 2., 2., 2., 2., 2., 2., (1.5571, 2.))])
def test__calc_theta_phi_exact(vmp, imp, iph, io, rs, rsh, nnsvth, expected):
    theta, phi = _calc_theta_phi_exact(vmp, imp, iph, io, rs, rsh, nnsvth)
    np.testing.assert_allclose(theta, expected[0], atol=.0001)
    np.testing.assert_allclose(phi, expected[1], atol=.0001)


@requires_scipy
@pytest.mark.parametrize('vmp, imp, iph, io, rs, rsh, nnsvth', [
    (2., 2., 2., 0., 2., 2., 2.),
    (2., 2., 2., 2., 2., 2., 0.),
    (2., 0., 2., 2., 2., 0., 2.)])
def test__calc_theta_phi_exact_both_nan(vmp, imp, iph, io, rs, rsh, nnsvth):
    theta, phi = _calc_theta_phi_exact(vmp, imp, iph, io, rs, rsh, nnsvth)
    assert np.isnan(theta)
    assert np.isnan(phi)


@requires_scipy
def test__calc_theta_phi_exact_one_nan():
    theta, phi = _calc_theta_phi_exact(imp=2., iph=2., vmp=2., io=2.,
                                       nnsvth=2., rs=0., rsh=2.)
    assert np.isnan(theta)
    np.testing.assert_allclose(phi, 2., atol=.0001)


@requires_scipy
def test__calc_theta_phi_exact_vector():
    theta, phi = _calc_theta_phi_exact(imp=np.array([1., -1.]),
                                       iph=np.array([-1., 1.]),
                                       vmp=np.array([1., -1.]),
                                       io=np.array([-1., 1.]),
                                       nnsvth=np.array([1., -1.]),
                                       rs=np.array([-1., 1.]),
                                       rsh=np.array([1., -1.]))
    assert np.isnan(theta[0])
    assert np.isnan(theta[1])
    assert np.isnan(phi[0])
    np.testing.assert_allclose(phi[1], 2.2079, atol=.0001)
