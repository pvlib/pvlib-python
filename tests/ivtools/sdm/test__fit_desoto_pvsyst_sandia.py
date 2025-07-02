import numpy as np

import pytest
from numpy.testing import assert_allclose

from pvlib.ivtools.sdm._fit_desoto_pvsyst_sandia import (
    _calc_theta_phi_exact,
    _update_rsh_fixed_pt,
    _update_io
)


@pytest.mark.parametrize('vmp, imp, iph, io, rs, rsh, nnsvth, expected', [
    (2., 2., 2., 2., 2., 2., 2., np.nan),
    (2., 2., 0., 2., 2., 2., 2., np.nan),
    (2., 2., 2., 0., 2., 2., 2., np.nan),
    (2., 2., 2., 2., 0., 2., 2., np.nan),
    (2., 2., 2., 2., 2., 0., 2., np.nan),
    (2., 2., 2., 2., 2., 2., 0., np.nan)])
def test__update_rsh_fixed_pt_nans(vmp, imp, iph, io, rs, rsh, nnsvth,
                                   expected):
    outrsh = _update_rsh_fixed_pt(vmp, imp, iph, io, rs, rsh, nnsvth)
    assert np.all(np.isnan(outrsh))


def test__update_rsh_fixed_pt_vmp0():
    outrsh = _update_rsh_fixed_pt(vmp=0., imp=2., iph=2., io=2., rs=2.,
                                  rsh=2., nnsvth=2.)
    assert_allclose(outrsh, np.array([502.]), atol=.0001)


def test__update_rsh_fixed_pt_vector():
    outrsh = _update_rsh_fixed_pt(rsh=np.array([-1., 3, .5, 2.]),
                                  rs=np.array([1., -.5, 2., 2.]),
                                  io=np.array([.2, .3, -.4, 2.]),
                                  iph=np.array([-.1, 1, 3., 2.]),
                                  nnsvth=np.array([4., -.2, .1, 2.]),
                                  imp=np.array([.2, .2, -1., 2.]),
                                  vmp=np.array([0., -1, 0., 0.]))
    assert np.all(np.isnan(outrsh[0:3]))
    assert_allclose(outrsh[3], np.array([502.]), atol=.0001)


@pytest.mark.parametrize('voc, iph, io, rs, rsh, nnsvth, expected', [
    (2., 2., 2., 2., 2., 2., 0.5911),
    (2., 2., 2., 0., 2., 2., 0.5911),
    (2., 2., 0., 2., 2., 2., 0.),
    (2., 0., 2., 2., 2., 2., 1.0161e-4),
    (0., 2., 2., 2., 2., 2., 17.9436)])
def test__update_io(voc, iph, io, rs, rsh, nnsvth, expected):
    outio = _update_io(voc, iph, io, rs, rsh, nnsvth)
    assert_allclose(outio, expected, atol=.0001)


@pytest.mark.parametrize('voc, iph, io, rs, rsh, nnsvth', [
    (2., 2., 2., 2., 2., 0.),
    (-1., -1., -1., -1., -1., -1.)])
def test__update_io_nan(voc, iph, io, rs, rsh, nnsvth):
    with np.errstate(invalid='ignore', divide='ignore'):
        outio = _update_io(voc, iph, io, rs, rsh, nnsvth)
    assert np.isnan(outio)


@pytest.mark.parametrize('vmp, imp, iph, io, rs, rsh, nnsvth, expected', [
    (2., 2., 2., 2., 2., 2., 2., (1.8726, 2.)),
    (2., 0., 2., 2., 2., 2., 2., (1.8726, 3.4537)),
    (2., 2., 0., 2., 2., 2., 2., (1.2650, 0.8526)),
    (0., 2., 2., 2., 2., 2., 2., (1.5571, 2.))])
def test__calc_theta_phi_exact(vmp, imp, iph, io, rs, rsh, nnsvth, expected):
    theta, phi = _calc_theta_phi_exact(vmp, imp, iph, io, rs, rsh, nnsvth)
    assert_allclose(theta, expected[0], atol=.0001)
    assert_allclose(phi, expected[1], atol=.0001)


@pytest.mark.parametrize('vmp, imp, iph, io, rs, rsh, nnsvth', [
    (2., 2., 2., 0., 2., 2., 2.),
    (2., 2., 2., 2., 2., 2., 0.),
    (2., 0., 2., 2., 2., 0., 2.)])
def test__calc_theta_phi_exact_both_nan(vmp, imp, iph, io, rs, rsh, nnsvth):
    theta, phi = _calc_theta_phi_exact(vmp, imp, iph, io, rs, rsh, nnsvth)
    assert np.isnan(theta)
    assert np.isnan(phi)


def test__calc_theta_phi_exact_one_nan():
    theta, phi = _calc_theta_phi_exact(imp=2., iph=2., vmp=2., io=2.,
                                       nnsvth=2., rs=0., rsh=2.)
    assert np.isnan(theta)
    assert_allclose(phi, 2., atol=.0001)


def test__calc_theta_phi_exact_vector():
    with np.errstate(invalid='ignore'):
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
    assert_allclose(phi[1], 2.2079, atol=.0001)
