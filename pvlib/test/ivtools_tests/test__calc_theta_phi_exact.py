import numpy as np
from pvlib.test.conftest import requires_scipy
from pvlib.ivtools.fit_sdm import _calc_phi_exact, _calc_theta_exact

# =============================================================================
# old: (imp, il, vmp, io, nnsvth, rs, rsh)
# new: (imp, il, io, rsh, nnsvth)
# new: (vmp, il, io, rsh, rs, nnsvth)
#
# =============================================================================
@requires_scipy
def test__calc_phi_exact():
    imp = np.array([5., 2., 5.])
    il = np.array([6., 2., 6.])
    io = np.array([1e-7, 2., 1e-7])
    rsh = np.array([1000., 2., 5000.])
    nnsvth = np.array([3., 2., 3.])
    exp_phi = np.array([317.2647, 2., 1650.55845])
    phi = _calc_phi_exact(imp=imp, il=il, io=io, rsh=rsh, nnsvth=nnsvth)
    assert np.allclose(exp_phi, phi)
    # test with nnsvth = 0, negative arguments
    phi = _calc_phi_exact(imp=5., il=6., io=1e-7,
                          rsh=np.array([1000., -1000.]), nnsvth=np.array([0.]))
    assert all(np.isnan(phi))
    # test each input with np.nan
    imp = np.array([np.nan, 2., 2., 2., 2.])
    il = np.array([2., np.nan, 2., 2., 2.])
    io = np.array([2., 2., np.nan, 2., 2.])
    rsh = np.array([2., 2., 2., np.nan, 2.])
    nnsvth = np.array([2., 2., 2., 2., np.nan])
    phi = _calc_phi_exact(imp=imp, il=il, io=io, rsh=rsh, nnsvth=nnsvth)
    assert all(np.isnan(phi))

@requires_scipy
def test__calc_theta_exact():
    vmp = np.array([25., 0., 5000.])
    il = np.array([6., 0., 6.])
    io = np.array([1e-7, 2., 1e-7])
    rsh = np.array([1000., 2., 1000.])
    rs = np.array([0.5, 2., 0.5])
    nnsvth = np.array([3., 2., 3.])
    exp_theta = np.array([1.874734e-4, 1., 1.6415195e+3])
    theta = _calc_theta_exact(vmp=vmp, il=il, io=io, rsh=rsh, rs=rs,
                              nnsvth=nnsvth)
    assert np.allclose(exp_theta, theta)
    # test with nnsvth = 0, negative arguments
    theta = _calc_theta_exact(vmp=25., il=6., io=1e-7,
                              rsh=np.array([1000., -1000.]), rs=0.5,
                              nnsvth=np.array([0.]))
    assert all(np.isnan(theta))
    # test each input with np.nan
    vmp = np.array([np.nan, 2., 2., 2., 2., 2.])
    il = np.array([2., np.nan, 2., 2., 2., 2.])
    io = np.array([2., 2., np.nan, 2., 2., 2.])
    rsh = np.array([2., 2., 2., np.nan, 2., 2.])
    rs = np.array([2., 2., 2., 2., np.nan, 2.])
    nnsvth = np.array([2., 2., 2., 2., 2., np.nan])
    theta = _calc_theta_exact(vmp=vmp, il=il, io=io, rsh=rsh, rs=rs,
                              nnsvth=nnsvth)
    assert all(np.isnan(theta))
