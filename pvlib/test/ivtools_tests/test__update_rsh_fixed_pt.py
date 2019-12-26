import numpy as np
from pvlib.test.conftest import requires_scipy
from pvlib.ivtools.fit_sdm import _update_rsh_fixed_pt


@requires_scipy
def test__update_rsh_fixed_pt_nan():
    il = np.array([2., 0., 2., 2., 2., 2.])
    io = np.array([2., 2., 0., 2., 2., 2.])
    rsh = np.array([2., 2., 2., 0., 2., 2.])
    rs = np.array([2., 2., 2., 2., 0., 2.])
    nnsvth = np.array([2., 2., 2., 2., 0., 2.])
    vmp = np.array([2., 2., 2., 2., 2., 2.])
    imp = np.array([2., 2., 2., 2., 2., 0.])
    outrsh = _update_rsh_fixed_pt(il=il, io=io, rsh=rsh, rs=rs, nnsvth=nnsvth,
                                  vmp=vmp, imp=imp)
    assert all(np.isnan(outrsh))


@requires_scipy
def test__update_rsh_fixed_pt():
    outrsh = _update_rsh_fixed_pt(il=np.array([2.]), io=np.array([2.]),
                                  rsh=np.array([2.]), rs=np.array([2.]),
                                  nnsvth=np.array([2.]), vmp=np.array([0.]),
                                  imp=np.array([2.]))
    np.testing.assert_allclose(outrsh, np.array([502]), atol=.0001)


@requires_scipy
def test_answer8():
    outrsh = _update_rsh_fixed_pt(il=np.array([-.1, 1, 3]),
                                  io=np.array([.2, .3, -.4]),
                                  rsh=np.array([-1., 3, .5]),
                                  rs=np.array([1., -.5, 2.]),
                                  nnsvth=np.array([4., -.2, .1]),
                                  vmp=np.array([0., -1, 0.]),
                                  imp=np.array([.2, .2, -1]))
    assert all(np.isnan(outrsh))
