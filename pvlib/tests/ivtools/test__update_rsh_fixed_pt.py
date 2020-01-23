import numpy as np
import pytest

from pvlib.tests.conftest import requires_scipy
from pvlib.ivtools.sdm import _update_rsh_fixed_pt


@requires_scipy
@pytest.mark.parametrize('vmp, imp, iph, io, rs, rsh, nnsvth, expected', [
    (2., 2., 2., 2., 2., 2., 2., np.nan),
    (2., 0., 2., 2., 2., 2., 2., np.nan),
    (2., 2., 0., 2., 2., 2., 2., np.nan),
    (2., 2., 2., 0., 2., 2., 2., np.nan),
    (2., 2., 2., 2., 0., 2., 2., np.nan),
    (2., 2., 2., 2., 2., 0., 2., np.nan),
    (2., 2., 2., 2., 2., 2., 0., np.nan)])
def test__update_rsh_fixed_pt_nans(vmp, imp, iph, io, rs, rsh, nnsvth,
                                   expected):
    outrsh = _update_rsh_fixed_pt(vmp, imp, iph, io, rs, rsh, nnsvth)
    assert np.isnan(outrsh)

@requires_scipy
def test__update_rsh_fixed_pt_vmp0():
    outrsh = _update_rsh_fixed_pt(vmp=0., imp=2., iph=2., io=2., rs=2.,
                                  rsh=2., nnsvth=2.)
    np.testing.assert_allclose(outrsh, np.array([502]), atol=.0001)


@requires_scipy
def test__update_rsh_fixed_pt_vector():
    outrsh8 = _update_rsh_fixed_pt(rsh=np.array([-1., 3, .5]),
                                   rs=np.array([1., -.5, 2.]),
                                   io=np.array([.2, .3, -.4]),
                                   iph=np.array([-.1, 1, 3]),
                                   nnsvth=np.array([4., -.2, .1]),
                                   imp=np.array([.2, .2, -1]),
                                   vmp=np.array([0., -1, 0.]))
    assert np.isnan(outrsh8[0])
    assert np.isnan(outrsh8[1])
    assert np.isnan(outrsh8[2])
