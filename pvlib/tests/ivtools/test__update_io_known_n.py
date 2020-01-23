import numpy as np
import pytest
from pvlib.ivtools.sdm import _update_io
from pvlib.tests.conftest import requires_scipy

# old (rsh, rs, nnsvth, io, il, voc)
# new (voc, iph, io, rs, rsh, nnsvth)

@requires_scipy
@pytest.mark.parametrize('voc, iph, io, rs, rsh, nnsvth, expected', [
    (2., 2., 2., 2., 2., 2., 0.5911),
    (2., 2., 2., 2., 0., 2., 1.0161e-4),
    (2., 2., 2., 0., 2., 2., 0.5911),
    (2., 2., 0., 2., 2., 2., 0.),
    (2., 0., 2., 2., 2., 2., 1.0161e-4),
    (0., 2., 2., 2., 2., 2., 17.9436)])
def test__update_io(voc, iph, io, rs, rsh, nnsvth, expected):
    outio = _update_io(voc, iph, io, rs, rsh, nnsvth)
    np.testing.assert_allclose(outio, expected, atol=.0001)


@requires_scipy
@pytest.mark.parametrize('voc, iph, io, rs, rsh, nnsvth', [
    (2., 2., 2., 2., 2., 0.),
    (-1., -1., -1., -1., -1., -1.)])
def test__update_io_nan(voc, iph, io, rs, rsh, nnsvth):
    outio = _update_io(voc, iph, io, rs, rsh, nnsvth)
    assert np.isnan(outio)
