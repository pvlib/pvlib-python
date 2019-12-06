from pvlib.ivtools.update_io_known_n import update_io_known_n
import numpy as np
from conftest import requires_scipy


@requires_scipy
def test_answer9():
    outio = update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]),
                              np.array([2.]), np.array([2.]), np.array([2.]))
    np.testing.assert_allclose(outio, np.array([0.5911]), atol=.0001)


@requires_scipy
def test_answer10():
    outio1 = update_io_known_n(np.array([0.]), np.array([2.]), np.array([2.]),
                               np.array([2.]), np.array([2.]), np.array([2.]))
    np.testing.assert_allclose(outio1, np.array([1.0161e-4]), atol=.0001)


@requires_scipy
def test_answer11():
    outio2 = update_io_known_n(np.array([2.]), np.array([0.]), np.array([2.]),
                               np.array([2.]), np.array([2.]), np.array([2.]))
    np.testing.assert_allclose(outio2, np.array([0.5911]), atol=.0001)


@requires_scipy
def test_answer12():
    outio3 = update_io_known_n(np.array([2.]), np.array([2.]), np.array([0.]),
                               np.array([2.]), np.array([2.]), np.array([2.]))
    assert np.isnan(outio3)


@requires_scipy
def test_answer13():
    outio4 = update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]),
                               np.array([0.]), np.array([2.]), np.array([2.]))
    np.testing.assert_allclose(outio4, np.array([0]), atol=.0001)


@requires_scipy
def test_answer14():
    outio5 = update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]),
                               np.array([2.]), np.array([0.]), np.array([2.]))
    np.testing.assert_allclose(outio5, np.array([1.0161e-4]), atol=.0001)


@requires_scipy
def test_answer15():
    outio6 = update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]),
                               np.array([2.]), np.array([2.]), np.array([0.]))
    np.testing.assert_allclose(outio6, np.array([17.9436]), atol=.0001)


@requires_scipy
def test_answer16():
    outio7 = update_io_known_n(np.array([-1.]), np.array([-1.]),
                               np.array([-1.]), np.array([-1.]),
                               np.array([-1.]), np.array([-1.]))
    assert np.isnan(outio7)
