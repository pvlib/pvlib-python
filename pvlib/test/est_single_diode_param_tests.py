import numpy as np
from pvlib.est_single_diode_param import est_single_diode_param

i = np.array([4., 3.95, 3.92, 3.9, 3.89, 3.88, 3.82, 3.8, 3.75, 3.7, 3.68, 3.66, 3.65, 3.5, 3.2, 2.7, 2.2, 1.3, .6, 0.])
v = np.array([0., .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.7, 2.76, 2.78, 2.81, 2.85, 2.88])
nsvth = 2.

io, iph, rs, rsh, n = est_single_diode_param(i, v, nsvth)


def test_answer():
    np.testing.assert_array_almost_equal(io, np.array([96699.876]), 4)
    np.testing.assert_array_almost_equal(iph, np.array([-96695.792]), 4)
    np.testing.assert_array_almost_equal(rs, np.array([.0288]), 4)
    np.testing.assert_array_almost_equal(rsh, np.array([7.4791]), 4)
    np.testing.assert_array_almost_equal(n, np.array([-.1413]), 4)

i1 = np.array([3., 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 1.7, 0.8, 0.])
v1 = np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.45, 1.5])
nsvth1 = 10.

io1, iph1, rs1, rsh1, n1 = est_single_diode_param(i1, v1, nsvth1)


def test_answer1():
    np.testing.assert_array_almost_equal(io1, np.array([11.6865]), 4)
    np.testing.assert_array_almost_equal(iph1, np.array([2.3392]), 4)
    np.testing.assert_array_almost_equal(rs1, np.array([-.2596]), 4)
    np.testing.assert_array_almost_equal(rsh1, np.array([-.232]), 4)
    np.testing.assert_array_almost_equal(n1, np.array([-.7119]), 4)

i2 = np.array([5., 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4., 3.8, 3.5, 1.7, 0.])
v2 = np.array([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1., 1.1, 1.18, 1.2, 1.22])
nsvth2 = 15.

io2, iph2, rs2, rsh2, n2 = est_single_diode_param(i2, v2, nsvth2)


def test_answer2():
    np.testing.assert_array_almost_equal(io2, np.array([27.1196]), 4)
    np.testing.assert_array_almost_equal(iph2, np.array([-22.0795]), 4)
    np.testing.assert_array_almost_equal(rs2, np.array([-.0056]), 4)
    np.testing.assert_array_almost_equal(rsh2, np.array([-4.2076]), 4)
    np.testing.assert_array_almost_equal(n2, np.array([-.0498]), 4)

i3 = np.array([3., 2.8, 2.6, 2.4, 2.2, 1.5, .7, 0.])
v3 = np.array([0., .5, 1., 1.5, 2., 2.2, 2.3, 2.35])
nsvth3 = 20.

io3, iph3, rs3, rsh3, n3 = est_single_diode_param(i3, v3, nsvth3)


def test_answer3():
    np.testing.assert_array_almost_equal(io3, np.array([46.8046]), 4)
    np.testing.assert_array_almost_equal(iph3, np.array([-42.4727]), 4)
    np.testing.assert_array_almost_equal(rs3, np.array([-.0437]), 4)
    np.testing.assert_array_almost_equal(rsh3, np.array([21.4408]), 4)
    np.testing.assert_array_almost_equal(n3, np.array([-.0422]), 4)
