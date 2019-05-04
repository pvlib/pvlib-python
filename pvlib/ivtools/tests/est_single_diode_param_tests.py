import numpy as np
from pvlib.est_single_diode_param import estimate_parameters


def test_answer():
    i = np.array(
        [4., 3.95, 3.92, 3.9, 3.89, 3.88, 3.82, 3.8, 3.75, 3.7, 3.68, 3.66,
         3.65, 3.5, 3.2, 2.7, 2.2, 1.3, .6, 0.])
    v = np.array(
        [0., .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.7,
         2.76, 2.78, 2.81, 2.85, 2.88])
    nsvth = 2.

    io, iph, rs, rsh, n = estimate_parameters(i, v, nsvth)

    np.testing.assert_allclose(io, np.array([96699.876]), atol=.0001)
    np.testing.assert_allclose(iph, np.array([-96695.792]), atol=.0001)
    np.testing.assert_allclose(rs, np.array([.0288]), atol=.0001)
    np.testing.assert_allclose(rsh, np.array([7.4791]), atol=.0001)
    np.testing.assert_allclose(n, np.array([-.1413]), atol=.0001)


def test_answer1():
    i1 = np.array([3., 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 1.7, 0.8, 0.])
    v1 = np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.45, 1.5])
    nsvth1 = 10.

    io1, iph1, rs1, rsh1, n1 = estimate_parameters(i1, v1, nsvth1)

    np.testing.assert_allclose(io1, np.array([11.6865]), atol=.0001)
    np.testing.assert_allclose(iph1, np.array([2.3392]), atol=.0001)
    np.testing.assert_allclose(rs1, np.array([-.2596]), atol=.0001)
    np.testing.assert_allclose(rsh1, np.array([-.232]), atol=.0001)
    np.testing.assert_allclose(n1, np.array([-.7119]), atol=.0001)


def test_answer2():
    i2 = np.array(
        [5., 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4., 3.8, 3.5, 1.7,
         0.])
    v2 = np.array(
        [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1., 1.1, 1.18, 1.2, 1.22])
    nsvth2 = 15.

    io2, iph2, rs2, rsh2, n2 = estimate_parameters(i2, v2, nsvth2)

    np.testing.assert_allclose(io2, np.array([27.1196]), atol=.0001)
    np.testing.assert_allclose(iph2, np.array([-22.0795]), atol=.0001)
    np.testing.assert_allclose(rs2, np.array([-.0056]), atol=.0001)
    np.testing.assert_allclose(rsh2, np.array([-4.2076]), atol=.0001)
    np.testing.assert_allclose(n2, np.array([-.0498]), atol=.0001)


def test_answer3():
    i3 = np.array([3., 2.8, 2.6, 2.4, 2.2, 1.5, .7, 0.])
    v3 = np.array([0., .5, 1., 1.5, 2., 2.2, 2.3, 2.35])
    nsvth3 = 20.

    io3, iph3, rs3, rsh3, n3 = estimate_parameters(i3, v3, nsvth3)

    np.testing.assert_allclose(io3, np.array([46.8046]), atol=.0001)
    np.testing.assert_allclose(iph3, np.array([-42.4727]), atol=.0001)
    np.testing.assert_allclose(rs3, np.array([-.0437]), atol=.0001)
    np.testing.assert_allclose(rsh3, np.array([21.4408]), atol=.0001)
    np.testing.assert_allclose(n3, np.array([-.0422]), atol=.0001)
