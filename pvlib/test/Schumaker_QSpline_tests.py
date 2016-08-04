import numpy as np
from pvlib.Schumaker_QSpline import schumaker_qspline

x = np.array([0., 1., 2., 3., 4., 1., 2., 3., 4., 5.])
y = np.array([2., 1., 0., 1., 2., 3., 2., 1., 2., 3.])

outa, outxk, outy, kflag = schumaker_qspline(x, y)


def test_answer():
    np.testing.assert_array_almost_equal(outa, np.array([[0., -1., 2.], [-0.5, -1., 1.], [-0.75, -0.5, 3.],
                                                         [0.75, -1.5, 0.375], [0.125, -1.25, 2.5625], [1.5, 0., 0.],
                                                         [-0.5, -1., 2.], [-0.25, 1.5, 0.375], [0.75, -1.5, 1.375],
                                                         [0.5, 1., 1.], [1.5, 0., 1.], [0.0278, -0.3333, 2.1667],
                                                         [-0.75, 1.5, 1.625], [-0.25, 1.5, 1.375], [0.1667, 0., 2.],
                                                         [0., 1., 2.]]), 4)
    np.testing.assert_array_almost_equal(outxk, np.array([0., 1., 1., 1.5, 1.5, 2., 2., 2.5, 2.5, 3., 3., 3., 3.5, 3.5,
                                                          4., 4., 5.]), 4)
    np.testing.assert_array_almost_equal(outy, np.array([2., 1., 3., 0.375, 2.5625, 0., 2., 0.375, 1.375, 1., 1.,
                                                         2.1667, 1.625, 1.375, 2., 2., 3.]), 4)
    np.testing.assert_array_almost_equal(kflag, np.array([0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0.,
                                                          0., 0.]), 4)

x1 = np.array([1., 2., 3., 4., 5.])
y1 = np.array([-2., -1., 0., 1., 2.])

outa1, outxk1, outy1, kflag1 = schumaker_qspline(x1, y1)


def test_answer1():
    np.testing.assert_array_almost_equal(outa1, np.array([[0., 1., -2.], [0., 1., -1.], [0., 1., 0.],
                                                         [0., 1., 1.]]), 4)
    np.testing.assert_array_almost_equal(outxk1, np.array([1., 2., 3., 4., 5.]), 4)
    np.testing.assert_array_almost_equal(outy1, np.array([-2., -1., 0., 1., 2.]), 4)
    np.testing.assert_array_almost_equal(kflag1, np.array([0., 0., 0., 0., 0.]), 4)

x2 = np.array([-.5, -.1, 0., .2, .3])
y2 = np.array([-5., -1., .2, .5, 2.])

outa2, outxk2, outy2, kflag2 = schumaker_qspline(x2, y2)


def test_answer2():
    np.testing.assert_array_almost_equal(outa2, np.array([[2.2727, 9.0909, -5.], [63.0303, 10.9091, -1.],
                                                          [-72.7273, 17.2121, -.297], [-11.8182, 2.6667, .2],
                                                          [6.0606, .303, .3485], [122.7273, 2.7273, .5]]), 4)
    np.testing.assert_array_almost_equal(outxk2, np.array([-.5, -.1, -.05, 0., .1, .2, .3]), 4)
    np.testing.assert_array_almost_equal(outy2, np.array([-5., -1., -.297, .2, .3485, .5, 2.]), 4)
    np.testing.assert_array_almost_equal(kflag2, np.array([0., 0., 1., 0., 1., 0., 0.]), 4)
