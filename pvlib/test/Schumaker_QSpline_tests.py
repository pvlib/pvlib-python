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
