from pvlib.lambertw import lambertw
import numpy as np

w = np.array([0])
z = lambertw(w)


def test_answer():
    np.testing.assert_array_almost_equal(z, np.array([0]), 4)

w1 = np.array([1, -1])
z1 = lambertw(w1)


def test_answer1():
    np.testing.assert_array_almost_equal(z1, np.array([0.5671, np.complex(-.3181, 1.3372)]), 4)

w2 = np.array([3, 4, 5])
z2 = lambertw(w2)


def test_answer2():
    np.testing.assert_array_almost_equal(z2, np.array([1.0499, 1.2022, 1.3267]), 4)

w3 = np.array([-3, -4, -5])
z3 = lambertw(w3)


def test_answer3():
    np.testing.assert_array_almost_equal(z3, np.array([np.complex(.4670, 1.8217), np.complex(.6788, 1.9120),
                                                       np.complex(.8448, 1.9750)]), 4)

w4 = np.array([-5, 0, .5, 1, 2, -1.5, 3])
z4 = lambertw(w4)


def test_answer4():
    np.testing.assert_array_almost_equal(z4, np.array([np.complex(.8448, 1.9750), 0, .3517, .5671, .8526,
                                                       np.complex(-.0328, 1.5496), 1.0499]), 4)
