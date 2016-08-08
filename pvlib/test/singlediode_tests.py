import numpy as np
from pvlib.singlediode import singlediode, i_from_v, g, calc_imp_bisect, calc_pmp_bisect
from nose.tools import assert_raises

y = g(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer():
    np.testing.assert_array_almost_equal(y, np.array([-9.3333]), 4)

y1 = g(np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer1():
    np.testing.assert_array_almost_equal(y1, np.array([1.0926]), 4)

y2 = g(np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer2():
    np.testing.assert_array_almost_equal(y2, np.array([-11.8643]), 4)

y3 = g(np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer3():
    assert np.isnan(y3)

y4 = g(np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]))


def test_answer4():
    assert np.isnan(y4)

y5 = g(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]))


def test_answer5():
    np.testing.assert_array_almost_equal(y5, np.array([-1.3333]), 4)

y6 = g(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]))


def test_answer6():
    assert np.isnan(y6)

y7 = g(np.array([1., -1.]), np.array([-1., 1.]), np.array([1., -1.]), np.array([-1., 1.]), np.array([1., -1.]),
       np.array([-1., 1.]))


def test_answer7():
    np.testing.assert_array_almost_equal(y7[0], np.array([0.0607]), 4)
    assert np.isnan(y7[1])

y8 = g(np.array([1., 2., -3.]), np.array([-.5, .2, 1.]), np.array([1.5, -.1, -.4]), np.array([-1., 1., .2]),
       np.array([4., -1., -.3]), np.array([.2, -1., 1.]))


def test_answer8():
    assert np.isnan(y8[0])
    np.testing.assert_array_almost_equal(y8[1], np.array([6.8618]), 4)
    assert np.isnan(y8[2])

imp = calc_imp_bisect(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer9():
    np.testing.assert_array_almost_equal(imp, np.array([.2218]), 4)

imp1 = calc_imp_bisect(np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer10():
    np.testing.assert_array_almost_equal(imp1, np.array([2]), 4)

imp2 = calc_imp_bisect(np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer11():
    np.testing.assert_array_almost_equal(imp2, np.array([2]), 4)

imp3 = calc_imp_bisect(np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]))


def test_answer12():
    np.testing.assert_array_almost_equal(imp3, np.array([4]), 4)

imp4 = calc_imp_bisect(np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]))


def test_answer13():
    np.testing.assert_array_almost_equal(imp4, np.array([1.0498]), 4)

imp5 = calc_imp_bisect(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]))


def test_answer14():
    np.testing.assert_array_almost_equal(imp5, np.array([4]), 4)

imp6 = calc_imp_bisect(np.array([1., -1.]), np.array([-1., 1.]), np.array([1., -1.]), np.array([-1., 1.]),
                       np.array([1., -1.]))


def test_answer15():
    np.testing.assert_array_almost_equal(imp6[0], np.array([0]), 4)
    assert np.isnan(imp6[1])

imp7 = calc_imp_bisect(np.array([.2, -.1, 1.]), np.array([-.3, .2, -.4]), np.array([2., .1, .2]),
                       np.array([.3, -.3, -.1]), np.array([.4, -.5, .6]))


def test_answer16():
    np.testing.assert_array_almost_equal(imp7[0], np.array([-.1]), 4)
    np.testing.assert_array_almost_equal(imp7[1], np.array([.1]), 4)
    np.testing.assert_array_almost_equal(imp7[2], np.array([.6]), 4)

impp, vmp, pmp = calc_pmp_bisect(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer17():
    np.testing.assert_array_almost_equal(impp, np.array([.2218]), 4)
    np.testing.assert_array_almost_equal(vmp, np.array([.5473]), 4)
    np.testing.assert_array_almost_equal(pmp, np.array([.1214]), 4)

impp1, vmp1, pmp1 = calc_pmp_bisect(np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer18():
    np.testing.assert_array_almost_equal(impp1, np.array([2]), 4)
    np.testing.assert_array_almost_equal(vmp1, np.array([-5.7052]), 4)
    np.testing.assert_array_almost_equal(pmp1, np.array([-11.4104]), 4)

impp2, vmp2, pmp2 = calc_pmp_bisect(np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer19():
    np.testing.assert_array_almost_equal(impp2, np.array([2]), 4)
    assert np.isnan(vmp2)
    assert np.isnan(pmp2)

impp3, vmp3, pmp3 = calc_pmp_bisect(np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]))


def test_answer20():
    np.testing.assert_array_almost_equal(impp3, np.array([4]), 4)
    assert np.isnan(vmp3)
    assert np.isnan(pmp3)

impp4, vmp4, pmp4 = calc_pmp_bisect(np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]))


def test_answer21():
    np.testing.assert_array_almost_equal(impp4, np.array([1.0498]), 4)
    np.testing.assert_array_almost_equal(vmp4, np.array([.5731]), 4)
    np.testing.assert_array_almost_equal(pmp4, np.array([.6016]), 4)

impp5, vmp5, pmp5 = calc_pmp_bisect(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]))


def test_answer22():
    np.testing.assert_array_almost_equal(impp5, np.array([4]), 4)
    assert np.isnan(vmp5)
    assert np.isnan(pmp5)

impp6, vmp6, pmp6 = calc_pmp_bisect(np.array([1., -1.]), np.array([-1., 1.]), np.array([1., -1.]), np.array([-1., 1.]),
                                    np.array([1., -1.]))


def test_answer23():
    np.testing.assert_array_almost_equal(impp6[0], np.array([0]), 4)
    assert np.isnan(impp6[1])
    assert np.isnan(vmp6[0])
    assert np.isnan(vmp6[1])
    assert np.isnan(pmp6[0])
    assert np.isnan(pmp6[1])

impp7, vmp7, pmp7 = calc_pmp_bisect(np.array([.2, .3, .4]), np.array([-.1, .3, .1]), np.array([-1.5, .1, -.4]),
                                    np.array([2., 1., -.5]), np.array([-2., 1., .4]))


def test_answer24():
    np.testing.assert_array_almost_equal(impp7, np.array([.1, .0254, .5]), 4)
    assert np.isnan(vmp7[0])
    np.testing.assert_array_almost_equal(vmp7[1], np.array([.0295]), 4)
    assert np.isnan(vmp7[2])
    assert np.isnan(pmp7[0])
    np.testing.assert_array_almost_equal(pmp7[1], np.array([.7503e-3]), 4)
    assert np.isnan(pmp7[2])

i = i_from_v(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer25():
    np.testing.assert_array_almost_equal(i, np.array([-.3726]), 4)

i1 = i_from_v(np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer26():
    np.testing.assert_array_almost_equal(i1, np.array([-1]), 4)

i2 = i_from_v(np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer27():
    assert np.isnan(i2)

i3 = i_from_v(np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer28():
    assert np.isnan(i3)

i4 = i_from_v(np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]))


def test_answer29():
    np.testing.assert_array_almost_equal(i4, np.array([.4429]), 4)

i5 = i_from_v(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]))


def test_answer30():
    np.testing.assert_array_almost_equal(i5, np.array([.5]), 4)

i6 = i_from_v(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]))


def test_answer31():
    np.testing.assert_array_almost_equal(i6, np.array([-.765]), 4)

i7 = i_from_v(np.array([1., -1.]), np.array([-1., 1.]), np.array([1., -1.]),
              np.array([-1., 1.]), np.array([1., -1.]), np.array([-1., 1.]))


def test_answer32():
    assert np.isnan(i7[0])
    assert np.isnan(i7[1])

i8 = i_from_v(np.array([.1, .2, -.1]), np.array([1., .5, -.3]), np.array([1.5, 2.1, -1.]), np.array([.4, .2, -.01]),
              np.array([2.2, .23, -4.]), np.array([.15, .3, -.5]))


def test_answer33():
    np.testing.assert_array_almost_equal(i8, np.array([-.3560, -.2032, -.1983]), 4)

result = singlediode(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer34():
    np.testing.assert_array_almost_equal(result['isc'], np.array([.4429]), 4)
    np.testing.assert_array_almost_equal(result['voc'], np.array([1.0926]), 4)
    np.testing.assert_array_almost_equal(result['imp'], np.array([.2218]), 4)
    np.testing.assert_array_almost_equal(result['vmp'], np.array([.5473]), 4)
    np.testing.assert_array_almost_equal(result['pmp'], np.array([.1214]), 4)
    np.testing.assert_array_almost_equal(result['ix'], np.array([.2223]), 4)
    np.testing.assert_array_almost_equal(result['ixx'], np.array([.1111]), 4)
    np.testing.assert_array_almost_equal(result['v'], np.array([]), 4)
    np.testing.assert_array_almost_equal(result['i'], np.array([]), 4)

result1 = singlediode(np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer35():
    np.testing.assert_array_almost_equal(result1['isc'], np.array([1]), 4)
    np.testing.assert_array_almost_equal(result1['voc'], np.array([4]), 4)
    np.testing.assert_array_almost_equal(result1['imp'], np.array([2]), 4)
    assert np.isnan(result1['vmp'])
    assert np.isnan(result1['pmp'])
    np.testing.assert_array_almost_equal(result1['ix'], np.array([.5]), 4)
    assert np.isnan(result1['ixx'])
    np.testing.assert_array_almost_equal(result1['v'], np.array([]), 4)
    np.testing.assert_array_almost_equal(result1['i'], np.array([]), 4)

result2 = singlediode(np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]))


def test_answer36():
    assert np.isnan(result2['isc'])
    np.testing.assert_array_almost_equal(result2['voc'], np.array([1.0926]), 4)
    np.testing.assert_array_almost_equal(result2['imp'], np.array([1.0498]), 4)
    np.testing.assert_array_almost_equal(result2['vmp'], np.array([.5731]), 4)
    np.testing.assert_array_almost_equal(result2['pmp'], np.array([.6016]), 4)
    assert np.isnan(result2['ix'])
    assert np.isnan(result2['ixx'])
    np.testing.assert_array_almost_equal(result2['v'], np.array([]), 4)
    np.testing.assert_array_almost_equal(result2['i'], np.array([]), 4)

result3 = singlediode(np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]))


def test_answer37():
    np.testing.assert_array_almost_equal(result3['isc'], np.array([0]), 4)
    np.testing.assert_array_almost_equal(result3['voc'], np.array([0]), 4)
    np.testing.assert_array_almost_equal(result3['imp'], np.array([4]), 4)
    assert np.isnan(result3['vmp'])
    assert np.isnan(result3['pmp'])
    np.testing.assert_array_almost_equal(result3['ix'], np.array([0]), 4)
    assert np.isnan(result3['ixx'])
    np.testing.assert_array_almost_equal(result3['v'], np.array([]), 4)
    np.testing.assert_array_almost_equal(result3['i'], np.array([]), 4)

result4 = singlediode(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]))


def test_answer38():
    assert np.isnan(result4['isc'])
    assert np.isnan(result4['voc'])
    np.testing.assert_array_almost_equal(result4['imp'], np.array([4]), 4)
    assert np.isnan(result4['vmp'])
    assert np.isnan(result4['pmp'])
    assert np.isnan(result4['ix'])
    assert np.isnan(result4['ixx'])
    np.testing.assert_array_almost_equal(result4['v'], np.array([]), 4)
    np.testing.assert_array_almost_equal(result4['i'], np.array([]), 4)

result5 = singlediode(np.array([1., 2.]), np.array([1., 2.]), np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer39():
    np.testing.assert_array_almost_equal(result5['isc'], np.array([.3149, .4429]), 4)
    np.testing.assert_array_almost_equal(result5['voc'], np.array([.8857, 1.0926]), 4)
    np.testing.assert_array_almost_equal(result5['imp'], np.array([.1579, .2218]), 4)
    np.testing.assert_array_almost_equal(result5['vmp'], np.array([.4441, .5473]), 4)
    np.testing.assert_array_almost_equal(result5['pmp'], np.array([.0701, .1214]), 4)
    np.testing.assert_array_almost_equal(result5['ix'], np.array([.1583, .2223]), 4)
    np.testing.assert_array_almost_equal(result5['ixx'], np.array([.0792, .1111]), 4)
    np.testing.assert_array_almost_equal(result5['v'], np.array([]), 4)
    np.testing.assert_array_almost_equal(result5['i'], np.array([]), 4)

result6 = singlediode(np.array([1.5, .5, 3.]), np.array([4., .2, .8]), np.array([2.5, 7., 1.]), np.array([2.]),
                      np.array([2.]), np.array([3.]))


def test_answer40():
    np.testing.assert_array_almost_equal(result6['isc'], np.array([.19, .0938, 1.4385]), 4)
    np.testing.assert_array_almost_equal(result6['voc'], np.array([.5368, .8025, 2.3691]), 4)
    np.testing.assert_array_almost_equal(result6['imp'], np.array([.095, .0469, .7308]), 4)
    np.testing.assert_array_almost_equal(result6['vmp'], np.array([.2685, .4014, 1.2019]), 4)
    np.testing.assert_array_almost_equal(result6['pmp'], np.array([.0255, .0188, .8784]), 4)
    np.testing.assert_array_almost_equal(result6['ix'], np.array([.0951, .047, .7414]), 4)
    np.testing.assert_array_almost_equal(result6['ixx'], np.array([.0475, .0235, .3706]), 4)
    np.testing.assert_array_almost_equal(result6['v'], np.array([[0., .2684, .5368], [0., .4013, .8025],
                                                                 [0., 1.1846, 2.3691]]), 4)
    np.testing.assert_array_almost_equal(result6['i'], np.array([[.19, .0951, 0.], [.0938, .047, 0.],
                                                                 [1.4385, .7414, 0.]]), 4)

result7 = singlediode(np.array([1.5]), np.array([4.]), np.array([1.]), np.array([2.]), np.array([2.]), np.array([10.]))


def test_answer41():
    np.testing.assert_array_almost_equal(result7['isc'], np.array([.4036]), 4)
    np.testing.assert_array_almost_equal(result7['voc'], np.array([.5368]), 4)
    np.testing.assert_array_almost_equal(result7['imp'], np.array([.2022]), 4)
    np.testing.assert_array_almost_equal(result7['vmp'], np.array([.2689]), 4)
    np.testing.assert_array_almost_equal(result7['pmp'], np.array([.0544]), 4)
    np.testing.assert_array_almost_equal(result7['ix'], np.array([.2025]), 4)
    np.testing.assert_array_almost_equal(result7['ixx'], np.array([.1013]), 4)
    np.testing.assert_array_almost_equal(result7['v'], np.array([0., .0596, .1193, .1789, .2386, .2982, .3579, .4175,
                                                                 .4772, .5368]), 4)
    np.testing.assert_array_almost_equal(result7['i'], np.array([.4036, .3591, .3144, .2697, .2249, .1801, .1352, .0902,
                                                                 .0451, 0.]), 4)


def test_answer42():
    assert_raises(ValueError, singlediode, np.array([-1.5]), np.array([4.]), np.array([1.]), np.array([2.]),
                  np.array([2.]))


def test_answer43():
    assert_raises(ValueError, singlediode, np.array([1.5]), np.array([4.]), np.array([1.]), np.array([2.]),
                  np.array([-2.]))


def test_answer44():
    assert_raises(ValueError, singlediode, np.array([1.5]), np.array([4.]), np.array([1.]), np.array([2.]),
                  np.array([2.]), np.array([2., 2.]))


def test_answer45():
    a = float("inf")
    assert_raises(ValueError, singlediode, np.array([1.5]), np.array([4.]), np.array([1.]), np.array([2.]),
                  np.array([2.]), np.array([a]))


def test_answer46():
    assert_raises(ValueError, singlediode, np.array([1.5, 2.5]), np.array([4.]), np.array([1.]), np.array([2.]),
                  np.array([2., 2., 2.5]))
