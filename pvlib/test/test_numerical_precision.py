"""
Numerical Precision
http://docs.sympy.org/latest/modules/evalf.html#accuracy-and-error-handling
"""

import logging
import numpy as np
from sympy import symbols, exp as sy_exp
from pvlib import pvsystem
from pvlib import way_faster

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

POA = 888
TCELL = 55
CECMOD = pvsystem.retrieve_sam('cecmod')


def test_numerical_precicion():
    """
    Test that there are no numerical errors due to floating point arithmetic.
    """
    il, io, rs, rsh, nnsvt, vd = symbols('il, io, rs, rsh, nnsvt, vd')
    a = sy_exp(vd / nnsvt)
    b = 1.0 / rsh
    i = il - io * (a - 1.0) - vd * b
    v = vd - i * rs
    c = io * a / nnsvt
    grad_i = - c - b  # di/dvd
    grad_v = 1.0 - grad_i * rs  # dv/dvd
    # dp/dv = d(iv)/dv = v * di/dv + i
    grad = grad_i / grad_v  # di/dv
    p = i * v
    grad_p = v * grad + i  # dp/dv
    grad2i = -c / nnsvt
    grad2v = -grad2i * rs
    grad2p = (
        grad_v * grad + v * (grad2i/grad_v - grad_i*grad2v/grad_v**2) + grad_i
    )
    # get module from cecmod and apply temp/irrad desoto corrections
    spr_e20_327 = CECMOD.SunPower_SPR_E20_327
    x = pvsystem.calcparams_desoto(
        poa_global=POA, temp_cell=TCELL,
        alpha_isc=spr_e20_327.alpha_sc, module_parameters=spr_e20_327,
        EgRef=1.121, dEgdT=-0.0002677)
    data = dict(zip((il, io, rs, rsh, nnsvt), x))
    vdtest = np.linspace(0, way_faster.est_voc(x[0], x[1], x[4]), 100)
    results = way_faster.bishop88(vdtest, *x)
    for test, expected in zip(vdtest, zip(*results)):
        data[vd] = test
        LOGGER.debug('expected = %r', expected)
        LOGGER.debug('test data = %r', data)
        assert np.isclose(np.float64(i.evalf(subs=data)), expected[0])
        assert np.isclose(np.float64(v.evalf(subs=data)), expected[1])
        assert np.isclose(np.float64(grad_i.evalf(subs=data)), expected[2])
        assert np.isclose(np.float64(grad_v.evalf(subs=data)), expected[3])
        assert np.isclose(np.float64(grad.evalf(subs=data)),
                          expected[2] / expected[3])
        assert np.isclose(np.float64(p.evalf(subs=data)), expected[4])
        assert np.isclose(np.float64(grad_p.evalf(subs=data)), expected[5])
        assert np.isclose(np.float64(grad2p.evalf(subs=data)), expected[6])
    return i, v, grad_i, grad_v, grad, p, grad_p, grad2p


if __name__ == '__main__':
    syms = test_numerical_precicion()
