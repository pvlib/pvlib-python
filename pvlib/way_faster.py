"""faster ways"""

import logging
import numpy as np

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

EPS = np.finfo(float).eps
DAMP = 1.5
DELTA = EPS**0.33


def est_voc(il, io, nnsvt):
    # http://www.pveducation.org/pvcdrom/open-circuit-voltage
    return nnsvt * np.log(il / io + 1.0)


def bishop88(vd, il, io, rs, rsh, nnsvt):
    """bishop 1988"""
    a = np.exp(vd / nnsvt)
    b = 1.0 / rsh
    i = il - io * (a - 1.0) - vd * b
    v = vd - i * rs
    c = io * a / nnsvt
    grad_i = - c - b  # di/dvd
    grad_v = 1.0 - grad_i * rs  # dv/dvd
    # dp/dv = d(iv)/dv = v * di/dv + i
    grad = grad_i / grad_v
    grad_p = v * grad + i  # dp/dv
    grad2i = -c / nnsvt
    grad2v = -grad2i * rs
    grad2p = (grad_v * grad + v * (grad2i/grad_v - grad_i*grad2v/grad_v**2)
              + grad_i)
    return i, v, grad_i, grad_v, i*v, grad_p, grad2p


def newton_solver(fjx, x0, x, tol=EPS, damp=DAMP, log=False, test=True):
    resnorm = np.inf  # nor of residuals
    while resnorm > tol:
        f, j = fjx(x0, *x)
        newton_step = f / j
        # don't let step get crazy
        if np.abs(newton_step / x0) > damp:
            break
        x0 -= newton_step
        resnorm = f**2
        if log:
            LOGGER.debug(
                'x0=%g, newton step=%g, f=%g, resnorm=%g',
                x0, newton_step, f, resnorm
            )
        if test:
            f2, _ = fjx(x0 * (1.0 + DELTA), *x)
            LOGGER.debug('test_grad=%g', (f2 - f) / x0 / DELTA)
            LOGGER.debug('grad=%g', j)
    return x0, f, j


def faster_way(il, io, rs, rsh, nnsvt,
               tol=EPS, damp=DAMP, log=True, test=True):
    """a faster way"""
    x = (il, io, rs, rsh, nnsvt)  # collect args
    # first estimate Voc
    voc_est = est_voc(il, io, nnsvt)
    # find the real voc
    resnorm = np.inf  # nor of residuals
    while resnorm > tol:
        i_test, v_test, grad, _, _, _, _ = bishop88(voc_est, *x)
        newton_step = i_test / grad
        # don't let step get crazy
        if np.abs(newton_step / voc_est) > damp:
            break
        voc_est -= newton_step
        resnorm = i_test**2
        if log:
            LOGGER.debug(
                'voc_est=%g, step=%g, i_test=%g, v_test=%g, resnorm=%g',
                voc_est, newton_step, i_test, v_test, resnorm
            )
        if test:
            delta = EPS**0.3
            i_test2, _, _, _, _, _, _ = bishop88(voc_est * (1.0 + delta), *x)
            LOGGER.debug('test_grad=%g', (i_test2 - i_test) / voc_est / delta)
            LOGGER.debug('grad=%g', grad)
    # find isc too
    vd_sc = 0.0
    resnorm = np.inf  # nor of residuals
    while resnorm > tol:
        isc_est, v_test, _, grad, _, _, _ = bishop88(vd_sc, *x)
        newton_step = v_test / grad
        # don't let step get crazy
        if np.abs(newton_step / voc_est) > damp:
            break
        vd_sc -= newton_step
        resnorm = v_test**2
        if log:
            LOGGER.debug(
                'vd_sc=%g, step=%g, isc_est=%g, v_test=%g, resnorm=%g',
                vd_sc, newton_step, isc_est, v_test, resnorm
            )
        if test:
            delta = EPS**0.3
            _, v_test2, _, _, _, _, _ = bishop88(vd_sc * (1.0 + delta), *x)
            LOGGER.debug('test_grad=%g', (v_test2 - v_test) / vd_sc / delta)
            LOGGER.debug('grad=%g', grad)
    # find the mpp
    vd_mp = voc_est
    resnorm = np.inf  # nor of residuals
    while resnorm > tol:
        imp_est, vmp_est, _, _, pmp_est, grad_p, grad2p = bishop88(vd_mp, *x)
        newton_step = grad_p / grad2p
        # don't let step get crazy
        if np.abs(newton_step / voc_est) > damp:
            break
        vd_mp -= newton_step
        resnorm = grad_p**2
        if log:
            LOGGER.debug(
                'vd_mp=%g, step=%g, pmp_est=%g, resnorm=%g',
                vd_mp, newton_step, pmp_est, resnorm
            )
        if test:
            delta = EPS**0.3
            _, _, _, _, _, grad_p2, _ = bishop88(vd_mp * (1.0 + delta), *x)
            LOGGER.debug('test_grad=%g', (grad_p2 - grad_p) / vd_mp / delta)
            LOGGER.debug('grad=%g', grad2p)
    return voc_est, isc_est, imp_est, vmp_est, pmp_est
