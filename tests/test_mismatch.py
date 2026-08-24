import numpy as np
from pvlib import singlediode as _singlediode
from pvlib.mismatch import _setup_currents, _iv_series_lambertw


def test__setup_currents():

    # 2 devices, 1 time
    cur_bkpts = np.array([[4.1, 5.2]]).T
    string_isc = np.array([6.])
    curs = _setup_currents(cur_bkpts, string_isc, 10)
    assert np.isin(cur_bkpts, curs).all()
    assert (np.diff(curs) < 0.).all()  # strictly decreasing
    assert curs[:, -1] == 0.
    assert curs[:, 0] == string_isc


def test__iv_series_lambertw_isc():

    # 2 devices, 1 time
    # Isc should be equal to the current on the higher curve where voltage is
    # +neg_v_limit
    IL = np.array([[1.0], [6.01]])
    Io = 1e-9
    nNsVth = 2.5
    Rs = 0.5
    Rsh = 1000.
    npts = 5
    neg_v_limit = -5.
    expected_isc = _singlediode._lambertw_i_from_v(-neg_v_limit, IL[1], Io, Rs,
                                                   Rsh, nNsVth)
    vs, cs, _, _ = _iv_series_lambertw(IL, Io, Rs, Rsh, nNsVth,
                                       neg_v_limit=neg_v_limit,
                                       npts=npts)
    assert vs.shape == cs.shape
    assert vs.shape == (1, npts)  # ntimes x npts
    assert np.isclose(cs[0, 0], expected_isc)
    assert np.isclose(vs[0, 0], 0.)


def test__iv_series_lambertw_voc():

    # 2 devices, 1 time
    # Voc should be equal to the sum of Voc for each device
    IL = np.array([[1.0], [6.01]])
    Io = 1e-9
    nNsVth = 2.5
    Rs = 0.5
    Rsh = 1000.
    npts = 5
    neg_v_limit = -5.
    expected_voc = _singlediode._lambertw_v_from_i(0, IL, Io, Rs,
                                                   Rsh, nNsVth)
    vs, cs, _, _ = _iv_series_lambertw(IL, Io, Rs, Rsh, nNsVth,
                                       neg_v_limit=neg_v_limit,
                                       npts=npts)

    assert np.isclose(vs[0, -1], expected_voc.sum())
    assert np.isclose(cs[0, -1], 0.)


def test__iv_series_lambertw_breakpoints():

    # 2 devices, 1 time
    # Voc should be equal to the sum of Voc for each device
    IL = np.array([[1.0], [6.01]])
    Io = 1e-9
    nNsVth = 2.5
    Rs = 0.5
    Rsh = 1000.
    npts = 10
    neg_v_limit = -5.

    cur_breakpoints = _singlediode._lambertw_i_from_v(neg_v_limit, IL, Io, Rs,
                                                      Rsh, nNsVth)
    vs, cs, _, _ = _iv_series_lambertw(IL, Io, Rs, Rsh, nNsVth,
                                       neg_v_limit=neg_v_limit,
                                       npts=npts)
    # current cs should contain all breakpoints less than string_isc
    assert np.isin(cur_breakpoints[cur_breakpoints < cs[0, 0]], cs).all()
