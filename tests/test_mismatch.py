import numpy as np
from pvlib import singlediode as _singlediode
from pvlib.mismatch import _setup_currents, _singlediode_mismatch


def test__setup_currents():

    # 2 devices, 1 time
    cur_bkpts = np.array([[4.1, 5.2]]).T
    string_isc = np.array([6.])
    curs = _setup_currents(cur_bkpts, string_isc, 10)
    assert np.isin(cur_bkpts, curs).all()
    assert (np.diff(curs) < 0.).all()  # strictly decreasing
    assert curs[:, -1] == 0.
    assert curs[:, 0] == string_isc


def test__singlediode_mismatch_isc_voc():

    # 2 devices, 1 time
    # Isc should be equal to the current on the higher curve where voltage is
    # +neg_v_limit
    # Voc should be equal to the sum of Voc from each device
    IL = np.array([[1.0], [6.01]])
    Io = 1e-9
    nNsVth = 2.5
    Rs = 0.5
    Rsh = 1000.
    neg_v_limit = -5.
    expected_isc = _singlediode._lambertw_i_from_v(
        -neg_v_limit, IL[1], Io, Rs, Rsh, nNsVth)
    expected_voc = _singlediode._lambertw_v_from_i(
        0, IL, Io, Rs, Rsh, nNsVth)
    isc, voc, _, _ = _singlediode_mismatch(
        IL, Io, Rs, Rsh, nNsVth, neg_v_limit=neg_v_limit)
    assert np.isclose(isc[0], expected_isc)
    assert np.isclose(voc[0], expected_voc.sum())
