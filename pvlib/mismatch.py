"""
Contains functions for solving for DC power in arrays with mismatched
conditions.

"""
import numpy as np
from scipy.optimize.elementwise import find_root, find_minimum
from pvlib import singlediode as _singlediode


def _iv_series_lambert_v_from_i(current, il, io, rs, rsh, a, neg_v_limit,
                                ndevices=None, idx=None):
    # wrapper for pvlib._singlediode._lambertw_v_from_i, handles
    # dimensions expected in series calculation
    # solve voltages at each current for each IV curve

    # slice each parameter on its ntimes dimension with idx
    if idx is not None:
        il, io, rs, rsh, a = (il[:, idx], io[:, idx], rs[:, idx], rsh[:, idx],
                              a[:, idx])

    voltages = _singlediode._lambertw_v_from_i(
        current.flatten(), il.flatten(), io.flatten(), rs.flatten(),
        rsh.flatten(), a.flatten())

    # apply negative voltage limit
    voltages[voltages < neg_v_limit] = neg_v_limit

    # reshape
    voltages = voltages.reshape(current.shape)
    return voltages


def _setup_currents(current_bkpts, string_isc, npts):
    r''' Form array of currents from string_isc down to 0.
    The array of currents will contain all values
    from current_bkpts which are less than string_isc. Remaining points are
    selected from a linear spacing [0, string_isc], avoiding points that are
    closest to values in current_bkpts.

    Parameters
    ----------
    device_isc : ndarray
        Shape (ndevices, ntimes)
    string_isc : ndarray
        Shape (ntimes,)
    npts : int
        number of current points in the returned array

    Returns
    -------
    ndarray
        shape (ntimes, npts)

    '''
    ntimes = len(string_isc)
    currents = np.zeros((ntimes, npts))

    u = current_bkpts < string_isc[np.newaxis, :]

    # have to loop on ntimes since count of device_isc < string_isc may
    # differ for each time
    for i in range(ntimes):
        # start with values in current_bkpts
        vals = np.unique(current_bkpts[u[:, i], i])
        # add string_isc and 0.
        vals = np.append(vals, [string_isc[i], 0.])
        k_i = len(vals)

        if k_i == 0:
            continue

        # Store these values
        currents[i, :k_i] = vals

        # how many more we need
        n_fill = npts - k_i
        if n_fill <= 0:
            continue

        # Build linear spacing to draw from
        grid = np.linspace(string_isc[i], 0., npts)

        # Compute distances to nearest points in current_bkpts etc.
        # shape: (grid_size, k_i)
        dists = np.abs(grid[:, None] - vals[None, :])

        # nearest distance per grid point
        nearest_dist = np.min(dists, axis=1)

        # inverse-distance weights (higher near must-have values)
        weights = 1.0 / (nearest_dist + 1e-12)  # 1e-12 to avoid div by 0

        # Avoid re-selecting original values exactly
        mask_existing = np.isclose(nearest_dist, 0.0, atol=1e-12)
        weights[mask_existing] = 0.0

        # Select top-weighted grid points
        idx = np.argpartition(weights, -n_fill)[-n_fill:]
        selected = grid[idx]

        # Combine and add to A
        currents[i, :] = np.concatenate([vals, selected])

    # return sorted in descending order for each time
    currents = -np.sort(-currents, axis=1)

    return currents


def _insert_mpp(vol, cur, vmp, imp):
    # insert MPP into current and voltage arrays
    cur_out = np.empty((vol.shape[0], vol.shape[1] + 1))
    vol_out = np.empty((vol.shape[0], vol.shape[1] + 1))
    # insert mpp value
    for i in range(vol.shape[0]):
        idx = vol[i, :].searchsorted(vmp[i], side='left')
        vol_out[i, :idx] = vol[i, :idx]
        vol_out[i, idx] = vmp[i]
        vol_out[i, idx+1:] = vol[i, idx:]
        cur_out[i, :idx] = cur[i, :idx]
        cur_out[i, idx] = imp[i]
        cur_out[i, idx+1:] = cur[i, idx:]

    return vol_out, cur_out


def _singlediode_mismatch(photocurrent, saturation_current, resistance_series,
                        resistance_shunt, nNsVth, neg_v_limit=0.,
                        npts=100):
    r'''Solve the IV curve for series-connected devices where each device
    is described by the single diode equation.

    Uses a simplified model for reverse bias behavior, where current is
    unbounded at a constant reverse bias voltage ``neg_v_limit``.

    Input parameter ``photocurrent`` must have shape (devices, times).
    Input parameters ``saturation_current``, ``resistance_series``,
    ``resistance_shunt``, ``nNsVth`` may be arrays. If arrays, must be
    broadcastable to the shape of ``photocurrent``.

    Parameters
    ----------
    photocurrent : numeric
        photocurrent (A). Must have shape (devices, times).
    saturation_current : numeric
        saturation current (A). Must be broadcastable with photocurrent.
    resistance_series : numeric
        series resistance (ohm). Must be broadcastable with photocurrent.
    resistance_shunt : numeric
        shunt resistance (ohm). Must be broadcastable with photocurrent.
    nNsVth : numeric
        product of diode factor n, number of series cells Ns, and
        thermal voltage (Vth), (V). Must be broadcastable with photocurrent.
    neg_v_limit : float, optional
        Limit on reverse bias voltage, from cell breakdown voltage or reverse
        bias diode activation voltage (V). Should be negative. For example,
        if neg_v_limit=-5, then at V=-5 current is unbounded in the positive
        direction.
    npts : int, optional
        Number of points used to discretize the returned IV curves.

    Returns
    -------
    string_isc : numeric
        Short-circuit current for the string. [A]
    string_voc : numeric
        Open-circuit voltage for the string. [V]
    string_imp : numeric
        Current at maximum power for the string. [A]
    string_vmp : numeric
        Voltage at maximum power for the string. [V]

    '''
    # target shape is ndevices x ntimes
    IL, I0, Rs, Rsh, a = \
        np.broadcast_arrays(photocurrent, saturation_current,
                            resistance_series, resistance_shunt, nNsVth)

    ndevices, ntimes = IL.shape

    # solve for current at negative voltage limit for each device.
    # these currents create breakpoints in the series IV curve
    current_bkpts = _singlediode._lambertw_i_from_v(
        neg_v_limit, IL, I0, Rs, Rsh, a)

    # find Isc for string IV curve
    # bounds, 1d array for each time
    max_isc = current_bkpts.max(axis=0) * 1.01
    min_isc = current_bkpts.min(axis=0) * 0.99

    # Use an index idx so that find_root can slice arguments
    # Internally find_root will slice current as each element converges
    # As an argument, idx lets find_root also slice the other parameters
    # Remove idx and use preserve_shape once available in find_root
    # https://github.com/scipy/scipy/issues/24869
    idx = np.arange(ntimes)
    def isc_optfn(current, idx):
        # current is ntimes only since it is common for all devices.
        # other parameters are ntimes x ndevices
        # broadcast current to ntimes x ndevices
        cur2 = np.broadcast_to(current[np.newaxis, :], (ndevices, len(current)))
        v = _iv_series_lambert_v_from_i(
            cur2, IL, I0, Rs,
            Rsh, a, neg_v_limit, ndevices, idx)
        # return string voltage
        return v.sum(axis=0)

    isc_result = find_root(
        isc_optfn,
        (min_isc, max_isc), args=(idx,))
    string_isc = isc_result.x  # 1d in ntimes

    # find Voc for string-level curves
    device_voc = _singlediode._lambertw_v_from_i(0., IL, I0, Rs, Rsh, a)
    string_voc = device_voc.sum(axis=0)

    # prepare for MPP calculation by computing voltages on a grid of currents
    # discretize current from string_isc down to 0 at each time step
    # Include each device's current at neg_v_limit since these will be the
    # curvature breakpoints in the series IV curve.
    # Leave a space for inserting MPP
    # currents is ntimes x npts-1, decreasing
    currents = _setup_currents(current_bkpts, string_isc, npts-1)

    # shape all arrays to be ndevices x ntimes x ncurrents
    cur3 = np.repeat(currents[np.newaxis, :, :], ndevices, axis=0)
    cur3, il, io, rs, rsh, a3 = np.broadcast_arrays(
        cur3, IL[:, :, np.newaxis], I0[:, :, np.newaxis], Rs[:, :, np.newaxis],
        Rsh[:, :, np.newaxis], a[:, :, np.newaxis])

    # solve voltages at each current for each IV curve
    device_voltages = _iv_series_lambert_v_from_i(
        cur3, il, io, rs, rsh, a3, neg_v_limit)

    # add voltage across devices to get string voltage
    # voltages is ntimes x ncurrents
    voltages = device_voltages.sum(axis=0)

    # objective function for MPP
    idx = np.arange(ntimes)
    def mpp_optfn(current, idx):
        # solve power at each current for each device
        # current is ntimes only since it is common for all devices.
        # other parameters are ntimes x ndevices
        # broadcast current to ntimes x ndevices
        cur2 = np.broadcast_to(current[np.newaxis, :],
                               (ndevices, len(current)))
        voltage = _iv_series_lambert_v_from_i(
            cur2, IL, I0, Rs, Rsh, a, neg_v_limit, ndevices, idx)

        # return negative of string power
        return current * -voltage.sum(axis=0)

    # mpp calculation
    idxs = np.argmax(currents * voltages, axis=1)
    # reversed since currents is decreasing
    idcs = idxs[:, np.newaxis] + [1, 0, -1]
    intervals = np.take_along_axis(currents, idcs, axis=1)
    init = (intervals[:, 0], intervals[:, 1], intervals[:, 2])

    idx = np.arange(ntimes)
    imp_result = find_minimum(mpp_optfn, init, args=(idx,))
    string_imp = imp_result.x

    # use string_imp to calculate string_vmp
    idx = np.arange(ntimes)
    cur2 = np.broadcast_to(string_imp[np.newaxis, :],
                           (ndevices, ntimes))
    device_vmp = _iv_series_lambert_v_from_i(
        cur2, IL, I0, Rs, Rsh, a, neg_v_limit, ndevices)
    string_vmp = device_vmp.sum(axis=0)

    # in case we decide that this function should return (voltage, current)
    voltages, currents = _insert_mpp(
        voltages, currents, string_vmp, string_imp)

    return string_isc, string_voc, string_imp, string_vmp
    

def _v_from_i_mismatch(currents, photocurrent, saturation_current,
                       resistance_series,
                       resistance_shunt, nNsVth, neg_v_limit=0.):
    r'''Solve the IV curve for series-connected devices where each device
    is described by the single diode equation.

    Uses a simplified model for reverse bias behavior, where current is
    unbounded at a constant reverse bias voltage ``neg_v_limit``.

    Input parameter ``photocurrent`` must have shape (devices, times).
    Input parameters ``saturation_current``, ``resistance_series``,
    ``resistance_shunt``, ``nNsVth`` may be arrays. If arrays, must be
    broadcastable to the shape of ``photocurrent``.

    Parameters
    ----------
    currents : numeric
        String current (A) at which string voltage is to be computed.
        Must have shape (times, currents).
    photocurrent : numeric
        photocurrent (A). Must have shape (devices, times).
    saturation_current : numeric
        saturation current (A). Must be broadcastable with photocurrent.
    resistance_series : numeric
        series resistance (ohm). Must be broadcastable with photocurrent.
    resistance_shunt : numeric
        shunt resistance (ohm). Must be broadcastable with photocurrent.
    nNsVth : numeric
        product of diode factor n, number of series cells Ns, and
        thermal voltage (Vth), (V). Must be broadcastable with photocurrent.
    neg_v_limit : float, optional
        Limit on reverse bias voltage, from cell breakdown voltage or reverse
        bias diode activation voltage (V). Should be negative. For example,
        if neg_v_limit=-5, then at V=-5 current is unbounded in the positive
        direction.

    Returns
    -------
    voltages : numeric
        String voltage (V) at the current points, shape (times).

    '''
    # target shape is ndevices x ntimes
    IL, I0, Rs, Rsh, a = \
        np.broadcast_arrays(photocurrent, saturation_current,
                            resistance_series, resistance_shunt, nNsVth)

    ndevices, ntimes = IL.shape

    # shape all arrays to be ndevices x ntimes x ncurrents
    cur3 = np.repeat(currents[np.newaxis, :, :], ndevices, axis=0)
    cur3, il, io, rs, rsh, a3 = np.broadcast_arrays(
        cur3, IL[:, :, np.newaxis], I0[:, :, np.newaxis], Rs[:, :, np.newaxis],
        Rsh[:, :, np.newaxis], a[:, :, np.newaxis])

    # solve voltages at each current for each IV curve
    # applies neg_v_limit
    device_voltages = _iv_series_lambert_v_from_i(
        cur3, il, io, rs, rsh, a3, neg_v_limit)

    # sum voltage across devices to get string voltage
    # voltages is ntimes x ncurrents
    voltages = device_voltages.sum(axis=0)

    return voltages
