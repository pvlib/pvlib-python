"""
Contains functions for solving for DC power in arrays with mismatched conditions.

"""
import numpy as np
import singlediode as _singlediode


def _iv_series_lambertw(photocurrent, saturation_current, resistance_series,
                        resistance_shunt, nNsVth, neg_v_limit=None,
                        delta_i=0.001):
    r'''Solve the IV curve for series-connected devices using a single diode
    equivalent circuit model.

    Uses a simplified model for reverse bias behavior, where current is
    unbounded at a constant reverse bias voltage ``neg_v_limit``.

    Input parameters photocurrent, saturation_current, resistance_series,
    resistance_shunt, nNsVth may be arrays. If arrays, all must be
    broadcastable to a common shape. The first dimension of each array
    is time. The 2nd dimension is devices in series.

    Parameters
    ----------
    photocurrent : numeric
        photocurrent (A).
    saturation_current : numeric
        saturation current (A).
    resistance_series : numeric
        series resistance (ohm).
    resistance_shunt : numeric
        shunt resistance (ohm).
    nNsVth : numeric
        product of diode factor n, number of series cells Ns, and
        thermal voltage (Vth), (V).
    neg_v_limit : float, optional
        Limit on reverse bias voltage, from cell breakdown voltage or reverse
        bias diode activation voltage (V). Should be negative. For example,
        if neg_v_limit=-5, then at V=-5 current is unbounded in the positive
        direction.
    delta_i : float, optional
        Width of interval used to discretize current (A).

    Returns
    -------
    None.

    '''
    # solve for Isc
    isc = _singlediode._lambertw_i_from_v(
        0., photocurrent, saturation_current, resistance_series,
        resistance_shunt, nNsVth)

    # discretize current from max(Isc) down to 0.
    currents = np.arange(isc.max(), 0., step=-delta_i)

    # shape all the arrays
    # target shape is ntimes x ndevices x ncurrents
    # use a dict so we can add axes using a loop
    params = {'photocurrent': photocurrent,
              'saturation_current': saturation_current,
              'resistance_series': resistance_series,
              'resistance_shunt': resistance_shunt,
              'nNsVth': nNsVth}

    for p in params:
        if not isinstance(params[p], np.ndarray):
            pass  # float, int
        if len(params[p].shape) == 1:
            params[p] = params[p][:, np.newaxis, np.newaxis]
        elif len(params[p].shape) == 2:
            params[p] = params[p][:, :, np.newaxis]
        else:
            pass  # already 3d

    il, io, rs, rsh, a = (params[p] for p in params)

    currents = currents[np.newaxis, np.newaxis, :]

    il, io, rs, rsh, a, currents = np.broadcast_arrays(
        il, io, rs, rsh, a, currents)

    # solve voltages at each current for each IV curve
    voltages = _singlediode._lambertw_v_from_i(
        currents, il, io, rs, rsh, a)

    # apply negative voltage limit
    if neg_v_limit is not None:
        voltages[voltages < neg_v_limit] = neg_v_limit

    # add voltage at common current to get series voltage
    voltage_sum = voltages.sum(axis=1)

    # drop currents dimension for devices
    currents = currents[:, 0, :]

    return voltage_sum, currents
