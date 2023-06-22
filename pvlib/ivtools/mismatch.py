"""
The `mismatch` module contains functions for combining curves in
series using the single diode model.
"""

import numpy as np
from pvlib.singlediode import bishop88_i_from_v, bishop88_v_from_i


def prepare_curves(params, num_pts, breakdown_voltage=-0.5):
    """
    Calculates currents and voltages on IV curves with the given
    parameters, using the single diode equation. Returns values
    in format needed for inputs to :func:`combine_curves`.

    Parameters
    ----------
    params : array-like
        An array of parameters representing a set of :math:`n` IV
        curves. The array should contain :math:`n` rows and five
        columns. Each row contains the five parameters needed for a
        single curve. The parameters should be in the following order:

            photocurrent : numeric
                photo-generated current :math:`I_{L}` [A]

            saturation_current : numeric
                diode reverse saturation current :math:`I_{0}` [A]

            resistance_series : numeric
                series resistance :math:`R_{s}` [ohms]

            resistance_shunt : numeric
                shunt resistance :math:`R_{sh}` [ohms]

            nNsVth : numeric
                product of thermal voltage :math:`V_{th}` [V], diode
                ideality factor :math:`n`, and number of series cells
                :math:`N_{s}` [V]

    num_pts : int
        Number of points to compute for each IV curve.

    breakdown_voltage : float
        Vertical asymptote to use left of the y-axis. Any voltages that
        are smaller than ``breakdown_voltage`` will be replaced by it.

    Returns
    -------
    tuple
        currents : np.ndarray
            A 1D array of current values. Has shape (``num_pts``,).

        voltages : np.ndarray
            A 2D array of voltage values, where each row corresponds to
            a single IV curve. Has shape (:math:`n`, ``num_pts``), where
            :math:`n` is the number of IV curves passed in.

    Notes
    -----
    This function assumes a simplified reverse bias model. When using
    :func:`pvlib.singlediode.bishop88_v_from_i`, ``breakdown_factor`` is
    left at the default value, which excludes the reverse bias term from
    the model. Instead, any returned voltages that are less than
    ``breakdown_voltage`` are replaced by it, yielding a vertical line
    at ``breakdown_voltage``.

    """

    # in case params is a list containing scalars, add a dimension
    if len(np.shape(params)) == 1:
        params = params[np.newaxis,:]

    # get range of currents from 0 to max_isc
    max_isc = np.max(pvlib.singlediode.bishop88_i_from_v(0.0, *params.T,
              method='newton'))
    currents = np.linspace(0, max_isc, num=num_pts, endpoint=True)

    # prepare inputs for bishop88
    bishop_inputs = np.array([[currents[idx]]*len(params) for idx in
                    range(num_pts)])
    # each row of bishop_inputs contains n copies of a single current
    # value, where n is the number of curves being added together
    # there is a row for each current value

    # get voltages for each curve
    # (note: expecting to vectorize for both the inputted currents and
    # the inputted curve parameters)
    # transpose result so each row contains voltages for a single curve
    voltages = pvlib.singlediode.bishop88_v_from_i(bishop_inputs, *params.T,
               method='newton').T

    # any voltages in array that are smaller than breakdown_voltage are
    # clipped to be breakdown_voltage
    voltages = np.clip(voltages, a_min=breakdown_voltage, a_max=None)

    return currents, voltages


def combine_curves(currents, voltages):
    """
    Combines IV curves in series.

    Parameters
    ----------
    currents : array-like
        A 1D array-like object. Its first element must be zero, and it
        should be increasing.

    voltages : array-like
        A 2D array-like object. Each row corresponds to a single IV
        curve and contains the voltages for that curve that are
        associated to elements of ``currents``. Each row must be
        decreasing.

    Returns
    -------
    dict
        Contains the following keys:
            i_sc : scalar
                short circuit current of combined curve [A]

            v_oc : scalar
                open circuit voltage of combined curve [V]

            i_mp : scalar
                current at maximum power point of combined curve [A]

            v_mp : scalar
                voltage at maximum power point of combined curve [V]

            p_mp : scalar
                power at maximum power point of combined curve [W]

            i : np.ndarray
                currents of combined curve [A]

            v : np.ndarray
                voltages of combined curve [V]

    Notes
    -----
    If the combined curve does not cross the y-axis, then the last (and
    hence largest) current is returned for short circuit current.

    The maximum power point that is returned is the maximum power point
    of the dataset. Its accuracy will improve as more points are passed
    in.

    """

    currents = np.asarray(currents)
    voltages = np.asarray(voltages)
    assert currents.ndim == 1
    assert voltages.ndim == 2

    # for each current, add the associated voltages of all the curves
    # in our setup, this means summing each column of the voltage array
    combined_voltages = np.sum(clipped_voltages, axis=0)

    # combined_voltages should now have same shape as currents
    assert np.shape(combined_voltages) == np.shape(currents)

    # find max power point (in the dataset)
    powers = currents*combined_voltages
    mpp_idx = np.argmax(powers)
    vmp = combined_voltages[mpp_idx]
    imp = currents[mpp_idx]
    pmp = powers[mpp_idx]

    # we're assuming voltages are decreasing, so combined_voltages
    # should also be decreasing
    if not np.all(np.diff(combined_voltages) < 0):
        raise ValueError("Each row of voltages array must be decreasing.")
    # get isc
    # np.interp requires second argument is increasing, so flip
    # combined_voltages and currents
    isc = np.interp(0., np.flip(combined_voltages), np.flip(currents))

    # the first element of currents must be zero
    if currents[0] != 0:
        raise ValueError("First element of currents array must be zero.")
    # get voc
    voc = combined_voltages[0]

    return {'i_sc': isc, 'v_oc': voc, 'i_mp': imp, 'v_mp': vmp, 'p_mp': pmp,
            'i': currents, 'v': combined_voltages}

