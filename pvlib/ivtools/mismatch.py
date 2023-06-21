import pvlib
import numpy as np


def prepare_curves(params, num_pts, breakdown_voltage=-0.5):
    # params is an n x 5 array where each row corresponds to one of
    # the n curves' five defining parameters (photocurrent, saturation
    # current, series resistance, shunt resistance, nNsVth)
    # num_pts is the number of points you want calculated for each curve
    # (this will also be the number of points in the aggregate curve)
    # breakdown_voltage is the asymptote we use left of the y-axis

    # in case params is a list containing scalars, add a dimension
    if len(np.shape(params)) == 1:
        params = params[np.newaxis,:]

    # get range of currents from 0 to max_isc
    max_isc = np.max(pvlib.singlediode.bishop88_i_from_v(0.0, *params.T,
              method='newton')
    currents = np.linspace(0, max_isc, num=num_pts, endpoint=True)

    # prepare inputs for bishop88
    bishop_inputs = np.array([[currents[idx]]*len(params) for idx in
                    range(num_pts)])
    # each row of bishop_inputs contains n copies of a single current
    # value, where n is the number of curves being added together
    # there is a row for each current value

    # get voltages for each curve
    # transpose result so each row contains voltages for a single curve
    voltages = pvlib.singlediode.bishop88_v_from_i(bishop_inputs, *params.T,
               method='newton').T

    # any voltages in array that are smaller than breakdown_voltage are
    # clipped to be breakdown_voltage
    clipped_voltages = np.clip(voltages, a_min=breakdown_voltage, a_max=None)

    return currents, clipped_voltages


def combine_curves(currents, voltages):
    # currents is a 1D array that contains a range from 0 to max_isc
    # voltages is a 2D array where each row corresponds to the
    # associated voltages of a single curve (should be n by
    # len(currents), where n is the number of curves we're summing over)

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
        raise ValueError("Each row of voltages must be decreasing.")

    # get isc
    # np.interp requires second argument is increasing, so flip
    # combined_voltages and currents
    isc = np.interp(0., np.flip(combined_voltages), np.flip(currents))

    # get voc
    voc = combined_voltages[0]

    return {'i_sc': isc, 'v_oc': voc, 'i_mp': imp, 'v_mp': vmp, 'p_mp': pmp,
            'i': currents, 'v': combined_voltages}

