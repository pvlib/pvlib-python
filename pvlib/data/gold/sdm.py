"""
The ``sdm`` module contains functions for generating gold current-voltage
(I-V) data for the five-parameter single-diode model (SDM) and gauging
functions that compute I-V values in terms of accuracy and computation speed.

Run this module from the command line to generate a gold dataset in two
formats, pvlib/data/gold/sdm.json (lossles, machine-interoperable) and
pvlib/data/gold/sdm.csv (human readable and lossy) and then guage functions
against it that are compatible with the pvlib API.
"""

import copy
import datetime
import json
import os
import platform
import sys
from timeit import default_timer as timer

import numpy

from pvlib import pvsystem
import pvlib.version


# Useful constants for photovoltaic modeling that are versioned. Several
# constants are derived from scipy.constants v1.1.0, which are from
# standardized CODATA. See https://physics.nist.gov/cuu/Constants/. We do not
# import scipy because it is not required in the base pvlib installation. Units
# appear in the variable suffix to eliminate ambiguity.

# elementary charge, from scipy.constants.value('elementary charge')
elementary_charge_C = 1.6021766208e-19

# Boltzmann constant, from scipy.constants.value('Boltzmann constant')
boltzmann_J_per_K = 1.38064852e-23

# Define standard test condition (STC) temperature in degrees Celsius
T_stc_degC = 25.

# Define standard test condition (STC) temperature in Kelvin
T_stc_K = 273.15 + T_stc_degC


def sum_current(resistance_shunt, resistance_series, nNsVth, current, voltage,
                saturation_current, photocurrent):
    """
    Sum the current at the diode node of the single-diode model (SDM).

    The current sum equaling zero is the fundamental invariant of the SDM, as
    described in, e.g., De Soto et al. 2006 [1]. Reverse-voltage breakdown and
    bypass diodes are not included. Ideal device parameters are specified by
    resistance_shunt=numpy.inf and resistance_series=0.

    Inputs to this function can include any broadcast-compatible combination
    of scalar, numpy.ndarray, pandas.Series, and pandas.DataFrame, but it is
    the caller's responsibility to ensure that the arguments all are within
    their proper ranges.

    Parameters
    ----------
    resistance_shunt : numeric
        Shunt resistance in ohms under desired I-V curve conditions.
        Often abbreviated ``Rsh``. ``0 < Rsh <= numpy.inf``

    resistance_series : numeric
        Series resistance in ohms under desired I-V curve conditions.
        Often abbreviated ``Rs``. ``0 <= Rs < numpy.inf``

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (``n``), 2) the number of cells in series (``Ns``), and 3) the cell
        thermal voltage under the desired I-V curve conditions (``Vth``). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where ``k`` is Boltzmann's constant (J/K),
        ``temp_cell`` is the temperature of the p-n junction in Kelvin, and
        ``q`` is the charge of an electron (coulombs). ``0 < nNsVth``

    current : numeric
        The terminal current in amperes under desired I-V curve conditions.

    voltage : numeric
        The terminal voltage in volts under desired I-V curve conditions.

    saturation_current : numeric
        Diode saturation current in amperes under desired I-V curve
        conditions. Often abbreviated ``I_0``. ``0 < I_0``

    photocurrent : numeric
        Light-generated current (photocurrent) in amperes under desired
        I-V curve conditions. Often abbreviated ``I_L``. ``0 <= I_L``

    Returns
    -------
    current_sum : numeric

    References
    ----------
    [1] W. De Soto et al., "Improvement and validation of a model for
    photovoltaic array performance", Solar Energy, vol 80, pp. 78-88, 2006.
    """

    # Compute voltage at diode node first
    voltage_diode = voltage + resistance_series * current
    return photocurrent - saturation_current * \
        numpy.expm1(voltage_diode/nNsVth) - voltage_diode/resistance_shunt - \
        current


def bootstrap_si_device_bishop(N_s=1, cell_area_cm2=4., ideality_factor=1.5):
    """
    Create c-Si device based on cell parameters in Bishop's 1988 paper Fig. 6.

    Parameters are normalized to the cell area.
    """

    # Copy an existing mono-Si device and None entries we don't know or need
    device = copy.deepcopy(
        pvsystem.retrieve_sam('cecmod').SunPower_SPR_E20_327)
    device.name = 'Bishop_{}_{}_{}'.format(N_s, cell_area_cm2, ideality_factor)
    device.BIPV = 'N'
    device.Date = '6/20/1988'
    device.T_NOCT = None
    device.A_c = None
    device.N_s = N_s
    device.I_sc_ref = None
    device.V_oc_ref = None
    device.I_mp_ref = None
    device.V_mp_ref = None
    device.beta_oc = None
    device.a_ref = N_s * ideality_factor * boltzmann_J_per_K * T_stc_K / \
        elementary_charge_C  # Volt
    device.I_L_ref = cell_area_cm2 * 30.e-3  # Amp
    device.I_o_ref = cell_area_cm2 * 5500.e-12  # Amp
    device.R_s = N_s * cell_area_cm2 * 1.33  # Ohm
    device.R_sh_ref = cell_area_cm2 * 750. / N_s  # Ohm
    device.Adjust = None
    device.gamma_r = None
    device.Version = None
    device.PTC = None

    return device


def bootstrap_si_device_pvmismatch(N_s=1,
                                   cell_area_cm2=153.,
                                   ideality_factor=1.5):
    """
    Create a c-Si device based on default cell parameters in PVMismatch.

    Parameters are not normalized to the cell area.

    First diode taken from double-diode model used in PVMismatch, and default
    ideality factor is average of first and second diode ideality factors.
    https://github.com/SunPower/PVMismatch/blob/master/pvmismatch/pvmismatch_lib/pvcell.py
    """

    # TODO Check into normalizing for Ns

    # This should match the default value for cell_area_cm2 argument
    default_cell_area_cm2 = 153.

    # Copy an existing mono-Si device and None entries we don't know or need
    device = copy.deepcopy(
        pvsystem.retrieve_sam('cecmod').SunPower_SPR_E20_327)
    device.name = 'PVMismatch_{}_{}_{}'.format(N_s, cell_area_cm2,
                                               ideality_factor)
    device.BIPV = 'N'
    device.Date = '2/20/2018'
    device.T_NOCT = None
    device.A_c = None
    device.N_s = N_s
    device.I_sc_ref = None
    device.V_oc_ref = None
    device.I_mp_ref = None
    device.V_mp_ref = None
    device.beta_oc = None
    device.a_ref = N_s * ideality_factor * boltzmann_J_per_K * \
        T_stc_K / elementary_charge_C  # Volt
    device.I_L_ref = cell_area_cm2 * 6.3056 / default_cell_area_cm2  # Amp
    device.I_o_ref = cell_area_cm2 * 2.286188161253440e-11 / \
        default_cell_area_cm2  # Amp
    device.R_s = N_s * cell_area_cm2 * \
        0.004267236774264931/default_cell_area_cm2  # Ohm
    device.R_sh_ref = cell_area_cm2 * 10.01226369025448 / \
        default_cell_area_cm2 / N_s  # Ohm
    device.Adjust = None
    device.gamma_r = None
    device.Version = None
    device.PTC = None

    return device


def make_gold_dataset():
    """
    Make a gold dataset of I-V curves that accurately solve single-diode model.

    The set of I-V curves are generated from a range of PV devices: small c-Si
    cell, large c-Si cell, c-Si "mini-module", c-Si module, and CdTe module.
    Additionally, each device is ideal, nominal, or degraded in terms of its
    series and parallel resistances. I-V curves for each device are generated
    over a matrix of irradiance and temperature. The I-V curve data are
    returned in a python dict that is in a lossless, machine-interoperable
    format, which is the expected input to most of the other functions in this
    python module. (Other functions can convert this dict to a human-readable
    format.) An explict method of Bishop [1] is used to compute (V, I) pairs
    mostly in the first quadrant, and interval analysis guarantees the
    numerical computations up to a specified tolerance.

    New gold data can be added to the returned dict. Each change should be
    versioned in ``dataset_version``, following SemVer and maintaining
    backwards compatibility whenever possible. Be sure to update all dependent
    functions.

    Parameters
    ----------

    Returns
    -------
    gold_dataset : dict

    References
    ----------
    [1] J. W. Bishop, "Computer simulation of the effects of electrical
    mismatches in photovoltaic cell interconnection circuits", Solar Cells,
    vol 25, pp. 73-89, 1988.
    """

    # Load module inside function to avoid a global dependency requirement
    # http://pyinterval.readthedocs.io/en/latest/guide.html
    # Use `pip install pyinterval`, even in conda distro's
    from interval import interval, imath

    # Approximate largest possible absolute residual value, determined
    #  empirically
    tol = 1.e-12

    tol_interval = interval([-tol, tol])

    # Generate I-V curves, each with same number of points, over matrix of
    #  (G,T) combinations
    G_list = [100., 200., 400., 600., 800., 1000., 1100.]  # W/m^2
    T_list = [15., 25., 50., 75.]  # degC
    NUM_PTS = 21

    # The "base" devices, having ideal, normal, and degraded versions
    devices = [
        bootstrap_si_device_bishop(),
        bootstrap_si_device_bishop(N_s=8),
        bootstrap_si_device_pvmismatch(),
        pvsystem.retrieve_sam('cecmod').SunPower_SPR_E20_327,
        pvsystem.retrieve_sam('cecmod').First_Solar_FS_495
    ]

    # Initialize the output, recording metadata
    gold_dataset = {
        "timestamp_utc_iso": datetime.datetime.utcnow().isoformat(),
        "dataset_version": '1.0.0',  # dataset follows SemVer
        "python_version": sys.version,
        "numpy_version": numpy.__version__,
        "pvlib_version": pvlib.version.__version__,
        "platform_version": platform.platform(),
        "residual_interval_tolerance": tol,
        "units": {
            "G": 'W/m^2',
            "T": 'degC',
            "i_l": 'A',
            "i_0": 'A',
            "r_s": 'Ohm',
            "g_sh": 'S',
            "nNsVth": 'V',
            "v_gold": 'V',
            "i_gold": 'A'
        },
        "devices": []
    }

    # For each ideal, nominal, or degraded device, create matrix of I-V curves
    for device in devices:
        device_dict = {
            "name": device.name,
            "Technology": device.Technology,
            "N_s": device.N_s,
            "iv_curves": []
        }

        # Choose bandgap parameters based on device technology
        # Add new technologies if/when needed, should match a CEC module
        #  database type
        # These are NOT saved, but are versioned with the dataset version
        if device.Technology == 'Mono-c-Si' or \
           device.Technology == 'Multi-c-Si':
            EgRef = 1.121
            dEgdT = -0.0002677
        elif device.Technology == 'CdTe':
            EgRef = 1.475
            dEgdT = -0.0003
        else:
            raise ValueError('Unsupported device technology.')

        # Do all "flavor" combinations of ideal device, nominal device, and
        #  degraded device
        r_s_scale_grid, g_sh_scale_grid = numpy.meshgrid([0., 1., 5.],
                                                         [0., 1., 5.])

        # For all "flavors" of this device, create I-V curves over (G,T) matrix
        for r_s_scale, g_sh_scale in zip(r_s_scale_grid.flat,
                                         g_sh_scale_grid.flat):
            for G in G_list:
                for T in T_list:
                    # Compute the model parameters at each
                    #  irradiance-temperature combo
                    device_params = pvsystem.calcparams_desoto(
                        effective_irradiance=G,
                        temp_cell=T,
                        alpha_sc=device.alpha_sc,
                        a_ref=device.a_ref,
                        I_L_ref=device.I_L_ref,
                        I_o_ref=device.I_o_ref,
                        R_sh_ref=device.R_sh_ref,
                        R_s=device.R_s,
                        EgRef=EgRef,
                        dEgdT=dEgdT)
                    # More convienent variable names
                    i_l, i_0, r_s, r_sh, nNsVth = device_params
                    # Ideal, regular, or degraded series resistance
                    r_s = r_s_scale * r_s
                    # Ideal, regular, or degraded shunt conductance
                    # json spec excludes IEEE 754 inf, so use shunt
                    #  conductance instead of shunt resistance
                    g_sh = g_sh_scale / r_sh
                    # Reverse engineer the diode voltage range
                    i_sc = pvsystem.i_from_v(r_sh, r_s, nNsVth, 0., i_0,
                                             i_l)  # Not a gold value
                    v_oc = pvsystem.v_from_i(r_sh, r_s, nNsVth, 0., i_0,
                                             i_l)  # Not a gold value
                    # For min diode voltage, go slightly less than Isc, even
                    #  when r_s==0
                    v_d_min = i_sc * r_s - 0.02 * v_oc
                    # For max diode voltage, go slightly greater than Voc
                    v_d_max = 1.01 * v_oc
                    # More I-V points towards Voc
                    v_d = v_d_min + \
                        (v_d_max - v_d_min) * \
                        numpy.log10(numpy.linspace(1., 10.**1.01, NUM_PTS))
                    # Bishop's method is explicit and inherently vectorized
                    i_gold = i_l - i_0 * numpy.expm1(v_d / nNsVth) - g_sh * v_d
                    v_gold = v_d - r_s * i_gold

                    # Record lossless, machine-interoperable hex values for
                    #  the (V, I) pairs
                    device_dict["iv_curves"].append({
                        "G": G.hex(),
                        "T": T.hex(),
                        "i_l": i_l.hex(),
                        "i_0": i_0.hex(),
                        "r_s": r_s.hex(),
                        "g_sh": g_sh.hex(),
                        "nNsVth": nNsVth.hex(),
                        "v_gold": [v.hex() for v in v_gold.tolist()],
                        "i_gold": [i.hex() for i in i_gold.tolist()]
                    })

                    # Solve the current sum at diode node residual as a
                    #  numerically reliable interval
                    current_sum_residual_interval_list = \
                        [interval(i_l) - interval(i_0) *
                         imath.expm1((interval(vi_pair[0]) +
                                      interval(vi_pair[1])*interval(r_s)) /
                                     interval(nNsVth)) -
                         interval(g_sh)*(interval(vi_pair[0]) +
                         interval(vi_pair[1])*interval(r_s)) -
                         interval(vi_pair[1])
                         for vi_pair in zip(v_gold, i_gold)]

                    # Make sure the computed interval is within the specified
                    #  tolerance, report bad value in exception if not
                    for v, i, current_sum_residual_interval in zip(
                            v_gold, i_gold,
                            current_sum_residual_interval_list):
                        if current_sum_residual_interval not in tol_interval:
                            # Don't include entire curve in exception message
                            del device_dict["iv_curves"][-1]["v_gold"]
                            del device_dict["iv_curves"][-1]["i_gold"]
                            raise ValueError("Residual interval {} not in {}, \
                                 for (V, I)=({}, {}) at {}".format(
                                current_sum_residual_interval, tol_interval, v,
                                i, [
                                    float.fromhex(item)
                                    for item in device_dict["iv_curves"][-1]
                                ]))

        # Append this I-V curve to the list for this device
        gold_dataset["devices"].append(device_dict)

    # Return the dictionary, with floating point values in hex format
    return gold_dataset


def convert_gold_dataset(gold_dataset):
    """
    Convert gold dataset into machine-specific format.

    From a gold dataset in the lossless, machine-interoperable format, create
    and return a copy in a numpy-based machine-specific format for the current
    architecture that is compatible with the pvlib API.

    Parameters
    ----------
    gold_dataset : dict

    Returns
    -------
    gold_dataset_converted : dict
    """

    # Return a new dict instead of clobbering existing (uses more memory)
    gold_dataset_converted = copy.deepcopy(gold_dataset)

    # Convert the relevant header items
    # FUTURE Switch pvlib over to shunt conductance instead of resistance
    gold_dataset_converted["units"]["r_sh"] = "Ohm"
    del gold_dataset_converted["units"]["g_sh"]

    # Convert the values
    for device in gold_dataset_converted["devices"]:
        for iv_curve in device["iv_curves"]:
            # Load G, T, and 5 single-diode model parameters for each I-V curve
            # Floats converted from lossless, machine-interoperable hex format

            iv_curve["G"] = float.fromhex(iv_curve["G"])
            iv_curve["T"] = float.fromhex(iv_curve["T"])

            # json spec doesn't support IEEE 754 inf, so shunt conductance
            #  saved instead
            g_sh = float.fromhex(iv_curve["g_sh"])
            if g_sh == 0.:
                iv_curve["r_sh"] = numpy.inf
            else:
                iv_curve["r_sh"] = 1. / g_sh
            del iv_curve["g_sh"]

            iv_curve["r_s"] = float.fromhex(iv_curve["r_s"])
            iv_curve["nNsVth"] = float.fromhex(iv_curve["nNsVth"])
            iv_curve["i_0"] = float.fromhex(iv_curve["i_0"])
            iv_curve["i_l"] = float.fromhex(iv_curve["i_l"])
            iv_curve["v_gold"] = numpy.array(
                [float.fromhex(v) for v in iv_curve["v_gold"]])
            iv_curve["i_gold"] = numpy.array(
                [float.fromhex(i) for i in iv_curve["i_gold"]])

    return gold_dataset_converted


def pretty_print_gold_dataset(gold_dataset):
    """
    Create human-readable gold dataset.

    From a gold dataset in the lossless, machine-interoperable format, create
    and return the dataset as a human-readale, comma-separated-value (csv)
    string.

    Parameters
    ----------
    gold_dataset : dict

    Returns
    -------
    gold_dataset_csv : str

    """

    # Convert gold dataset from hex
    gold_dataset_converted = convert_gold_dataset(gold_dataset)

    num_pts = len(
        gold_dataset_converted["devices"][0]["iv_curves"][0]["v_gold"])

    # Initialize output
    gold_dataset_csv = gold_dataset_converted["timestamp_utc_iso"] + \
        ", " + gold_dataset_converted["dataset_version"] + "\n\n"
    gold_dataset_csv = gold_dataset_csv + "G (W/m^2), T (degC)" + "\n"
    gold_dataset_csv = gold_dataset_csv + \
        "r_sh (Ohm), r_s (Ohm), nNsVth (V), i_0 (A), i_l (A)" + "\n"
    gold_dataset_csv = gold_dataset_csv + \
        ", ".join(["v_gold_" + str(idx) +
                   " (V)" for idx in range(num_pts)]) + "\n"
    gold_dataset_csv = gold_dataset_csv + \
        ", ".join(["i_gold_" + str(idx) +
                   " (A)" for idx in range(num_pts)]) + "\n"
    gold_dataset_csv = gold_dataset_csv + "\n\n"

    for device in gold_dataset_converted["devices"]:
        for iv_curve in device["iv_curves"]:
            # Load G, T, and 5 single-diode model parameters for each I-V curve
            # Floats are converted from lossless, interchangeable hex format
            gold_dataset_csv = gold_dataset_csv + ", ".join(
                map(str, [iv_curve["G"], iv_curve["T"]])) + "\n"
            gold_dataset_csv = gold_dataset_csv + ", ".join(
                map(str, [
                    iv_curve["r_sh"], iv_curve["r_s"], iv_curve["nNsVth"],
                    iv_curve["i_0"], iv_curve["i_l"]
                ])) + "\n"
            gold_dataset_csv = gold_dataset_csv + ", ".join(
                map(str, iv_curve["v_gold"].tolist())) + "\n"
            gold_dataset_csv = gold_dataset_csv + ", ".join(
                map(str, iv_curve["i_gold"].tolist())) + "\n"

    return gold_dataset_csv


def save_gold_dataset(gold_dataset,
                      json_rel_path="sdm.json",
                      csv_rel_path="sdm.csv"):
    """
    Save gold dataset to file in two possible formats.

    The gold dataset dict is saved in a lossless, machine-interoperable json
    file and in a lossy, but human-readable, csv file.

    Relative file paths are used, which are relative to the current directory.
    Defaults are customizable. Set a file path to None to skip saving that
    file.

    Parameters
    ----------
    gold_dataset : dict

    json_rel_path : str
        Pass ``None`` to skip writing this file.

    csv_rel_path :str
        Pass ``None`` to skip writing this file.

    Returns
    -------

    """

    # When requested, write dataset in json file, skips when empty path string
    if json_rel_path:
        # This should normalize the path for the particular OS
        with open(os.path.abspath(json_rel_path), 'w') as fp:
            json.dump(gold_dataset, fp)

    # When requested, write dataset in csv file, skips when empty path string
    if csv_rel_path:
        # This should normalize the path for the particular OS
        with open(os.path.abspath(csv_rel_path), 'w') as fp:
            fp.write(pretty_print_gold_dataset(gold_dataset))


def load_gold_dataset(json_rel_path="sdm.json"):
    """
    Load gold dataset from disk.

    The gold dataset is loaded from a json file into a dict in lossless,
    machine-interoperable format. Pass this to convert_gold_dataset() in order
    to use the data with pvlib on a specific machine.

    Parameters
    ----------
    json_rel_path : str

    Returns
    -------
    gold_dataset : dict
    """

    # This should normalize the path for the particular OS
    with open(os.path.abspath(json_rel_path), 'r') as fp:
        return json.load(fp)


def gauge_gold_dataset(gold_dataset, test_func):
    """
    Gauge modeling functions for accuracy and speed against the gold dataset.

    For each modeling function passed in the test_func dictionary, this gauges
    and returns the I-V curve for each device with the least accuracy as
    compared to the gold dataset. Gauges are for both the residual of the
    current sum at the diode node and the residual in terminal I or V. Timing
    data and summary statistics are also returned for the ensemble of
    computations per function and per device.

    This also helps verify that a given function meets the pvlib interface.

    Parameters
    ----------
    gold_dataset : dict

    test_func : dict

    Returns
    -------
    results : list
        A per-device list of results with each function tested/benchmarked. The
worst computed I-V curve is returned along with performance timing information
and statistics for the device-ensemble of I-V curve computations (without any
repetitions).
    """

    # Convert gold dataset from hex
    gold_dataset = convert_gold_dataset(gold_dataset)

    # Follow SemVer and maintain backwards dataset compatibility whenever
    #  possible. Future code may require logic based on version loaded.
    print("Gold dataset version {}.".format(gold_dataset["dataset_version"]))

    # Initialize output
    results = []

    for device in gold_dataset["devices"]:
        # Reset the worst max abs residuals over all I-V curves per device
        worst_i_test_current_sum_max_abs_res = 0.
        worst_i_test_terminal_max_abs_res = 0.
        worst_v_test_current_sum_max_abs_res = 0.
        worst_v_test_terminal_max_abs_res = 0.

        # Initialize the worst results and benchmarks for this device
        # Note that a caller may choose not to test all the possible functions
        # FUTURE This dictionary can be expanded in a backwards compatible
        #  manner to support additional gold values and residuals for
        #  Isc, R@Isc, Ix, Pmp, Ixx, R@Voc, and Voc
        device_dict = \
            {"name": device["name"],
             "Technology": device["Technology"],
             "N_s": device["N_s"]}
        if "i_from_v" in test_func:
            device_dict["i_from_v"] = {
                "current_sum": None,
                "terminal": None,
                "performance": {
                    "time_s": {
                        "data": [],
                        "min": None,
                        "med": None,
                        "max": None
                    }
                }
            }
        if "v_from_i" in test_func:
            device_dict["v_from_i"] = {
                "current_sum": None,
                "terminal": None,
                "performance": {
                    "time_s": {
                        "data": [],
                        "min": None,
                        "med": None,
                        "max": None
                    }
                }
            }

        # Gauge each I-V curve for current sum and terminal residuals
        for iv_curve in device["iv_curves"]:
            # Check if caller wants to test an i_from_v function
            if "i_from_v" in test_func:
                # Compute test values using provided test function that meets
                #  pvlib API
                start = timer()
                i_test = test_func["i_from_v"](
                    iv_curve["r_sh"], iv_curve["r_s"], iv_curve["nNsVth"],
                    iv_curve["v_gold"], iv_curve["i_0"], iv_curve["i_l"])
                device_dict["i_from_v"]["performance"]["time_s"][
                    "data"].append(timer() - start)

                # Gauge computed values against gold values for sum of
                #  currents residual at diode node
                i_test_current_sum_res = sum_current(
                    iv_curve["r_sh"], iv_curve["r_s"], iv_curve["nNsVth"],
                    i_test, iv_curve["v_gold"], iv_curve["i_0"],
                    iv_curve["i_l"])
                i_test_current_sum_max_abs_res = numpy.amax(
                    numpy.abs(i_test_current_sum_res))
                if worst_i_test_current_sum_max_abs_res < \
                   i_test_current_sum_max_abs_res:
                    worst_i_test_current_sum_max_abs_res = \
                        i_test_current_sum_max_abs_res
                    device_dict["i_from_v"]["current_sum"] = \
                        {
                            "G": iv_curve["G"],
                            "T": iv_curve["T"],
                            "i_l": iv_curve["i_l"],
                            "i_0": iv_curve["i_0"],
                            "r_s": iv_curve["r_s"],
                            "r_sh": iv_curve["r_sh"],
                            "nNsVth": iv_curve["nNsVth"],
                            "v_gold": iv_curve["v_gold"].tolist(),
                            "i_gold": iv_curve["i_gold"].tolist(),
                            "i_test": i_test.tolist(),
                            "residual": i_test_current_sum_res.tolist(),
                            "max_abs_residual":
                                i_test_current_sum_max_abs_res.tolist(),
                    }

                # Gauge computed values against gold values for difference
                #  residual at terminal
                i_test_terminal_res = i_test - iv_curve["i_gold"]
                i_test_terminal_max_abs_res = numpy.amax(
                    numpy.abs(i_test_terminal_res))
                if worst_i_test_terminal_max_abs_res < \
                   i_test_terminal_max_abs_res:
                    worst_i_test_terminal_max_abs_res = \
                        i_test_terminal_max_abs_res
                    device_dict["i_from_v"]["terminal"] = \
                        {
                            "G": iv_curve["G"],
                            "T": iv_curve["T"],
                            "i_l": iv_curve["i_l"],
                            "i_0": iv_curve["i_0"],
                            "r_s": iv_curve["r_s"],
                            "r_sh": iv_curve["r_sh"],
                            "nNsVth": iv_curve["nNsVth"],
                            "v_gold": iv_curve["v_gold"].tolist(),
                            "i_gold": iv_curve["i_gold"].tolist(),
                            "i_test": i_test.tolist(),
                            "residual": i_test_terminal_res.tolist(),
                            "max_abs_residual":
                                i_test_terminal_max_abs_res.tolist(),
                    }

            # Check if caller wants to test a v_from_i function
            if "v_from_i" in test_func:
                # Compute test values using provided test function that meets
                #  pvlib API
                start = timer()
                v_test = test_func["v_from_i"](
                    iv_curve["r_sh"], iv_curve["r_s"], iv_curve["nNsVth"],
                    iv_curve["i_gold"], iv_curve["i_0"], iv_curve["i_l"])
                device_dict["v_from_i"]["performance"]["time_s"][
                    "data"].append(timer() - start)

                # Gauge computed values against gold values for sum of
                #  currents residual at diode node
                v_test_current_sum_res = sum_current(
                    iv_curve["r_sh"], iv_curve["r_s"], iv_curve["nNsVth"],
                    iv_curve["i_gold"], v_test, iv_curve["i_0"],
                    iv_curve["i_l"])
                v_test_current_sum_max_abs_res = numpy.amax(
                    numpy.abs(v_test_current_sum_res))
                if worst_v_test_current_sum_max_abs_res < \
                   v_test_current_sum_max_abs_res:
                    worst_v_test_current_sum_max_abs_res = \
                        v_test_current_sum_max_abs_res
                    device_dict["v_from_i"]["current_sum"] = \
                        {
                            "G": iv_curve["G"],
                            "T": iv_curve["T"],
                            "i_l": iv_curve["i_l"],
                            "i_0": iv_curve["i_0"],
                            "r_s": iv_curve["r_s"],
                            "r_sh": iv_curve["r_sh"],
                            "nNsVth": iv_curve["nNsVth"],
                            "v_gold": iv_curve["v_gold"].tolist(),
                            "i_gold": iv_curve["i_gold"].tolist(),
                            "v_test": v_test.tolist(),
                            "residual": v_test_current_sum_res.tolist(),
                            "max_abs_residual":
                                v_test_current_sum_max_abs_res.tolist(),
                    }

                # Gauge computed values against gold values for difference
                #  residual at terminal
                v_test_terminal_res = v_test - iv_curve["v_gold"]
                v_test_terminal_max_abs_res = numpy.amax(
                    numpy.abs(v_test_terminal_res))
                if worst_v_test_terminal_max_abs_res < \
                   v_test_terminal_max_abs_res:
                    worst_v_test_terminal_max_abs_res = \
                        v_test_terminal_max_abs_res
                    device_dict["v_from_i"]["terminal"] = \
                        {
                            "G": iv_curve["G"],
                            "T": iv_curve["T"],
                            "i_l": iv_curve["i_l"],
                            "i_0": iv_curve["i_0"],
                            "r_s": iv_curve["r_s"],
                            "r_sh": iv_curve["r_sh"],
                            "nNsVth": iv_curve["nNsVth"],
                            "v_gold": iv_curve["v_gold"].tolist(),
                            "i_gold": iv_curve["i_gold"].tolist(),
                            "v_test": v_test.tolist(),
                            "residual": v_test_terminal_res.tolist(),
                            "max_abs_residual":
                                v_test_terminal_max_abs_res.tolist(),
                    }

        # Compute summary performance statistics for this device
        if "i_from_v" in test_func:
            device_dict["i_from_v"]["performance"]["time_s"]["min"] = \
                min(device_dict["i_from_v"]["performance"]["time_s"]["data"])
            device_dict["i_from_v"]["performance"]["time_s"]["med"] = \
                numpy.median(device_dict["i_from_v"]
                             ["performance"]["time_s"]["data"])
            device_dict["i_from_v"]["performance"]["time_s"]["max"] = \
                max(device_dict["i_from_v"]["performance"]["time_s"]["data"])
            device_dict["i_from_v"]["performance"]["time_s"]["avg"] = \
                numpy.average(device_dict["i_from_v"]
                              ["performance"]["time_s"]["data"])
            device_dict["i_from_v"]["performance"]["time_s"]["std"] = \
                numpy.std(device_dict["i_from_v"]
                          ["performance"]["time_s"]["data"], ddof=1)

        if "v_from_i" in test_func:
            device_dict["v_from_i"]["performance"]["time_s"]["min"] = \
                min(device_dict["v_from_i"]["performance"]["time_s"]["data"])
            device_dict["v_from_i"]["performance"]["time_s"]["med"] = \
                numpy.median(device_dict["v_from_i"]
                             ["performance"]["time_s"]["data"])
            device_dict["v_from_i"]["performance"]["time_s"]["max"] = \
                max(device_dict["v_from_i"]["performance"]["time_s"]["data"])
            device_dict["v_from_i"]["performance"]["time_s"]["avg"] = \
                numpy.average(device_dict["v_from_i"]
                              ["performance"]["time_s"]["data"])
            device_dict["v_from_i"]["performance"]["time_s"]["std"] = \
                numpy.std(device_dict["v_from_i"]
                          ["performance"]["time_s"]["data"], ddof=1)

        # Append the worst results and the benchmarking for this device to the
        #  output list of devices
        # For convienent post processing and analysis, floats are NOT in hex
        #  format
        results.append(device_dict)

    # Unit test consumers should be able to pass/fail based on the worst
    #  results returned for each device
    return results


# Command line option
if __name__ == '__main__':

    gold_dataset = make_gold_dataset()

    # File locations are robust to running script from different directories
    rel_path = os.path.dirname(__file__)
    json_rel_path = os.path.join(rel_path, "sdm.json")
    csv_rel_path = os.path.join(rel_path, "sdm.csv")

    save_gold_dataset(
        gold_dataset, json_rel_path=json_rel_path, csv_rel_path=csv_rel_path)

    import pprint
    pprint.pprint(
        gauge_gold_dataset(gold_dataset, {
            "i_from_v": pvsystem.i_from_v,
            "v_from_i": pvsystem.v_from_i,
        }))
