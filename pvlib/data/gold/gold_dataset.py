import datetime
import json
import sys

import numpy
from pvlib import pvsystem
import pvlib.version
import scipy
import scipy.constants as constants

# Some useful physical constants (CODATA: https://physics.nist.gov/cuu/Constants/)

# elementary charge, 1.6021766208e-19 C
elementary_charge_C = constants.value('elementary charge')

# Boltzmann constant, 1.38064852e-23 J/K
boltzmann_J_per_K = constants.value('Boltzmann constant')

# Define standard test condition (STC) temperature
T_STC_K = 298.15  # K, equivalent to 25 degC

# Approximate largest possible absolute residual value, determined empirically
TOL_CURRENT_SUM_RESIDUAL_GOLD = 1.e-12

# Generate I-V curves, each with same number of points, over matrix of (G,T) combinations
G_list = [100., 200., 400., 600., 800., 1000., 1100.]  # W/m^2
T_list = [15., 25., 50., 75.]  # degC
NUM_PTS = 21


def generate_gold_dataset(json_filepath=None):
    '''
    TODO
    Returns the dataset in a python dictionary.
    Optionally writes result in lossless interoperable format to a json file.
    '''

    # Load module inside function to avoid a global dependency requirement
    # http://pyinterval.readthedocs.io/en/latest/guide.html
    # pip install pyinterval, even in conda distro's
    from interval import interval, imath

    TOL_CURRENT_SUM_RESIDUAL_GOLD_INTERVAL = interval(
        [-TOL_CURRENT_SUM_RESIDUAL_GOLD, TOL_CURRENT_SUM_RESIDUAL_GOLD])

    CECMOD = pvsystem.retrieve_sam('cecmod')

    # Create a reference-like x-Si cell based on limited parameters in Bishop's 1988 paper Fig. 6
    cell_area = 4.  # cm^2
    ideality_factor = 1.5

    # Copy an existing mono-Si cell and None the entries we don't know or need
    mono_Si_cell = CECMOD.SunPower_SPR_E20_327
    mono_Si_cell.name = 'Bishop_ref_cell'
    mono_Si_cell.BIPV = 'N'
    mono_Si_cell.Date = '6/20/1988'
    mono_Si_cell.T_NOCT = None
    mono_Si_cell.A_c = None
    mono_Si_cell.N_s = 1
    mono_Si_cell.I_sc_ref = cell_area*30.e-3  # Amp
    mono_Si_cell.V_oc_ref = None
    mono_Si_cell.I_mp_ref = None
    mono_Si_cell.V_mp_ref = None
    mono_Si_cell.beta_oc = None
    mono_Si_cell.a_ref = mono_Si_cell.N_s*ideality_factor * \
        boltzmann_J_per_K*T_STC_K/elementary_charge_C  # Volt
    mono_Si_cell.I_L_ref = cell_area*30.e-3  # Amp
    mono_Si_cell.I_o_ref = cell_area*5500.e-12  # Amp
    mono_Si_cell.R_s = cell_area*1.33  # Ohm
    mono_Si_cell.R_sh_ref = cell_area*750.  # Ohm
    mono_Si_cell.Adjust = None
    mono_Si_cell.gamma_r = None
    mono_Si_cell.Version = None
    mono_Si_cell.PTC = None

    # The "base" devices, of which there are ideal, normal, and degraded versions
    devices = [mono_Si_cell, CECMOD.SunPower_SPR_E20_327,
               CECMOD.First_Solar_FS_495]

    # Initialize the output
    gold_dataset = {"version": '1.0.0',
                    "python": sys.version,
                    "numpy": numpy.__version__,
                    "scipy": scipy.__version__,
                    "pvlib": pvlib.version.__version__,
                    "timestamp_utc_iso": datetime.datetime.utcnow().isoformat(),
                    "units": {"G": 'W/m^2',
                              "T": 'degC',
                              "i_l": 'A',
                              "i_0": 'A',
                              "r_s": 'Ohm',
                              "g_sh": 'S',
                              "nNsVth": 'V',
                              "v_gold": 'V',
                              "i_gold": 'A'},
                    "devices": []}

    # For each ideal, normal, or degraded device, create matrix of I-V curves
    for device in devices:
        device_dict = {"name": device.name,
                       "Technology": device.Technology,
                       "N_s": device.N_s,
                       "iv_curves": []}

        # Choose bandgap parameters based on device technology
        # Add new technologies if/when needed, should match a CEC module database type
        # These are NOT saved, but should be versioned
        if device.Technology == 'CdTe':
            EgRef = 1.475
            dEgdT = -0.0003
        elif device.Technology == 'Mono-c-Si' or device.Technology == 'Multi-c-Si':
            EgRef = 1.121
            dEgdT = -0.0002677
        else:
            raise ValueError('Unsupported device technology.')

        # Do all "flavor" combinations of ideal device, regular device, and degraded device
        r_s_scale_grid, g_sh_scale_grid = numpy.meshgrid(
            [0., 1., 5.], [0., 1., 5.])

        # For all "flavors" of this device, create I-V curves over the (G,T) matrix
        for r_s_scale, g_sh_scale in zip(r_s_scale_grid.flat, g_sh_scale_grid.flat):
            for G in G_list:
                for T in T_list:
                    # Compute the model parameters at each irradiance-temperature combo
                    device_params = pvsystem.calcparams_desoto(
                        poa_global=G, temp_cell=T,
                        alpha_isc=device.alpha_sc, module_parameters=device,
                        EgRef=EgRef, dEgdT=dEgdT)
                    # More convienent variable names
                    i_l, i_0, r_s, r_sh, nNsVth = device_params
                    # Ideal, regular, or degraded series resistance
                    r_s = r_s_scale*r_s
                    # Ideal, regular, or degraded shunt conductance
                    # json spec excludes IEEE 754 inf, so use shunt conductance instead of shunt resistance
                    g_sh = g_sh_scale/r_sh
                    # Reverse engineer the diode voltage range
                    i_sc = pvsystem.i_from_v(
                        r_sh, r_s, nNsVth, 0., i_0, i_l)  # Not a gold value
                    v_oc = pvsystem.v_from_i(
                        r_sh, r_s, nNsVth, 0., i_0, i_l)  # Not a gold value
                    # For min diode voltage, go slightly less than Isc, even when r_s==0
                    v_d_min = i_sc*r_s - 0.02*v_oc
                    # For max diode voltage, go slightly greater than Voc
                    v_d_max = 1.01*v_oc
                    # More I-V points towards Voc
                    v_d = v_d_min + \
                        (v_d_max - v_d_min) * \
                        numpy.log10(numpy.linspace(1., 10.**1.01, NUM_PTS))
                    # Bishop's method is explicit and vectorized
                    i_gold = i_l - i_0*numpy.expm1(v_d/nNsVth) - g_sh*v_d
                    v_gold = v_d - r_s*i_gold

                    # Record interoperable hex values for vi_pair in zip(v_gold, i_gold):
                    device_dict["iv_curves"].append(
                        {
                            "G": G.hex(),
                            "T": T.hex(),
                            "i_l": i_l.hex(),
                            "i_0": i_0.hex(),
                            "r_s": r_s.hex(),
                            "g_sh": g_sh.hex(),
                            "nNsVth": nNsVth.hex(),
                            "v_gold": [v.hex() for v in v_gold.tolist()],
                            "i_gold": [i.hex() for i in i_gold.tolist()]
                        }
                    )

                    # Solve the current sum at diode node residual as a numerically reliable interval
                    current_sum_residual_interval_list = \
                        [interval(i_l) - interval(i_0) *
                         imath.expm1((interval(vi_pair[0]) + interval(vi_pair[1])*interval(r_s))/interval(nNsVth)) -
                         interval(g_sh)*(interval(vi_pair[0]) + interval(vi_pair[1])*interval(r_s)) -
                         interval(vi_pair[1]) for vi_pair in zip(v_gold, i_gold)]

                    # Make sure the computed interval is within the specified tolerance
                    for v, i, current_sum_residual_interval in zip(v_gold, i_gold, current_sum_residual_interval_list):
                        if current_sum_residual_interval not in TOL_CURRENT_SUM_RESIDUAL_GOLD_INTERVAL:
                            raise ValueError("Residual interval {} not in {}, for (V,I)=({},{})".format(
                                current_sum_residual_interval, TOL_CURRENT_SUM_RESIDUAL_GOLD_INTERVAL, v, i))

        # Append this I-V curve to the list for this device
        gold_dataset["devices"].append(device_dict)

    # When requested, write dataset in json file, skips empty path strings too
    if json_filepath:
        with open(json_filepath, 'w') as fp:
            json.dump(gold_dataset, fp)

    # Return the dictionary, but note floating point values in hex format
    return gold_dataset


def gauge_against_gold_dataset(test_func_dict, json_filepath=None):
    '''
    TODO
    Function returns worst curve for each device for each function in test_func_dict
    '''

    # Initialize output
    worst_results = []

    # Generate gold dataset on the fly if not specified as file
    if not json_filepath:
        gold_dataset = generate_gold_dataset()
    else:
        with open(json_filepath, 'r') as fp:
            gold_dataset = json.load(fp)

    # Follow SemVer and maintain backwards compatibility whenever possible
    version = gold_dataset["version"]
    print("{} at version {}".format(json_filepath, version))

    devices = gold_dataset["devices"]
    for device in devices:
        # Reset the worst max abs residuals over all I-V curves per device
        worst_i_test_current_sum_max_abs_res = 0.
        worst_i_test_terminal_max_abs_res = 0.
        worst_v_test_current_sum_max_abs_res = 0.
        worst_v_test_terminal_max_abs_res = 0.

        # Initialize the worst results for this device
        # Note that a caller may choose not to test all the possible functions
        # FUTURE This dictionary can be expanded in a backwards compatible manner to support
        #  additional gold values and residuals for Isc, R@Isc, Ix, Pmp, Ixx, R@Voc, and Voc
        device_dict = \
            {"name": device["name"],
             "Technology": device["Technology"],
             "N_s": device["N_s"]}
        if "i_from_v" in test_func_dict:
            device_dict["i_from_v"] = {"current_sum": None, "terminal": None}
        if "v_from_i" in test_func_dict:
            device_dict["v_from_i"] = {"current_sum": None, "terminal": None}

        # Gauge each I-V curve for current sum and terminal residuals
        for iv_curve in device["iv_curves"]:
            # Load 5 single diode model parameters for each I-V curve
            # Floats are saved in lossless, interchangeable hex format
            g_sh = float.fromhex(iv_curve["g_sh"])
            # json spec doesn't support IEEE 754 inf, so shunt conductance saved instead
            # FUTURE Switch pvlib over to using shunt conductance instead of shunt resistance
            if g_sh == 0.:
                r_sh = numpy.inf
            else:
                r_sh = 1./g_sh
            r_s = float.fromhex(iv_curve["r_s"])
            nNsVth = float.fromhex(iv_curve["nNsVth"])
            i_0 = float.fromhex(iv_curve["i_0"])
            i_l = float.fromhex(iv_curve["i_l"])
            v_gold = numpy.array([float.fromhex(v)
                                  for v in iv_curve["v_gold"]])
            i_gold = numpy.array([float.fromhex(i)
                                  for i in iv_curve["i_gold"]])

            # Check if caller wants to test an i_from_v function
            if "i_from_v" in test_func_dict:
                # Compute test values using provided test function that meets pvlib API
                i_test = test_func_dict["i_from_v"](
                    r_sh, r_s, nNsVth, v_gold, i_0, i_l)

                # Gauge computed values against gold values for sum of currents residual at diode node
                i_test_current_sum_res = i_l - i_0 * \
                    numpy.expm1((v_gold + i_test*r_s)/nNsVth) - \
                    g_sh*(v_gold + i_test*r_s) - i_test
                i_test_current_sum_max_abs_res = numpy.amax(
                    numpy.abs(i_test_current_sum_res))
                if worst_i_test_current_sum_max_abs_res < i_test_current_sum_max_abs_res:
                    worst_i_test_current_sum_max_abs_res = i_test_current_sum_max_abs_res
                    device_dict["i_from_v"]["current_sum"] = \
                        {
                        "G": float.fromhex(iv_curve["G"]),
                        "T": float.fromhex(iv_curve["T"]),
                        "i_l": i_l,
                        "i_0": i_0,
                        "r_s": r_s,
                        "r_sh": r_sh,
                        "nNsVth": nNsVth,
                        "v_gold": v_gold.tolist(),
                        "i_gold": i_gold.tolist(),
                        "i_test": i_test.tolist(),
                        "residual": i_test_current_sum_res.tolist(),
                        "max_abs_residual": i_test_current_sum_max_abs_res.tolist()
                    }

                # Gauge computed values against gold values for difference residual at terminal
                i_test_terminal_res = i_test - i_gold
                i_test_terminal_max_abs_res = numpy.amax(
                    numpy.abs(i_test_terminal_res))
                if worst_i_test_terminal_max_abs_res < i_test_terminal_max_abs_res:
                    worst_i_test_terminal_max_abs_res = i_test_terminal_max_abs_res
                    device_dict["i_from_v"]["terminal"] = \
                        {
                        "G": float.fromhex(iv_curve["G"]),
                        "T": float.fromhex(iv_curve["T"]),
                        "i_l": i_l,
                        "i_0": i_0,
                        "r_s": r_s,
                        "r_sh": r_sh,
                        "nNsVth": nNsVth,
                        "v_gold": v_gold.tolist(),
                        "i_gold": i_gold.tolist(),
                        "i_test": i_test.tolist(),
                        "residual": i_test_terminal_res.tolist(),
                        "max_abs_residual": i_test_terminal_max_abs_res.tolist()
                    }

            # Check if caller wants to test a v_from_i function
            if "v_from_i" in test_func_dict:
                # Compute test values using provided test function that meets pvlib API
                v_test = test_func_dict["v_from_i"](
                    r_sh, r_s, nNsVth, i_gold, i_0, i_l)

                # Gauge computed values against gold values for sum of currents residual at diode node
                v_test_current_sum_res = i_l - i_0 * \
                    numpy.expm1((v_test + i_gold*r_s)/nNsVth) - \
                    g_sh*(v_test + i_gold*r_s) - i_gold
                v_test_current_sum_max_abs_res = numpy.amax(
                    numpy.abs(v_test_current_sum_res))
                if worst_v_test_current_sum_max_abs_res < v_test_current_sum_max_abs_res:
                    worst_v_test_current_sum_max_abs_res = v_test_current_sum_max_abs_res
                    device_dict["v_from_i"]["current_sum"] = \
                        {
                        "G": float.fromhex(iv_curve["G"]),
                        "T": float.fromhex(iv_curve["T"]),
                        "i_l": i_l,
                        "i_0": i_0,
                        "r_s": r_s,
                        "r_sh": r_sh,
                        "nNsVth": nNsVth,
                        "v_gold": v_gold.tolist(),
                        "i_gold": i_gold.tolist(),
                        "v_test": v_test.tolist(),
                        "residual": v_test_current_sum_res.tolist(),
                        "max_abs_residual": v_test_current_sum_max_abs_res.tolist()
                    }

                # Gauge computed values against gold values for difference residual at terminal
                v_test_terminal_res = v_test - v_gold
                v_test_terminal_max_abs_res = numpy.amax(
                    numpy.abs(v_test_terminal_res))
                if worst_v_test_terminal_max_abs_res < v_test_terminal_max_abs_res:
                    worst_v_test_terminal_max_abs_res = v_test_terminal_max_abs_res
                    device_dict["v_from_i"]["terminal"] = \
                        {
                        "G": float.fromhex(iv_curve["G"]),
                        "T": float.fromhex(iv_curve["T"]),
                        "i_l": i_l,
                        "i_0": i_0,
                        "r_s": r_s,
                        "r_sh": r_sh,
                        "nNsVth": nNsVth,
                        "v_gold": v_gold.tolist(),
                        "i_gold": i_gold.tolist(),
                        "v_test": v_test.tolist(),
                        "residual": v_test_terminal_res.tolist(),
                        "max_abs_residual": v_test_terminal_max_abs_res.tolist()
                    }

        # Append the worst results found for this device to the output dict
        # For convienent post processing and analysis, floats are NOT in hex format
        worst_results.append(device_dict)

    # Unit test consumers should be able to pass/fail based on the worst results returned for each device
    return(worst_results)


# TODO Determine the final functionality here for command line operation
# TODO What about logging warnings along the way?
if __name__ == '__main__':

    generate_gold_dataset("gold_dataset.json")
    print(json.dumps(gauge_against_gold_dataset(
        {"i_from_v": pvsystem.i_from_v, "v_from_i": pvsystem.v_from_i}, json_filepath="gold_dataset.json")))
