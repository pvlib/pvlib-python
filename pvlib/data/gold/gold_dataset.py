import datetime
import json
import warnings

# interval package: http://pyinterval.readthedocs.io/en/latest/guide.html, pip install pyinterval
from interval import interval, imath
import numpy
from pvlib import pvsystem
import scipy.constants as constants

# Do this after imports to skip import warnings
# warnings.filterwarnings('always')

# Some useful physical constants (CODATA: https://physics.nist.gov/cuu/Constants/)

# elementary charge, 1.6021766208e-19 Coulombs
elementary_charge_C = constants.value('elementary charge')

# Boltzmann constant, 1.38064852e-23 J/K
boltzmann_J_per_K = constants.value('Boltzmann constant')

CECMOD = pvsystem.retrieve_sam('cecmod')
G_W_m2_list = [100., 200., 400., 600., 800., 1000., 1100.]
T_degC_list = [15., 25., 50., 75.]
NUM_PTS = 21
# Approximate largest possible residual value, determined empirically
TOL_CURRENT_SUM_RESIDUAL_GOLD = 1.e-12
TOL_CURRENT_SUM_RESIDUAL_GOLD_INTERVAL = interval(
    [-TOL_CURRENT_SUM_RESIDUAL_GOLD, TOL_CURRENT_SUM_RESIDUAL_GOLD])

def generate_gold_dataset(json_filepath=None):
    '''
    TODO
    Returns the dataset in a python dictionary.
    '''

    # Create a reference-like x-Si cell based on parameters in Bishop's 1988 paper Fig. 6
    cell_area = 4.  # cm^2
    ideality_factor = 1.5
    T_K = 298.15  # 25 degC
    # Copy an existing mono-Si cell and None the entries we don't know
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
        boltzmann_J_per_K*T_K/elementary_charge_C
    mono_Si_cell.I_L_ref = cell_area*30.e-3
    mono_Si_cell.I_o_ref = cell_area*5500.e-12
    mono_Si_cell.R_s = cell_area*1.33
    mono_Si_cell.R_sh_ref = cell_area*750.
    mono_Si_cell.Adjust = None
    mono_Si_cell.gamma_r = None
    mono_Si_cell.Version = None
    mono_Si_cell.PTC = None

    # devices = [mono_Si_cell]
    devices = [mono_Si_cell, CECMOD.SunPower_SPR_E20_327,
               CECMOD.First_Solar_FS_495]

    # Initialize the output
    gold_dataset = {"version": '1.0.0',
                    "timestamp_utc_iso": datetime.datetime.utcnow().isoformat(),
                    "devices": []}

    # For each ideal, normal, or degraded device, create matrix of I-V curves
    for device in devices:
        # print("\n" + device.name)

        device_dict = {"name": device.name,
                       "Technology": device.Technology,
                       "N_s": device.N_s,
                       "iv_curves": []}

        # Choose bandgap parameters based on device technology
        if device.Technology == 'CdTe':
            EgRef = 1.475
            dEgdT = -0.0003
        elif device.Technology == 'Mono-c-Si' or device.Technology == 'Multi-c-Si':
            EgRef = 1.121
            dEgdT = -0.0002677
        else:
            raise ValueError('Unsupported device technology.')

        # Do all combinations of ideal device, regular device, and degraded device
        r_s_scale_grid, g_sh_scale_grid = numpy.meshgrid(
            [0., 1., 5.], [0., 1., 5.])

        for r_s_scale, g_sh_scale in zip(r_s_scale_grid.flat, g_sh_scale_grid.flat):
            # print("r_s_scale = {}, g_sh_scale = {}".format(r_s_scale, g_sh_scale))

            for G_W_m2 in G_W_m2_list:
                for T_degC in T_degC_list:
                    # print((G_W_m2, T_degC))

                    # Compute the model parameters at each irradiance-temperature combo
                    device_params = pvsystem.calcparams_desoto(
                        poa_global=G_W_m2, temp_cell=T_degC,
                        alpha_isc=device.alpha_sc, module_parameters=device,
                        EgRef=EgRef, dEgdT=dEgdT)
                    # More convienent variable names
                    i_l, i_0, r_s, r_sh, nNsVth = device_params
                    r_s = r_s_scale*r_s
                    g_sh = g_sh_scale/r_sh
                    # Reverse engineer the diode voltage range
                    i_sc = pvsystem.i_from_v(r_sh, r_s, nNsVth, 0., i_0, i_l)
                    v_oc = pvsystem.v_from_i(r_sh, r_s, nNsVth, 0., i_0, i_l)
                    # Go slightly less than Isc, even when r_s==0
                    v_d_min = i_sc*r_s - 0.02*v_oc
                    # Go slightly greater than Voc
                    v_d_max = 1.01*v_oc
                    v_d = v_d_min + \
                        (v_d_max - v_d_min) * \
                        numpy.log10(numpy.linspace(1., 10.**1.01, NUM_PTS))
                    i_gold = i_l - i_0*numpy.expm1(v_d/nNsVth) - g_sh*v_d
                    v_gold = v_d - r_s*i_gold

                    # Record interoperable hex values for vi_pair in zip(v_gold, i_gold):
                    device_dict["iv_curves"].append(
                        {
                            "G": G_W_m2.hex(),
                            "T": T_degC.hex(),
                            "i_l": i_l.hex(),
                            "i_0": i_0.hex(),
                            "r_s": r_s.hex(),
                            "g_sh": g_sh.hex(),
                            "nNsVth": nNsVth.hex(),
                            "v_gold": [v.hex() for v in v_gold.tolist()],
                            "i_gold": [i.hex() for i in i_gold.tolist()]
                        }
                    )

                    current_sum_residual_interval_list = \
                        [interval(i_l) - interval(i_0) *
                         imath.expm1((interval(vi_pair[0]) + interval(vi_pair[1])*interval(r_s))/interval(nNsVth)) -
                         interval(g_sh)*(interval(vi_pair[0]) + interval(vi_pair[1])*interval(r_s)) -
                         interval(vi_pair[1]) for vi_pair in zip(v_gold, i_gold)]

                    for v, i, current_sum_residual_interval in zip(v_gold, i_gold, current_sum_residual_interval_list):
                        if current_sum_residual_interval not in TOL_CURRENT_SUM_RESIDUAL_GOLD_INTERVAL:
                            raise ValueError("Residual interval {} not in {}, for (V,I)=({},{})".format(
                                current_sum_residual_interval, TOL_CURRENT_SUM_RESIDUAL_GOLD_INTERVAL, v, i))
                    
        gold_dataset["devices"].append(device_dict)

    # When requested, write dataset in json file
    with open(json_filepath, 'w') as fp:
        json.dump(gold_dataset, fp)

    return gold_dataset


def gauge_against_gold_dataset(func_dict, json_filepath=None):
    '''TODO'''

    # Function returns worse curve for each device for each function in func_dict
    output = []

    if json_filepath is None:
        gold_dataset = generate_gold_dataset()
    else:
        with open(json_filepath, 'r') as fp:
            gold_dataset = json.load(fp)

    version = gold_dataset["version"]
    print("{} at version {}".format(json_filepath, version))

    devices = gold_dataset["devices"]
    for device in devices:
        print(device["name"])
        worst_i_test_current_sum_max_abs_res = 0.
        worst_i_test_terminal_max_abs_res = 0.
        worst_v_test_current_sum_max_abs_res = 0.
        worst_v_test_terminal_max_abs_res = 0.

        device_dict = \
            {"name": device["name"],
             "Technology": device["Technology"],
             "N_s": device["N_s"]}
        if "i_from_v" in func_dict:
            device_dict["i_from_v"] = {"current_sum": None, "terminal": None}
        if "v_from_i" in func_dict:
            device_dict["v_from_i"] = {"current_sum": None, "terminal": None}

        for iv_curve in device["iv_curves"]:
            # Load parameters
            g_sh = float.fromhex(iv_curve["g_sh"])
            # JSON spec doesn't support IEEE 754 inf, so conductance saved instead
            if g_sh == 0.:
                r_sh = numpy.inf
            else:
                r_sh = 1./g_sh
            r_s = float.fromhex(iv_curve["r_s"])
            nNsVth = float.fromhex(iv_curve["nNsVth"])
            i_0 = float.fromhex(iv_curve["i_0"])
            i_l = float.fromhex(iv_curve["i_l"])
            v_gold = numpy.array([float.fromhex(v) for v in iv_curve["v_gold"]])
            i_gold = numpy.array([float.fromhex(i) for i in iv_curve["i_gold"]])

            if "i_from_v" in func_dict:
                # Test computed values against gold values
                i_test = func_dict["i_from_v"](r_sh, r_s, nNsVth, v_gold, i_0, i_l)

                i_test_current_sum_res = i_l - i_0 * \
                    numpy.expm1((v_gold + i_test*r_s)/nNsVth) - \
                    g_sh*(v_gold + i_test*r_s) - i_test
                i_test_current_sum_max_abs_res = numpy.amax(numpy.abs(i_test_current_sum_res))
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

                i_test_terminal_res = i_test - i_gold
                i_test_terminal_max_abs_res = numpy.amax(numpy.abs(i_test_terminal_res))
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

            if "v_from_i" in func_dict:
                v_test = func_dict["v_from_i"](r_sh, r_s, nNsVth, i_gold, i_0, i_l)
                
                v_test_current_sum_res = i_l - i_0 * \
                    numpy.expm1((v_test + i_gold*r_s)/nNsVth) - \
                    g_sh*(v_test + i_gold*r_s) - i_gold
                v_test_current_sum_max_abs_res = numpy.amax(numpy.abs(v_test_current_sum_res))
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
                    # argmax = numpy.argmax(numpy.abs(v_test_current_sum_res))
                    # print("(G,T) = ({},{})".format(device_dict["v_from_i"]["current_sum"]["G"], device_dict["v_from_i"]["current_sum"]["T"]))
                    # print(i_gold[argmax])
                    # print(v_gold[argmax])
                    # print(v_test[argmax])
                    # print(func_dict["v_from_i"](r_sh, r_s, nNsVth, i_gold[argmax], i_0, i_l))
                    # print("")

                v_test_terminal_res = v_test - v_gold
                v_test_terminal_max_abs_res = numpy.amax(numpy.abs(v_test_terminal_res))
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

        output.append(device_dict)

    return(output)


if __name__ == '__main__':

    import sys
    import pvlib.version

    print("python version: {}".format(sys.version))
    print("numpy version: {}".format(numpy.__version__))
    print("pvlib version: {}".format(pvlib.version.__version__))

    generate_gold_dataset("test.json")
    print(json.dumps(gauge_against_gold_dataset(
        {"i_from_v": pvsystem.i_from_v, "v_from_i": pvsystem.v_from_i}, json_filepath="test.json")))
