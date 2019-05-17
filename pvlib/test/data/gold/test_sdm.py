import os
import pprint
import uuid

from numpy.testing import assert_allclose

from conftest import requires_interval, requires_scipy
from pvlib import pvsystem
from pvlib.data.gold import sdm


def test_constants():
    # This catches an unexpected value change in any of the constants
    assert sdm.elementary_charge_C == 1.6021766208e-19
    assert sdm.boltzmann_J_per_K == 1.38064852e-23
    assert sdm.T_stc_degC == 25.
    assert sdm.T_stc_K == 298.15


def test_sum_current():
    # Test sum_current computation against committed gold dataset
    gold_dataset = sdm.convert_gold_dataset(
        sdm.load_gold_dataset(json_path=os.path.join(
            os.path.dirname(sdm.__file__), "sdm.json")))

    for device in gold_dataset["devices"]:
        for iv_curve in device["iv_curves"]:
            current_sum_res = sdm.sum_current(
                iv_curve["r_sh"], iv_curve["r_s"], iv_curve["nNsVth"],
                iv_curve["i_gold"], iv_curve["v_gold"], iv_curve["i_0"],
                iv_curve["i_l"])

            # Because interval computations are conservative, the atol here
            #  should be no larger than the interval tolerance used for gold
            #  dataset generation.
            assert_allclose(current_sum_res, 0., rtol=0., atol=1.e-13)


@requires_interval
def test_file_operations():
    # Test file saving and data exactness after json file save and reload
    gold_dataset = sdm.make_gold_dataset()
    # Use unique file names for saved test files
    file_name_json = "sdm_test_" + str(uuid.uuid4()) + ".json"
    file_name_csv = "sdm_test_" + str(uuid.uuid4()) + ".csv"
    sdm.save_gold_dataset(gold_dataset, json_path=file_name_json,
                          csv_path=file_name_csv)
    gold_dataset_reloaded = sdm.load_gold_dataset(json_path=file_name_json)
    assert gold_dataset == gold_dataset_reloaded
    os.remove(file_name_json)
    os.remove(file_name_csv)


@requires_scipy
def test_gauge_functions():
    # Test that pvsystem functions can be guaged
    # FUTURE Report results somewhere using Airspeed Velocity or the like
    json_path = os.path.join(os.path.dirname(sdm.__file__), "sdm.json")
    gold_dataset = sdm.load_gold_dataset(json_path=json_path)
    pprint.pprint(sdm.gauge_functions(gold_dataset, {
        "i_from_v": pvsystem.i_from_v,
        "v_from_i": pvsystem.v_from_i,
    }))
