import os
import pprint
import uuid

from numpy.testing import assert_allclose

from conftest import requires_interval, requires_scipy
from pvlib import pvsystem
from pvlib.data.gold import sdm


def test_sum_current():
    # Test sum_current computation against comitted gold dataset
    gold_dataset = sdm.convert_gold_dataset(
        sdm.load_gold_dataset(json_rel_path=os.path.join(
            os.path.dirname(sdm.__file__), "sdm.json")))

    for device in gold_dataset["devices"]:
        for iv_curve in device["iv_curves"]:
            current_sum_res = sdm.sum_current(
                iv_curve["r_sh"], iv_curve["r_s"], iv_curve["nNsVth"],
                iv_curve["i_gold"], iv_curve["v_gold"], iv_curve["i_0"],
                iv_curve["i_l"])

            # atol typically smaller that the interval tolerance used for gold
            #  dataset generation
            assert_allclose(current_sum_res, 0., rtol=0., atol=1.e-13)


@requires_interval
def test_file_operations():
    # Test file saves and data exactness after json file save and reload
    gold_dataset = sdm.make_gold_dataset()
    file_name_json = "sdm_test_" + str(uuid.uuid4()) + ".json"
    file_name_csv = "sdm_test_" + str(uuid.uuid4()) + ".csv"
    sdm.save_gold_dataset(gold_dataset, json_rel_path=file_name_json,
                          csv_rel_path=file_name_csv)
    gold_dataset_reloaded = sdm.load_gold_dataset(json_rel_path=file_name_json)
    assert gold_dataset == gold_dataset_reloaded
    os.remove(file_name_json)
    os.remove(file_name_csv)


@requires_scipy
def test_gauge_gold_dataset():
    # Test that pvsystem functions can be guaged
    # FUTURE Report results somewhere using Airspeed Velocity or the like
    json_rel_path = os.path.join(os.path.dirname(sdm.__file__), "sdm.json")
    gold_dataset = sdm.load_gold_dataset(json_rel_path=json_rel_path)
    pprint.pprint(sdm.gauge_gold_dataset(gold_dataset, {
            "i_from_v": pvsystem.i_from_v,
            "v_from_i": pvsystem.v_from_i,
        }))
