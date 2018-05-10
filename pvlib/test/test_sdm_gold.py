import os

from numpy.testing import assert_allclose

from pvlib import sdm_gold


def test_sum_current():

    # Do this so test runs using pytest command from different directories
    json_relpath = os.path.join(
        os.path.dirname(__file__), "../data/sdm_gold.json")

    gold_dataset = sdm_gold.convert_gold_dataset(
        sdm_gold.load_gold_dataset(json_relpath=json_relpath))

    for device in gold_dataset["devices"]:
        for iv_curve in device["iv_curves"]:
            current_sum_res = sdm_gold.sum_current(
                iv_curve["r_sh"], iv_curve["r_s"], iv_curve["nNsVth"],
                iv_curve["i_gold"], iv_curve["v_gold"], iv_curve["i_0"],
                iv_curve["i_l"])

            # atol typically smaller that the interval tolerance used for gold
            #  dataset generation
            assert_allclose(current_sum_res, 0., rtol=0., atol=1.e-13)


# TODO test data exactness after file save and reload