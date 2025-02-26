import pytest
import pandas as pd
from numpy.testing import assert_allclose, assert_approx_equal, assert_equal
import numpy as np
from pvlib import spectrum


def test_get_example_spectral_response():
    # test that the sample sr is read and interpolated correctly
    sr = spectrum.get_example_spectral_response()
    assert_equal(len(sr), 185)
    assert_equal(np.sum(sr.index), 136900)
    assert_approx_equal(np.sum(sr), 107.6116)

    wavelength = [270, 850, 950, 1200, 4001]
    expected = [0.0, 0.92778, 1.0, 0.0, 0.0]

    sr = spectrum.get_example_spectral_response(wavelength)
    assert_equal(len(sr), len(wavelength))
    assert_allclose(sr, expected, rtol=1e-5)


@pytest.fixture
def sr_and_eqe_fixture():
    # Just some arbitrary data for testing the conversion functions
    df = pd.DataFrame(
        columns=("wavelength", "quantum_efficiency", "spectral_response"),
        data=[
            # nm, [0,1], A/W
            [300, 0.85, 0.205671370402405],
            [350, 0.86, 0.242772872514211],
            [400, 0.87, 0.280680929019753],
            [450, 0.88, 0.319395539919029],
            [500, 0.89, 0.358916705212040],
            [550, 0.90, 0.399244424898786],
            [600, 0.91, 0.440378698979267],
            [650, 0.92, 0.482319527453483],
            [700, 0.93, 0.525066910321434],
            [750, 0.94, 0.568620847583119],
            [800, 0.95, 0.612981339238540],
            [850, 0.90, 0.617014111207215],
            [900, 0.80, 0.580719163489143],
            [950, 0.70, 0.536358671833723],
            [1000, 0.6, 0.483932636240953],
            [1050, 0.4, 0.338752845368667],
        ],
    )
    df.set_index("wavelength", inplace=True)
    return df


def test_sr_to_qe(sr_and_eqe_fixture):
    # vector type
    qe = spectrum.sr_to_qe(
        sr_and_eqe_fixture["spectral_response"].values,
        sr_and_eqe_fixture.index.values,  # wavelength, nm
    )
    assert_allclose(qe, sr_and_eqe_fixture["quantum_efficiency"])
    # pandas series type
    # note: output Series' name should match the input
    qe = spectrum.sr_to_qe(
        sr_and_eqe_fixture["spectral_response"]
    )
    pd.testing.assert_series_equal(
        qe, sr_and_eqe_fixture["quantum_efficiency"],
        check_names=False
    )
    assert qe.name == "spectral_response"
    # series normalization
    qe = spectrum.sr_to_qe(
        sr_and_eqe_fixture["spectral_response"] * 10, normalize=True
    )
    pd.testing.assert_series_equal(
        qe,
        sr_and_eqe_fixture["quantum_efficiency"]
        / max(sr_and_eqe_fixture["quantum_efficiency"]),
        check_names=False,
    )
    # error on lack of wavelength parameter if no pandas object is provided
    with pytest.raises(TypeError, match="must have an '.index' attribute"):
        _ = spectrum.sr_to_qe(sr_and_eqe_fixture["spectral_response"].values)


def test_qe_to_sr(sr_and_eqe_fixture):
    # vector type
    sr = spectrum.qe_to_sr(
        sr_and_eqe_fixture["quantum_efficiency"].values,
        sr_and_eqe_fixture.index.values,  # wavelength, nm
    )
    assert_allclose(sr, sr_and_eqe_fixture["spectral_response"])
    # pandas series type
    # note: output Series' name should match the input
    sr = spectrum.qe_to_sr(
        sr_and_eqe_fixture["quantum_efficiency"]
    )
    pd.testing.assert_series_equal(
        sr, sr_and_eqe_fixture["spectral_response"],
        check_names=False
    )
    assert sr.name == "quantum_efficiency"
    # series normalization
    sr = spectrum.qe_to_sr(
        sr_and_eqe_fixture["quantum_efficiency"] * 10, normalize=True
    )
    pd.testing.assert_series_equal(
        sr,
        sr_and_eqe_fixture["spectral_response"]
        / max(sr_and_eqe_fixture["spectral_response"]),
        check_names=False,
    )
    # error on lack of wavelength parameter if no pandas object is provided
    with pytest.raises(TypeError, match="must have an '.index' attribute"):
        _ = spectrum.qe_to_sr(
            sr_and_eqe_fixture["quantum_efficiency"].values
        )


def test_qe_and_sr_reciprocal_conversion(sr_and_eqe_fixture):
    # test that the conversion functions are reciprocal
    qe = spectrum.sr_to_qe(sr_and_eqe_fixture["spectral_response"])
    sr = spectrum.qe_to_sr(qe)
    assert_allclose(sr, sr_and_eqe_fixture["spectral_response"])
    qe = spectrum.sr_to_qe(sr)
    assert_allclose(qe, sr_and_eqe_fixture["quantum_efficiency"])
