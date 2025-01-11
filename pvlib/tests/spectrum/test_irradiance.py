import pytest
from numpy.testing import assert_allclose, assert_approx_equal, assert_equal
import pandas as pd
import numpy as np
from pvlib import spectrum
from pvlib._deprecation import pvlibDeprecationWarning

from ..conftest import assert_series_equal, fail_on_pvlib_version


@fail_on_pvlib_version('0.12')
def test_get_am15g():
    # test that the reference spectrum is read and interpolated correctly
    with pytest.warns(pvlibDeprecationWarning,
                      match="get_reference_spectra instead"):
        e = spectrum.get_am15g()
    assert_equal(len(e), 2002)
    assert_equal(np.sum(e.index), 2761442)
    assert_approx_equal(np.sum(e), 1002.88, significant=6)

    wavelength = [270, 850, 950, 1200, 1201.25, 4001]
    expected = [0.0, 0.893720, 0.147260, 0.448250, 0.4371025, 0.0]

    with pytest.warns(pvlibDeprecationWarning,
                      match="get_reference_spectra instead"):
        e = spectrum.get_am15g(wavelength)
    assert_equal(len(e), len(wavelength))
    assert_allclose(e, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "reference_identifier,expected_sums",
    [
        (
            "ASTM G173-03",  # reference_identifier
            {  # expected_sums
                "extraterrestrial": 1356.15,
                "global": 1002.88,
                "direct": 887.65,
            },
        ),
    ],
)
def test_get_reference_spectra(reference_identifier, expected_sums):
    # test reading of a standard spectrum
    standard = spectrum.get_reference_spectra(standard=reference_identifier)
    assert set(standard.columns) == expected_sums.keys()
    assert standard.index.name == "wavelength"
    assert standard.index.is_monotonic_increasing is True
    expected_sums = pd.Series(expected_sums)  # convert prior to comparison
    assert_series_equal(np.sum(standard, axis=0), expected_sums, atol=1e-2)


def test_get_reference_spectra_custom_wavelengths():
    # test that the spectrum is interpolated correctly when custom wavelengths
    # are specified
    # only checked for ASTM G173-03 reference spectrum
    wavelength = [270, 850, 951.634, 1200, 4001]
    expected_sums = pd.Series(
        {"extraterrestrial": 2.23266, "global": 1.68952, "direct": 1.58480}
    )  # for given ``wavelength``
    standard = spectrum.get_reference_spectra(
        wavelength, standard="ASTM G173-03"
    )
    assert_equal(len(standard), len(wavelength))
    # check no NaN values were returned
    assert not standard.isna().any().any()  # double any to return one value
    assert_series_equal(np.sum(standard, axis=0), expected_sums, atol=1e-4)


def test_get_reference_spectra_invalid_reference():
    # test that an invalid reference identifier raises a ValueError
    with pytest.raises(ValueError, match="Invalid standard identifier"):
        spectrum.get_reference_spectra(standard="invalid")


def test_average_photon_energy_series():
    # test that the APE is calculated correctly with single spectrum
    # series input

    spectra = spectrum.get_reference_spectra()
    spectra = spectra['global']
    ape = spectrum.average_photon_energy(spectra)
    expected = 1.45017
    assert_allclose(ape, expected, rtol=1e-4)


def test_average_photon_energy_dataframe():
    # test that the APE is calculated correctly with multiple spectra
    # dataframe input and that the output is a series

    spectra = spectrum.get_reference_spectra().T
    ape = spectrum.average_photon_energy(spectra)
    expected = pd.Series([1.36848, 1.45017, 1.40885])
    expected.index = spectra.index
    assert_series_equal(ape, expected, rtol=1e-4)


def test_average_photon_energy_invalid_type():
    # test that spectrum argument is either a pandas Series or dataframe
    spectra = 5
    with pytest.raises(TypeError, match='must be either a pandas Series or'
                       ' DataFrame'):
        spectrum.average_photon_energy(spectra)


def test_average_photon_energy_neg_irr_series():
    # test for handling of negative spectral irradiance values with a
    # pandas Series input

    spectra = spectrum.get_reference_spectra()['global']*-1
    with pytest.raises(ValueError, match='must be positive'):
        spectrum.average_photon_energy(spectra)


def test_average_photon_energy_neg_irr_dataframe():
    # test for handling of negative spectral irradiance values with a
    # pandas DataFrame input

    spectra = spectrum.get_reference_spectra().T*-1

    with pytest.raises(ValueError, match='must be positive'):
        spectrum.average_photon_energy(spectra)


def test_average_photon_energy_zero_irr():
    # test for handling of zero spectral irradiance values with
    # pandas DataFrame and pandas Series input

    spectra_df_zero = spectrum.get_reference_spectra().T
    spectra_df_zero.iloc[1] = 0
    spectra_series_zero = spectrum.get_reference_spectra()['global']*0
    out_1 = spectrum.average_photon_energy(spectra_df_zero)
    out_2 = spectrum.average_photon_energy(spectra_series_zero)
    expected_1 = np.array([1.36848, np.nan, 1.40885])
    expected_2 = np.nan
    assert_allclose(out_1, expected_1, atol=1e-3)
    assert_allclose(out_2, expected_2, atol=1e-3)
