import pytest
from numpy.testing import assert_allclose, assert_approx_equal, assert_equal
import pandas as pd
import numpy as np
from pvlib import spectrum

from .conftest import DATA_DIR, assert_series_equal

SPECTRL2_TEST_DATA = DATA_DIR / 'spectrl2_example_spectra.csv'


@pytest.fixture
def spectrl2_data():
    # reference spectra generated with solar_utils==0.3
    """
    expected = solar_utils.spectrl2(
        units=1,
        location=[40, -80, -5],
        datetime=[2020, 3, 15, 10, 45, 59],
        weather=[1013, 15],
        orientation=[0, 180],
        atmospheric_conditions=[1.14, 0.65, 0.344, 0.1, 1.42],
        albedo=[0.3, 0.7, 0.8, 1.3, 2.5, 4.0] + [0.2]*6,
    )
    """
    kwargs = {
        'surface_tilt': 0,
        'relative_airmass': 1.4899535986910446,
        'apparent_zenith': 47.912086486816406,
        'aoi': 47.91208648681641,
        'ground_albedo': 0.2,
        'surface_pressure': 101300,
        'ozone': 0.344,
        'precipitable_water': 1.42,
        'aerosol_turbidity_500nm': 0.1,
        'dayofyear': 75
    }
    df = pd.read_csv(SPECTRL2_TEST_DATA, index_col=0)
    # convert um to nm
    df['wavelength'] = np.round(df['wavelength'] * 1000, 1)
    df[['specdif', 'specdir', 'specetr', 'specglo']] /= 1000
    return kwargs, df


def test_spectrl2(spectrl2_data):
    # compare against output from solar_utils wrapper around NREL spectrl2_2.c
    kwargs, expected = spectrl2_data
    actual = spectrum.spectrl2(**kwargs)
    assert_allclose(expected['wavelength'].values, actual['wavelength'])
    assert_allclose(expected['specdif'].values, actual['dhi'].ravel(),
                    atol=7e-5)
    assert_allclose(expected['specdir'].values, actual['dni'].ravel(),
                    atol=1.5e-4)
    assert_allclose(expected['specetr'], actual['dni_extra'].ravel(),
                    atol=2e-4)
    assert_allclose(expected['specglo'], actual['poa_global'].ravel(),
                    atol=1e-4)


def test_spectrl2_array(spectrl2_data):
    # test that supplying arrays instead of scalars works
    kwargs, expected = spectrl2_data
    kwargs = {k: np.array([v, v, v]) for k, v in kwargs.items()}
    actual = spectrum.spectrl2(**kwargs)

    assert actual['wavelength'].shape == (122,)

    keys = ['dni_extra', 'dhi', 'dni', 'poa_sky_diffuse', 'poa_ground_diffuse',
            'poa_direct', 'poa_global']
    for key in keys:
        assert actual[key].shape == (122, 3)


def test_spectrl2_series(spectrl2_data):
    # test that supplying Series instead of scalars works
    kwargs, expected = spectrl2_data
    kwargs.pop('dayofyear')
    index = pd.to_datetime(['2020-03-15 10:45:59']*3)
    kwargs = {k: pd.Series([v, v, v], index=index) for k, v in kwargs.items()}
    actual = spectrum.spectrl2(**kwargs)

    assert actual['wavelength'].shape == (122,)

    keys = ['dni_extra', 'dhi', 'dni', 'poa_sky_diffuse', 'poa_ground_diffuse',
            'poa_direct', 'poa_global']
    for key in keys:
        assert actual[key].shape == (122, 3)


def test_dayofyear_missing(spectrl2_data):
    # test that not specifying dayofyear with non-pandas inputs raises error
    kwargs, expected = spectrl2_data
    kwargs.pop('dayofyear')
    with pytest.raises(ValueError, match='dayofyear must be specified'):
        _ = spectrum.spectrl2(**kwargs)


def test_aoi_gt_90(spectrl2_data):
    # test that returned irradiance values are non-negative when aoi > 90
    # see GH #1348
    kwargs, _ = spectrl2_data
    kwargs['apparent_zenith'] = 70
    kwargs['aoi'] = 130
    kwargs['surface_tilt'] = 60

    spectra = spectrum.spectrl2(**kwargs)
    for key in ['poa_direct', 'poa_global']:
        message = f'{key} contains negative values for aoi>90'
        assert np.all(spectra[key] >= 0), message


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


def test_get_am15g():
    # test that the reference spectrum is read and interpolated correctly
    e = spectrum.get_am15g()
    assert_equal(len(e), 2002)
    assert_equal(np.sum(e.index), 2761442)
    assert_approx_equal(np.sum(e), 1002.88, significant=6)

    wavelength = [270, 850, 950, 1200, 4001]
    expected = [0.0, 0.893720, 0.147260, 0.448250, 0.0]

    e = spectrum.get_am15g(wavelength)
    assert_equal(len(e), len(wavelength))
    assert_allclose(e, expected, rtol=1e-6)


def test_calc_spectral_mismatch_field(spectrl2_data):
    # test that the mismatch is calculated correctly with
    # - default and custom reference spectrum
    # - single or multiple sun spectra

    # sample data
    _, e_sun = spectrl2_data
    e_sun = e_sun.set_index('wavelength')
    e_sun = e_sun.transpose()

    e_ref = spectrum.get_am15g()
    sr = spectrum.get_example_spectral_response()

    # test with single sun spectrum, same as ref spectrum
    mm = spectrum.calc_spectral_mismatch_field(sr, e_sun=e_ref)
    assert_approx_equal(mm, 1.0, significant=6)

    # test with single sun spectrum
    mm = spectrum.calc_spectral_mismatch_field(sr, e_sun=e_sun.loc['specglo'])
    assert_approx_equal(mm, 0.992397, significant=6)

    # test with single sun spectrum, also used as reference spectrum
    mm = spectrum.calc_spectral_mismatch_field(sr,
                                               e_sun=e_sun.loc['specglo'],
                                               e_ref=e_sun.loc['specglo'])
    assert_approx_equal(mm, 1.0, significant=6)

    # test with multiple sun spectra
    expected = [0.972982, 0.995581, 0.899782, 0.992397]

    mm = spectrum.calc_spectral_mismatch_field(sr, e_sun=e_sun)
    assert mm.index is e_sun.index
    assert_allclose(mm, expected, rtol=1e-6)


def test_martin_ruiz_spectral_modifier():
    # tests with only cell_type given
    # test with scalar values
    clearness_index = 0.82
    airmass_absolute = 1.2
    # Expected values: Direct | Sky diffuse | Ground diffuse
    # Do not change order in any 'expected' values list
    expected = (1.00197741, 0.71632057, 0.94757498)

    result = \
        spectrum.martin_ruiz_spectral_modifier(clearness_index,
                                               airmass_absolute,
                                               cell_type='monosi')
    assert_approx_equal(result['direct'], expected[0])
    assert_approx_equal(result['sky_diffuse'], expected[1])
    assert_approx_equal(result['ground_diffuse'], expected[2])

    # test NaN in, NaN out
    clearness_index = 0.82
    airmass_absolute = np.nan
    result = spectrum.martin_ruiz_spectral_modifier(clearness_index,
                                                    airmass_absolute,
                                                    cell_type='monosi')
    assert result.isna().all()

    # test with Series input
    clearness_index = pd.Series([0.56, 0.67, 0.80])
    airmass_absolute = pd.Series([1.6, 1.4, 1.2])
    expected = (
        pd.Series([1.088928, 1.050989, 1.008082]),  # Direct
        pd.Series([0.901327, 0.816901, 0.726754]),  # Sky diffuse
        pd.Series([1.019917, 0.986947, 0.949899]))  # Ground diffuse

    result = \
        spectrum.martin_ruiz_spectral_modifier(clearness_index,
                                               airmass_absolute,
                                               cell_type='polysi')
    assert_series_equal(result['direct'], expected[0], atol=1e-5)
    assert_series_equal(result['sky_diffuse'], expected[1], atol=1e-5)
    assert_series_equal(result['ground_diffuse'], expected[2], atol=1e-5)

    # test results when giving 'model_parameters' as DataFrame
    # test custom quantity of components and its names can be given
    clearness_index = np.array([0.56, 0.612, 0.664, 0.716, 0.768, 0.82])
    airmass_absolute = np.array([2, 1.8, 1.6, 1.4, 1.2, 1])
    model_parameters = pd.DataFrame({  # monosi values
        'direct': [1.029, -0.313, 0.00524],
        'diffuse_sky':  [0.764, -0.882, -0.0204]},
        index=('c', 'a', 'b'))
    expected = (  # Direct / Sky diffuse / Ground diffuse
        np.array([1.09149, 1.07274, 1.05432, 1.03621, 1.01841, 1.00092]),
        np.array([0.88636, 0.85009, 0.81530, 0.78193, 0.74993, 0.71924]))

    result = spectrum.martin_ruiz_spectral_modifier(
        clearness_index,
        airmass_absolute,
        model_parameters=model_parameters)
    assert_allclose(result['direct'], expected[0], atol=1e-5)
    assert_allclose(result['diffuse_sky'], expected[1], atol=1e-5)

    # test warning is raised with both 'cell_type' and 'model_parameters'
    # test results when giving 'model_parameters' as dict
    clearness_index = np.array([0.56, 0.612, 0.664, 0.716, 0.768, 0.82])
    airmass_absolute = np.array([2, 1.8, 1.6, 1.4, 1.2, 1])
    model_parameters = {  # Using 'monosi' values
        'direct': {'c': 1.029, 'a': -3.13e-1, 'b': 5.24e-3},
        'sky_diffuse': {'c': 0.764, 'a': -8.82e-1, 'b': -2.04e-2},
        'ground_diffuse': {'c': 0.970, 'a': -2.44e-1, 'b': 1.29e-2}}
    expected = (  # Direct / Sky diffuse / Ground diffuse
        np.array([1.09149, 1.07274, 1.05432, 1.03621, 1.01841, 1.00092]),
        np.array([0.88636, 0.85009, 0.81530, 0.78193, 0.74993, 0.71924]),
        np.array([1.02011, 1.00465, 0.98943, 0.97443, 0.95967, 0.94513]))

    with pytest.warns(UserWarning,
                      match='Both "cell_type" and "model_parameters" given! '
                            'Using provided "model_parameters".'):
        result = spectrum.martin_ruiz_spectral_modifier(
            clearness_index,
            airmass_absolute,
            cell_type='asi',
            model_parameters=model_parameters)
        assert_allclose(result['direct'], expected[0], atol=1e-5)
        assert_allclose(result['sky_diffuse'], expected[1], atol=1e-5)
        assert_allclose(result['ground_diffuse'], expected[2], atol=1e-5)


def test_martin_ruiz_spectral_modifier_errors():
    # mock values to run errors
    clearness_index = 0.75
    airmass_absolute = 1.6
    # test exception raised when cell_type does not exist in algorithm
    with pytest.raises(NotImplementedError,
                       match='Cell type parameters not defined in algorithm!'):
        _ = spectrum.martin_ruiz_spectral_modifier(clearness_index,
                                                   airmass_absolute,
                                                   cell_type='')
    # test exception raised when missing cell_type and model_parameters
    with pytest.raises(TypeError,
                       match='You must pass at least "cell_type" '
                             'or "model_parameters" as arguments!'):
        _ = spectrum.martin_ruiz_spectral_modifier(clearness_index,
                                                   airmass_absolute)
    # test for error in params keys
    clearness_index = 0.74
    airmass_absolute = 1.5
    model_parameters = {
        'direct': {'c': 1.029, 'a': -3.13e-1, 'b': 5.24e-3},
        'sky_diffuse': {'c': 0.764, 'a': -8.82e-1, 'b': -2.04e-2},
        'ground_diffuse': {'z': 0.970, 'x': -2.44e-1, 'y': 1.29e-2}}
    with pytest.raises(ValueError,
                       match="You must specify model parameters with keys "
                             "'a','b','c' for each irradiation component."):
        _ = spectrum.martin_ruiz_spectral_modifier(
            clearness_index,
            airmass_absolute,
            model_parameters=model_parameters)
