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


@pytest.fixture
def martin_ruiz_mismatch_data():
    # Data to run tests of martin_ruiz_spectral_modifier
    kwargs = {
        'clearness_index': [0.56, 0.612, 0.664, 0.716, 0.768, 0.82],
        'airmass_absolute': [2, 1.8, 1.6, 1.4, 1.2, 1],
        'monosi_expected': {
            'dir': [1.09149, 1.07275, 1.05432, 1.03622, 1.01842, 1.00093],
            'sky': [0.88636, 0.85009, 0.81530, 0.78194, 0.74994, 0.71925],
            'gnd': [1.02011, 1.00465, 0.98943, 0.97444, 0.95967, 0.94513]},
        'polysi_expected': {
            'dir': [1.09166, 1.07280, 1.05427, 1.03606, 1.01816, 1.00058],
            'sky': [0.89443, 0.85553, 0.81832, 0.78273, 0.74868, 0.71612],
            'gnd': [1.02638, 1.00888, 0.99168, 0.97476, 0.95814, 0.94180]},
        'asi_expected': {
            'dir': [1.07066, 1.05643, 1.04238, 1.02852, 1.01485, 1.00136],
            'sky': [0.94889, 0.91699, 0.88616, 0.85637, 0.82758, 0.79976],
            'gnd': [1.03801, 1.02259, 1.00740, 0.99243, 0.97769, 0.96316]},
        'monosi_model_params_dict': {
            'direct': {'c': 1.029, 'a': -3.13e-1, 'b': 5.24e-3},
            'sky_diffuse': {'c': 0.764, 'a': -8.82e-1, 'b': -2.04e-2},
            'ground_diffuse': {'c': 0.970, 'a': -2.44e-1, 'b': 1.29e-2}},
        'monosi_custom_params_df': pd.DataFrame({
            'direct': [1.029, -0.313, 0.00524],
            'sky_diffuse': [0.764, -0.882, -0.0204]},
            index=('c', 'a', 'b'))
    }
    return kwargs


def test_martin_ruiz_mm_scalar(martin_ruiz_mismatch_data):
    # test scalar input ; only cell_type given
    clearness_index = martin_ruiz_mismatch_data['clearness_index'][0]
    airmass_absolute = martin_ruiz_mismatch_data['airmass_absolute'][0]
    result = spectrum.martin_ruiz_spectral_modifier(clearness_index,
                                                    airmass_absolute,
                                                    cell_type='asi')

    assert_approx_equal(result['direct'],
                        martin_ruiz_mismatch_data['asi_expected']['dir'][0],
                        significant=5)
    assert_approx_equal(result['sky_diffuse'],
                        martin_ruiz_mismatch_data['asi_expected']['sky'][0],
                        significant=5)
    assert_approx_equal(result['ground_diffuse'],
                        martin_ruiz_mismatch_data['asi_expected']['gnd'][0],
                        significant=5)


def test_martin_ruiz_mm_series(martin_ruiz_mismatch_data):
    # test with Series input ; only cell_type given
    clearness_index = pd.Series(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = pd.Series(martin_ruiz_mismatch_data['airmass_absolute'])
    expected = {
        'dir': pd.Series(martin_ruiz_mismatch_data['polysi_expected']['dir']),
        'sky': pd.Series(martin_ruiz_mismatch_data['polysi_expected']['sky']),
        'gnd': pd.Series(martin_ruiz_mismatch_data['polysi_expected']['gnd'])}

    result = spectrum.martin_ruiz_spectral_modifier(clearness_index,
                                                    airmass_absolute,
                                                    cell_type='polysi')
    assert_series_equal(result['direct'], expected['dir'], atol=1e-5)
    assert_series_equal(result['sky_diffuse'], expected['sky'], atol=1e-5)
    assert_series_equal(result['ground_diffuse'], expected['gnd'], atol=1e-5)


def test_martin_ruiz_mm_nans(martin_ruiz_mismatch_data):
    # test NaN in, NaN out ; only cell_type given
    clearness_index = pd.Series(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = pd.Series(martin_ruiz_mismatch_data['airmass_absolute'])
    airmass_absolute[:5] = np.nan

    result = spectrum.martin_ruiz_spectral_modifier(clearness_index,
                                                    airmass_absolute,
                                                    cell_type='monosi')
    assert np.isnan(result['direct'][:5]).all()
    assert not np.isnan(result['direct'][5:]).any()
    assert np.isnan(result['sky_diffuse'][:5]).all()
    assert not np.isnan(result['sky_diffuse'][5:]).any()
    assert np.isnan(result['ground_diffuse'][:5]).all()
    assert not np.isnan(result['ground_diffuse'][5:]).any()


def test_martin_ruiz_mm_model_dict(martin_ruiz_mismatch_data):
    # test results when giving 'model_parameters' as dict
    # test custom quantity of components and its names can be given
    clearness_index = pd.Series(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = pd.Series(martin_ruiz_mismatch_data['airmass_absolute'])
    expected = {
        'dir': pd.Series(martin_ruiz_mismatch_data['monosi_expected']['dir']),
        'sky': pd.Series(martin_ruiz_mismatch_data['monosi_expected']['sky']),
        'gnd': pd.Series(martin_ruiz_mismatch_data['monosi_expected']['gnd'])}
    model_parameters = martin_ruiz_mismatch_data['monosi_model_params_dict']

    result = spectrum.martin_ruiz_spectral_modifier(
        clearness_index,
        airmass_absolute,
        model_parameters=model_parameters)
    assert_allclose(result['direct'], expected['dir'], atol=1e-5)
    assert_allclose(result['sky_diffuse'], expected['sky'], atol=1e-5)
    assert_allclose(result['ground_diffuse'], expected['gnd'], atol=1e-5)


def test_martin_ruiz_mm_model_df(martin_ruiz_mismatch_data):
    # test results when giving 'model_parameters' as DataFrame
    # test custom quantity of components and its names can be given
    clearness_index = np.array(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = np.array(martin_ruiz_mismatch_data['airmass_absolute'])
    model_parameters = martin_ruiz_mismatch_data['monosi_custom_params_df']
    expected = {
        'dir': np.array(martin_ruiz_mismatch_data['monosi_expected']['dir']),
        'sky': np.array(martin_ruiz_mismatch_data['monosi_expected']['sky'])}

    result = spectrum.martin_ruiz_spectral_modifier(
        clearness_index,
        airmass_absolute,
        model_parameters=model_parameters)
    assert_allclose(result['direct'], expected['dir'], atol=1e-5)
    assert_allclose(result['sky_diffuse'], expected['sky'], atol=1e-5)
    assert_equal(result['ground_diffuse'], None)


def test_martin_ruiz_mm_userwarning(martin_ruiz_mismatch_data):
    # test warning is raised with both 'cell_type' and 'model_parameters'
    clearness_index = pd.Series(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = pd.Series(martin_ruiz_mismatch_data['airmass_absolute'])
    model_parameters = martin_ruiz_mismatch_data['monosi_model_params_dict']

    with pytest.warns(UserWarning,
                      match='Both "cell_type" and "model_parameters" given! '
                            'Using provided "model_parameters".'):
        _ = spectrum.martin_ruiz_spectral_modifier(
            clearness_index,
            airmass_absolute,
            cell_type='asi',
            model_parameters=model_parameters)


def test_martin_ruiz_mm_error_notimplemented(martin_ruiz_mismatch_data):
    # test exception is raised when cell_type does not exist in algorithm
    clearness_index = np.array(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = np.array(martin_ruiz_mismatch_data['airmass_absolute'])

    with pytest.raises(NotImplementedError,
                       match='Cell type parameters not defined in algorithm!'):
        _ = spectrum.martin_ruiz_spectral_modifier(clearness_index,
                                                   airmass_absolute,
                                                   cell_type='')


def test_martin_ruiz_mm_error_missing_params(martin_ruiz_mismatch_data):
    # test exception is raised when missing cell_type and model_parameters
    clearness_index = np.array(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = np.array(martin_ruiz_mismatch_data['airmass_absolute'])

    with pytest.raises(TypeError,
                       match='You must pass at least "cell_type" '
                             'or "model_parameters" as arguments!'):
        _ = spectrum.martin_ruiz_spectral_modifier(clearness_index,
                                                   airmass_absolute)


def test_martin_ruiz_mm_error_model_keys(martin_ruiz_mismatch_data):
    # test exception is raised when  in params keys
    clearness_index = np.array(martin_ruiz_mismatch_data['clearness_index'])
    airmass_absolute = np.array(martin_ruiz_mismatch_data['airmass_absolute'])
    model_parameters = {
        'component_example': {'z': 0.970, 'x': -2.44e-1, 'y': 1.29e-2}}
    with pytest.raises(ValueError,
                       match="You must specify model parameters with keys "
                             "'a','b','c' for each irradiation component."):
        _ = spectrum.martin_ruiz_spectral_modifier(
            clearness_index,
            airmass_absolute,
            model_parameters=model_parameters)
