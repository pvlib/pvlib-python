import pytest
import pandas as pd
import numpy as np
from pvlib import spectrum
from numpy.testing import assert_allclose


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
