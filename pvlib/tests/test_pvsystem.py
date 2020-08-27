from collections import OrderedDict

import numpy as np
from numpy import nan, array
import pandas as pd

import pytest
from conftest import assert_series_equal, assert_frame_equal
from numpy.testing import assert_allclose

from pvlib import inverter, pvsystem
from pvlib import atmosphere
from pvlib import iam as _iam
from pvlib.location import Location
from pvlib import temperature
from pvlib._deprecation import pvlibDeprecationWarning

from conftest import needs_numpy_1_10, requires_scipy, fail_on_pvlib_version


@pytest.mark.parametrize('iam_model,model_params', [
    ('ashrae', {'b': 0.05}),
    ('physical', {'K': 4, 'L': 0.002, 'n': 1.526}),
    ('martin_ruiz', {'a_r': 0.16}),
])
def test_PVSystem_get_iam(mocker, iam_model, model_params):
    m = mocker.spy(_iam, iam_model)
    system = pvsystem.PVSystem(module_parameters=model_params)
    thetas = 1
    iam = system.get_iam(thetas, iam_model=iam_model)
    m.assert_called_with(thetas, **model_params)
    assert iam < 1.


def test_PVSystem_get_iam_sapm(sapm_module_params, mocker):
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)
    mocker.spy(_iam, 'sapm')
    aoi = 0
    out = system.get_iam(aoi, 'sapm')
    _iam.sapm.assert_called_once_with(aoi, sapm_module_params)
    assert_allclose(out, 1.0, atol=0.01)


def test_PVSystem_get_iam_interp(sapm_module_params, mocker):
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)
    with pytest.raises(ValueError):
        system.get_iam(45, iam_model='interp')


def test__normalize_sam_product_names():

    BAD_NAMES  = [' -.()[]:+/",', 'Module[1]']
    NORM_NAMES = ['____________', 'Module_1_']

    norm_names = pvsystem._normalize_sam_product_names(BAD_NAMES)
    assert list(norm_names) == NORM_NAMES

    BAD_NAMES  = ['Module[1]', 'Module(1)']
    NORM_NAMES = ['Module_1_', 'Module_1_']

    with pytest.warns(UserWarning):
        norm_names = pvsystem._normalize_sam_product_names(BAD_NAMES)
    assert list(norm_names) == NORM_NAMES

    BAD_NAMES  = ['Module[1]', 'Module[1]']
    NORM_NAMES = ['Module_1_', 'Module_1_']

    with pytest.warns(UserWarning):
        norm_names = pvsystem._normalize_sam_product_names(BAD_NAMES)
    assert list(norm_names) == NORM_NAMES


def test_PVSystem_get_iam_invalid(sapm_module_params, mocker):
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)
    with pytest.raises(ValueError):
        system.get_iam(45, iam_model='not_a_model')


def test_retrieve_sam_raise_no_parameters():
    """
    Raise an exception if no parameters are provided to `retrieve_sam()`.
    """
    with pytest.raises(ValueError) as error:
        pvsystem.retrieve_sam()
    assert 'A name or path must be provided!' == str(error.value)


def test_retrieve_sam_cecmod():
    """
    Test the expected data is retrieved from the CEC module database. In
    particular, check for a known module in the database and check for the
    expected keys for that module.
    """
    data = pvsystem.retrieve_sam('cecmod')
    keys = [
        'BIPV',
        'Date',
        'T_NOCT',
        'A_c',
        'N_s',
        'I_sc_ref',
        'V_oc_ref',
        'I_mp_ref',
        'V_mp_ref',
        'alpha_sc',
        'beta_oc',
        'a_ref',
        'I_L_ref',
        'I_o_ref',
        'R_s',
        'R_sh_ref',
        'Adjust',
        'gamma_r',
        'Version',
        'STC',
        'PTC',
        'Technology',
        'Bifacial',
        'Length',
        'Width',
    ]
    module = 'Itek_Energy_LLC_iT_300_HE'
    assert module in data
    assert set(data[module].keys()) == set(keys)


def test_retrieve_sam_cecinverter():
    """
    Test the expected data is retrieved from the CEC inverter database. In
    particular, check for a known inverter in the database and check for the
    expected keys for that inverter.
    """
    data = pvsystem.retrieve_sam('cecinverter')
    keys = [
        'Vac',
        'Paco',
        'Pdco',
        'Vdco',
        'Pso',
        'C0',
        'C1',
        'C2',
        'C3',
        'Pnt',
        'Vdcmax',
        'Idcmax',
        'Mppt_low',
        'Mppt_high',
        'CEC_Date',
        'CEC_Type',
    ]
    inverter = 'Yaskawa_Solectria_Solar__PVI_5300_208__208V_'
    assert inverter in data
    assert set(data[inverter].keys()) == set(keys)


def test_sapm(sapm_module_params):

    times = pd.date_range(start='2015-01-01', periods=5, freq='12H')
    effective_irradiance = pd.Series([-1000, 500, 1100, np.nan, 1000],
                                     index=times)
    temp_cell = pd.Series([10, 25, 50, 25, np.nan], index=times)

    out = pvsystem.sapm(effective_irradiance, temp_cell, sapm_module_params)

    expected = pd.DataFrame(np.array(
      [[  -5.0608322 ,   -4.65037767,           nan,           nan,
                  nan,   -4.91119927,   -4.15367716],
       [   2.545575  ,    2.28773882,   56.86182059,   47.21121608,
         108.00693168,    2.48357383,    1.71782772],
       [   5.65584763,    5.01709903,   54.1943277 ,   42.51861718,
         213.32011294,    5.52987899,    3.48660728],
       [          nan,           nan,           nan,           nan,
                  nan,           nan,           nan],
       [          nan,           nan,           nan,           nan,
                  nan,           nan,           nan]]),
        columns=['i_sc', 'i_mp', 'v_oc', 'v_mp', 'p_mp', 'i_x', 'i_xx'],
        index=times)

    assert_frame_equal(out, expected, check_less_precise=4)

    out = pvsystem.sapm(1000, 25, sapm_module_params)

    expected = OrderedDict()
    expected['i_sc'] = 5.09115
    expected['i_mp'] = 4.5462909092579995
    expected['v_oc'] = 59.260800000000003
    expected['v_mp'] = 48.315600000000003
    expected['p_mp'] = 219.65677305534581
    expected['i_x'] = 4.9759899999999995
    expected['i_xx'] = 3.1880204359100004

    for k, v in expected.items():
        assert_allclose(out[k], v, atol=1e-4)

    # just make sure it works with Series input
    pvsystem.sapm(effective_irradiance, temp_cell,
                  pd.Series(sapm_module_params))


def test_PVSystem_sapm(sapm_module_params, mocker):
    mocker.spy(pvsystem, 'sapm')
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)
    effective_irradiance = 500
    temp_cell = 25
    out = system.sapm(effective_irradiance, temp_cell)
    pvsystem.sapm.assert_called_once_with(effective_irradiance, temp_cell,
                                          sapm_module_params)
    assert_allclose(out['p_mp'], 100, atol=100)


@pytest.mark.parametrize('airmass,expected', [
    (1.5, 1.00028714375),
    (np.array([[10, np.nan]]), np.array([[0.999535, 0]])),
    (pd.Series([5]), pd.Series([1.0387675]))
])
def test_sapm_spectral_loss(sapm_module_params, airmass, expected):

    out = pvsystem.sapm_spectral_loss(airmass, sapm_module_params)

    if isinstance(airmass, pd.Series):
        assert_series_equal(out, expected, check_less_precise=4)
    else:
        assert_allclose(out, expected, atol=1e-4)


def test_PVSystem_sapm_spectral_loss(sapm_module_params, mocker):
    mocker.spy(pvsystem, 'sapm_spectral_loss')
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)
    airmass = 2
    out = system.sapm_spectral_loss(airmass)
    pvsystem.sapm_spectral_loss.assert_called_once_with(airmass,
                                                        sapm_module_params)
    assert_allclose(out, 1, atol=0.5)


# this test could be improved to cover all cell types.
# could remove the need for specifying spectral coefficients if we don't
# care about the return value at all
@pytest.mark.parametrize('module_parameters,module_type,coefficients', [
    ({'Technology': 'mc-Si'}, 'multisi', None),
    ({'Material': 'Multi-c-Si'}, 'multisi', None),
    ({'first_solar_spectral_coefficients': (
        0.84, -0.03, -0.008, 0.14, 0.04, -0.002)},
     None,
     (0.84, -0.03, -0.008, 0.14, 0.04, -0.002))
    ])
def test_PVSystem_first_solar_spectral_loss(module_parameters, module_type,
                                            coefficients, mocker):
    mocker.spy(atmosphere, 'first_solar_spectral_correction')
    system = pvsystem.PVSystem(module_parameters=module_parameters)
    pw = 3
    airmass_absolute = 3
    out = system.first_solar_spectral_loss(pw, airmass_absolute)
    atmosphere.first_solar_spectral_correction.assert_called_once_with(
        pw, airmass_absolute, module_type, coefficients)
    assert_allclose(out, 1, atol=0.5)


@pytest.mark.parametrize('test_input,expected', [
    ([1000, 100, 5, 45], 1140.0510967821877),
    ([np.array([np.nan, 1000, 1000]),
      np.array([100, np.nan, 100]),
      np.array([1.1, 1.1, 1.1]),
      np.array([10, 10, 10])],
     np.array([np.nan, np.nan, 1081.1574])),
    ([pd.Series([1000]), pd.Series([100]), pd.Series([1.1]),
      pd.Series([10])],
     pd.Series([1081.1574]))
])
def test_sapm_effective_irradiance(sapm_module_params, test_input, expected):
    test_input.append(sapm_module_params)
    out = pvsystem.sapm_effective_irradiance(*test_input)
    if isinstance(test_input, pd.Series):
        assert_series_equal(out, expected, check_less_precise=4)
    else:
        assert_allclose(out, expected, atol=1e-1)


def test_PVSystem_sapm_effective_irradiance(sapm_module_params, mocker):
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)
    mocker.spy(pvsystem, 'sapm_effective_irradiance')

    poa_direct = 900
    poa_diffuse = 100
    airmass_absolute = 1.5
    aoi = 0
    p = (sapm_module_params['A4'], sapm_module_params['A3'],
         sapm_module_params['A2'], sapm_module_params['A1'],
         sapm_module_params['A0'])
    f1 = np.polyval(p, airmass_absolute)
    expected = f1 * (poa_direct + sapm_module_params['FD'] * poa_diffuse)
    out = system.sapm_effective_irradiance(
        poa_direct, poa_diffuse, airmass_absolute, aoi)
    pvsystem.sapm_effective_irradiance.assert_called_once_with(
        poa_direct, poa_diffuse, airmass_absolute, aoi, sapm_module_params)
    assert_allclose(out, expected, atol=0.1)


def test_PVSystem_sapm_celltemp(mocker):
    a, b, deltaT = (-3.47, -0.0594, 3)  # open_rack_glass_glass
    temp_model_params = {'a': a, 'b': b, 'deltaT': deltaT}
    system = pvsystem.PVSystem(temperature_model_parameters=temp_model_params)
    mocker.spy(temperature, 'sapm_cell')
    temps = 25
    irrads = 1000
    winds = 1
    out = system.sapm_celltemp(irrads, temps, winds)
    temperature.sapm_cell.assert_called_once_with(irrads, temps, winds, a, b,
                                                  deltaT)
    assert_allclose(out, 57, atol=1)


def test_PVSystem_sapm_celltemp_kwargs(mocker):
    temp_model_params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][
        'open_rack_glass_glass']
    system = pvsystem.PVSystem(temperature_model_parameters=temp_model_params)
    mocker.spy(temperature, 'sapm_cell')
    temps = 25
    irrads = 1000
    winds = 1
    out = system.sapm_celltemp(irrads, temps, winds)
    temperature.sapm_cell.assert_called_once_with(irrads, temps, winds,
                                                  temp_model_params['a'],
                                                  temp_model_params['b'],
                                                  temp_model_params['deltaT'])
    assert_allclose(out, 57, atol=1)


def test_PVSystem_pvsyst_celltemp(mocker):
    parameter_set = 'insulated'
    temp_model_params = temperature.TEMPERATURE_MODEL_PARAMETERS['pvsyst'][
        parameter_set]
    alpha_absorption = 0.85
    eta_m = 0.17
    module_parameters = {'alpha_absorption': alpha_absorption, 'eta_m': eta_m}
    system = pvsystem.PVSystem(module_parameters=module_parameters,
                               temperature_model_parameters=temp_model_params)
    mocker.spy(temperature, 'pvsyst_cell')
    irrad = 800
    temp = 45
    wind = 0.5
    out = system.pvsyst_celltemp(irrad, temp, wind_speed=wind)
    temperature.pvsyst_cell.assert_called_once_with(
        irrad, temp, wind, temp_model_params['u_c'], temp_model_params['u_v'],
        eta_m, alpha_absorption)
    assert (out < 90) and (out > 70)


def test_PVSystem_faiman_celltemp(mocker):
    u0, u1 = 25.0, 6.84  # default values
    temp_model_params = {'u0': u0, 'u1': u1}
    system = pvsystem.PVSystem(temperature_model_parameters=temp_model_params)
    mocker.spy(temperature, 'faiman')
    temps = 25
    irrads = 1000
    winds = 1
    out = system.faiman_celltemp(irrads, temps, winds)
    temperature.faiman.assert_called_once_with(irrads, temps, winds, u0, u1)
    assert_allclose(out, 56.4, atol=1)


def test__infer_temperature_model_params():
    system = pvsystem.PVSystem(module_parameters={},
                               racking_model='open_rack',
                               module_type='glass_polymer')
    expected = temperature.TEMPERATURE_MODEL_PARAMETERS[
        'sapm']['open_rack_glass_polymer']
    assert expected == system._infer_temperature_model_params()
    system = pvsystem.PVSystem(module_parameters={},
                               racking_model='freestanding',
                               module_type='glass_polymer')
    expected = temperature.TEMPERATURE_MODEL_PARAMETERS[
        'pvsyst']['freestanding']
    assert expected == system._infer_temperature_model_params()


def test_calcparams_desoto(cec_module_params):
    times = pd.date_range(start='2015-01-01', periods=3, freq='12H')
    effective_irradiance = pd.Series([0.0, 800.0, 800.0], index=times)
    temp_cell = pd.Series([25, 25, 50], index=times)

    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                                  effective_irradiance,
                                  temp_cell,
                                  alpha_sc=cec_module_params['alpha_sc'],
                                  a_ref=cec_module_params['a_ref'],
                                  I_L_ref=cec_module_params['I_L_ref'],
                                  I_o_ref=cec_module_params['I_o_ref'],
                                  R_sh_ref=cec_module_params['R_sh_ref'],
                                  R_s=cec_module_params['R_s'],
                                  EgRef=1.121,
                                  dEgdT=-0.0002677)

    assert_series_equal(IL, pd.Series([0.0, 6.036, 6.096], index=times),
                        check_less_precise=3)
    assert_series_equal(I0, pd.Series([0.0, 1.94e-9, 7.419e-8], index=times),
                        check_less_precise=3)
    assert_allclose(Rs, 0.094)
    assert_series_equal(Rsh, pd.Series([np.inf, 19.65, 19.65], index=times),
                        check_less_precise=3)
    assert_series_equal(nNsVth, pd.Series([0.473, 0.473, 0.5127], index=times),
                        check_less_precise=3)


def test_calcparams_cec(cec_module_params):
    times = pd.date_range(start='2015-01-01', periods=3, freq='12H')
    effective_irradiance = pd.Series([0.0, 800.0, 800.0], index=times)
    temp_cell = pd.Series([25, 25, 50], index=times)

    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_cec(
                                  effective_irradiance,
                                  temp_cell,
                                  alpha_sc=cec_module_params['alpha_sc'],
                                  a_ref=cec_module_params['a_ref'],
                                  I_L_ref=cec_module_params['I_L_ref'],
                                  I_o_ref=cec_module_params['I_o_ref'],
                                  R_sh_ref=cec_module_params['R_sh_ref'],
                                  R_s=cec_module_params['R_s'],
                                  Adjust=cec_module_params['Adjust'],
                                  EgRef=1.121,
                                  dEgdT=-0.0002677)

    assert_series_equal(IL, pd.Series([0.0, 6.036, 6.0896], index=times),
                        check_less_precise=3)
    assert_series_equal(I0, pd.Series([0.0, 1.94e-9, 7.419e-8], index=times),
                        check_less_precise=3)
    assert_allclose(Rs, 0.094)
    assert_series_equal(Rsh, pd.Series([np.inf, 19.65, 19.65], index=times),
                        check_less_precise=3)
    assert_series_equal(nNsVth, pd.Series([0.473, 0.473, 0.5127], index=times),
                        check_less_precise=3)


def test_calcparams_pvsyst(pvsyst_module_params):
    times = pd.date_range(start='2015-01-01', periods=2, freq='12H')
    effective_irradiance = pd.Series([0.0, 800.0], index=times)
    temp_cell = pd.Series([25, 50], index=times)

    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_pvsyst(
        effective_irradiance,
        temp_cell,
        alpha_sc=pvsyst_module_params['alpha_sc'],
        gamma_ref=pvsyst_module_params['gamma_ref'],
        mu_gamma=pvsyst_module_params['mu_gamma'],
        I_L_ref=pvsyst_module_params['I_L_ref'],
        I_o_ref=pvsyst_module_params['I_o_ref'],
        R_sh_ref=pvsyst_module_params['R_sh_ref'],
        R_sh_0=pvsyst_module_params['R_sh_0'],
        R_s=pvsyst_module_params['R_s'],
        cells_in_series=pvsyst_module_params['cells_in_series'],
        EgRef=pvsyst_module_params['EgRef'])

    assert_series_equal(
        IL.round(decimals=3), pd.Series([0.0, 4.8200], index=times))
    assert_series_equal(
        I0.round(decimals=3), pd.Series([0.0, 1.47e-7], index=times))
    assert_allclose(Rs, 0.500)
    assert_series_equal(
        Rsh.round(decimals=3), pd.Series([1000.0, 305.757], index=times))
    assert_series_equal(
        nNsVth.round(decimals=4), pd.Series([1.6186, 1.7961], index=times))


def test_PVSystem_calcparams_desoto(cec_module_params, mocker):
    mocker.spy(pvsystem, 'calcparams_desoto')
    module_parameters = cec_module_params.copy()
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    system = pvsystem.PVSystem(module_parameters=module_parameters)
    effective_irradiance = np.array([0, 800])
    temp_cell = 25
    IL, I0, Rs, Rsh, nNsVth = system.calcparams_desoto(effective_irradiance,
                                                       temp_cell)
    pvsystem.calcparams_desoto.assert_called_once_with(
                                  effective_irradiance,
                                  temp_cell,
                                  alpha_sc=cec_module_params['alpha_sc'],
                                  a_ref=cec_module_params['a_ref'],
                                  I_L_ref=cec_module_params['I_L_ref'],
                                  I_o_ref=cec_module_params['I_o_ref'],
                                  R_sh_ref=cec_module_params['R_sh_ref'],
                                  R_s=cec_module_params['R_s'],
                                  EgRef=module_parameters['EgRef'],
                                  dEgdT=module_parameters['dEgdT'])
    assert_allclose(IL, np.array([0.0, 6.036]), atol=1)
    assert_allclose(I0, 2.0e-9, atol=1.0e-9)
    assert_allclose(Rs, 0.1, atol=0.1)
    assert_allclose(Rsh, np.array([np.inf, 20]), atol=1)
    assert_allclose(nNsVth, 0.5, atol=0.1)


def test_PVSystem_calcparams_pvsyst(pvsyst_module_params, mocker):
    mocker.spy(pvsystem, 'calcparams_pvsyst')
    module_parameters = pvsyst_module_params.copy()
    system = pvsystem.PVSystem(module_parameters=module_parameters)
    effective_irradiance = np.array([0, 800])
    temp_cell = np.array([25, 50])
    IL, I0, Rs, Rsh, nNsVth = system.calcparams_pvsyst(effective_irradiance,
                                                       temp_cell)
    pvsystem.calcparams_pvsyst.assert_called_once_with(
                                  effective_irradiance,
                                  temp_cell,
                                  alpha_sc=pvsyst_module_params['alpha_sc'],
                                  gamma_ref=pvsyst_module_params['gamma_ref'],
                                  mu_gamma=pvsyst_module_params['mu_gamma'],
                                  I_L_ref=pvsyst_module_params['I_L_ref'],
                                  I_o_ref=pvsyst_module_params['I_o_ref'],
                                  R_sh_ref=pvsyst_module_params['R_sh_ref'],
                                  R_sh_0=pvsyst_module_params['R_sh_0'],
                                  R_s=pvsyst_module_params['R_s'],
                    cells_in_series=pvsyst_module_params['cells_in_series'],
                                  EgRef=pvsyst_module_params['EgRef'],
                                  R_sh_exp=pvsyst_module_params['R_sh_exp'])

    assert_allclose(IL, np.array([0.0, 4.8200]), atol=1)
    assert_allclose(I0, np.array([0.0, 1.47e-7]), atol=1.0e-5)
    assert_allclose(Rs, 0.5, atol=0.1)
    assert_allclose(Rsh, np.array([1000, 305.757]), atol=50)
    assert_allclose(nNsVth, np.array([1.6186, 1.7961]), atol=0.1)


@pytest.fixture(params=[
    {  # Can handle all python scalar inputs
     'Rsh': 20.,
     'Rs': 0.1,
     'nNsVth': 0.5,
     'I': 3.,
     'I0': 6.e-7,
     'IL': 7.,
     'V_expected': 7.5049875193450521
    },
    {  # Can handle all rank-0 array inputs
     'Rsh': np.array(20.),
     'Rs': np.array(0.1),
     'nNsVth': np.array(0.5),
     'I': np.array(3.),
     'I0': np.array(6.e-7),
     'IL': np.array(7.),
     'V_expected': np.array(7.5049875193450521)
    },
    {  # Can handle all rank-1 singleton array inputs
     'Rsh': np.array([20.]),
     'Rs': np.array([0.1]),
     'nNsVth': np.array([0.5]),
     'I': np.array([3.]),
     'I0': np.array([6.e-7]),
     'IL': np.array([7.]),
     'V_expected': np.array([7.5049875193450521])
    },
    {  # Can handle all rank-1 non-singleton array inputs with infinite shunt
      #  resistance, Rsh=inf gives V=Voc=nNsVth*(np.log(IL + I0) - np.log(I0)
      #  at I=0
      'Rsh': np.array([np.inf, 20.]),
      'Rs': np.array([0.1, 0.1]),
      'nNsVth': np.array([0.5, 0.5]),
      'I': np.array([0., 3.]),
      'I0': np.array([6.e-7, 6.e-7]),
      'IL': np.array([7., 7.]),
      'V_expected': np.array([0.5*(np.log(7. + 6.e-7) - np.log(6.e-7)),
                             7.5049875193450521])
    },
    {  # Can handle mixed inputs with a rank-2 array with infinite shunt
      #  resistance, Rsh=inf gives V=Voc=nNsVth*(np.log(IL + I0) - np.log(I0)
      #  at I=0
      'Rsh': np.array([[np.inf, np.inf], [np.inf, np.inf]]),
      'Rs': np.array([0.1]),
      'nNsVth': np.array(0.5),
      'I': 0.,
      'I0': np.array([6.e-7]),
      'IL': np.array([7.]),
      'V_expected': 0.5*(np.log(7. + 6.e-7) - np.log(6.e-7))*np.ones((2, 2))
    },
    {  # Can handle ideal series and shunt, Rsh=inf and Rs=0 give
      #  V = nNsVth*(np.log(IL - I + I0) - np.log(I0))
      'Rsh': np.inf,
      'Rs': 0.,
      'nNsVth': 0.5,
      'I': np.array([7., 7./2., 0.]),
      'I0': 6.e-7,
      'IL': 7.,
      'V_expected': np.array([0., 0.5*(np.log(7. - 7./2. + 6.e-7) -
                              np.log(6.e-7)), 0.5*(np.log(7. + 6.e-7) -
                              np.log(6.e-7))])
    },
    {  # Can handle only ideal series resistance, no closed form solution
      'Rsh': 20.,
      'Rs': 0.,
      'nNsVth': 0.5,
      'I': 3.,
      'I0': 6.e-7,
      'IL': 7.,
      'V_expected': 7.804987519345062
    },
    {  # Can handle all python scalar inputs with big LambertW arg
      'Rsh': 500.,
      'Rs': 10.,
      'nNsVth': 4.06,
      'I': 0.,
      'I0': 6.e-10,
      'IL': 1.2,
      'V_expected': 86.320000493521079
    },
    {  # Can handle all python scalar inputs with bigger LambertW arg
      #  1000 W/m^2 on a Canadian Solar 220M with 20 C ambient temp
      #  github issue 225 (this appears to be from PR 226 not issue 225)
      'Rsh': 190.,
      'Rs': 1.065,
      'nNsVth': 2.89,
      'I': 0.,
      'I0': 7.05196029e-08,
      'IL': 10.491262,
      'V_expected': 54.303958833791455
    },
    {  # Can handle all python scalar inputs with bigger LambertW arg
      #  1000 W/m^2 on a Canadian Solar 220M with 20 C ambient temp
      #  github issue 225
      'Rsh': 381.68,
      'Rs': 1.065,
      'nNsVth': 2.681527737715915,
      'I': 0.,
      'I0': 1.8739027472625636e-09,
      'IL': 5.1366949999999996,
      'V_expected': 58.19323124611128
    },
    {  # Verify mixed solution type indexing logic
      'Rsh': np.array([np.inf, 190., 381.68]),
      'Rs': 1.065,
      'nNsVth': np.array([2.89, 2.89, 2.681527737715915]),
      'I': 0.,
      'I0': np.array([7.05196029e-08, 7.05196029e-08, 1.8739027472625636e-09]),
      'IL': np.array([10.491262, 10.491262, 5.1366949999999996]),
      'V_expected': np.array([2.89*np.log1p(10.491262/7.05196029e-08),
                              54.303958833791455, 58.19323124611128])
    }])
def fixture_v_from_i(request):
    return request.param


@requires_scipy
@pytest.mark.parametrize(
    'method, atol', [('lambertw', 1e-11), ('brentq', 1e-11), ('newton', 1e-8)]
)
def test_v_from_i(fixture_v_from_i, method, atol):
    # Solution set loaded from fixture
    Rsh = fixture_v_from_i['Rsh']
    Rs = fixture_v_from_i['Rs']
    nNsVth = fixture_v_from_i['nNsVth']
    I = fixture_v_from_i['I']
    I0 = fixture_v_from_i['I0']
    IL = fixture_v_from_i['IL']
    V_expected = fixture_v_from_i['V_expected']

    V = pvsystem.v_from_i(Rsh, Rs, nNsVth, I, I0, IL, method=method)
    assert(isinstance(V, type(V_expected)))
    if isinstance(V, type(np.ndarray)):
        assert(isinstance(V.dtype, type(V_expected.dtype)))
        assert(V.shape == V_expected.shape)
    assert_allclose(V, V_expected, atol=atol)


@requires_scipy
def test_i_from_v_from_i(fixture_v_from_i):
    # Solution set loaded from fixture
    Rsh = fixture_v_from_i['Rsh']
    Rs = fixture_v_from_i['Rs']
    nNsVth = fixture_v_from_i['nNsVth']
    I = fixture_v_from_i['I']
    I0 = fixture_v_from_i['I0']
    IL = fixture_v_from_i['IL']
    V = fixture_v_from_i['V_expected']

    # Convergence criteria
    atol = 1.e-11

    I_expected = pvsystem.i_from_v(Rsh, Rs, nNsVth, V, I0, IL,
                                   method='lambertw')
    assert_allclose(I, I_expected, atol=atol)
    I = pvsystem.i_from_v(Rsh, Rs, nNsVth, V, I0, IL)
    assert(isinstance(I, type(I_expected)))
    if isinstance(I, type(np.ndarray)):
        assert(isinstance(I.dtype, type(I_expected.dtype)))
        assert(I.shape == I_expected.shape)
    assert_allclose(I, I_expected, atol=atol)


@pytest.fixture(params=[
    {  # Can handle all python scalar inputs
      'Rsh': 20.,
      'Rs': 0.1,
      'nNsVth': 0.5,
      'V': 7.5049875193450521,
      'I0': 6.e-7,
      'IL': 7.,
      'I_expected': 3.
    },
    {  # Can handle all rank-0 array inputs
      'Rsh': np.array(20.),
      'Rs': np.array(0.1),
      'nNsVth': np.array(0.5),
      'V': np.array(7.5049875193450521),
      'I0': np.array(6.e-7),
      'IL': np.array(7.),
      'I_expected': np.array(3.)
    },
    {  # Can handle all rank-1 singleton array inputs
      'Rsh': np.array([20.]),
      'Rs': np.array([0.1]),
      'nNsVth': np.array([0.5]),
      'V': np.array([7.5049875193450521]),
      'I0': np.array([6.e-7]),
      'IL': np.array([7.]),
      'I_expected': np.array([3.])
    },
    {  # Can handle all rank-1 non-singleton array inputs with a zero
      #  series resistance, Rs=0 gives I=IL=Isc at V=0
      'Rsh': np.array([20., 20.]),
      'Rs': np.array([0., 0.1]),
      'nNsVth': np.array([0.5, 0.5]),
      'V': np.array([0., 7.5049875193450521]),
      'I0': np.array([6.e-7, 6.e-7]),
      'IL': np.array([7., 7.]),
      'I_expected': np.array([7., 3.])
    },
    {  # Can handle mixed inputs with a rank-2 array with zero series
      #  resistance, Rs=0 gives I=IL=Isc at V=0
      'Rsh': np.array([20.]),
      'Rs': np.array([[0., 0.], [0., 0.]]),
      'nNsVth': np.array(0.5),
      'V': 0.,
      'I0': np.array([6.e-7]),
      'IL': np.array([7.]),
      'I_expected': np.array([[7., 7.], [7., 7.]])
    },
    {  # Can handle ideal series and shunt, Rsh=inf and Rs=0 give
      #  V_oc = nNsVth*(np.log(IL + I0) - np.log(I0))
      'Rsh': np.inf,
      'Rs': 0.,
      'nNsVth': 0.5,
      'V': np.array([0., 0.5*(np.log(7. + 6.e-7) - np.log(6.e-7))/2.,
                     0.5*(np.log(7. + 6.e-7) - np.log(6.e-7))]),
      'I0': 6.e-7,
      'IL': 7.,
      'I_expected': np.array([7., 7. - 6.e-7*np.expm1((np.log(7. + 6.e-7) -
                              np.log(6.e-7))/2.), 0.])
    },
    {  # Can handle only ideal shunt resistance, no closed form solution
      'Rsh': np.inf,
      'Rs': 0.1,
      'nNsVth': 0.5,
      'V': 7.5049875193450521,
      'I0': 6.e-7,
      'IL': 7.,
      'I_expected': 3.2244873645510923
    }])
def fixture_i_from_v(request):
    return request.param


@requires_scipy
@pytest.mark.parametrize(
    'method, atol', [('lambertw', 1e-11), ('brentq', 1e-11), ('newton', 1e-11)]
)
def test_i_from_v(fixture_i_from_v, method, atol):
    # Solution set loaded from fixture
    Rsh = fixture_i_from_v['Rsh']
    Rs = fixture_i_from_v['Rs']
    nNsVth = fixture_i_from_v['nNsVth']
    V = fixture_i_from_v['V']
    I0 = fixture_i_from_v['I0']
    IL = fixture_i_from_v['IL']
    I_expected = fixture_i_from_v['I_expected']

    I = pvsystem.i_from_v(Rsh, Rs, nNsVth, V, I0, IL, method=method)
    assert(isinstance(I, type(I_expected)))
    if isinstance(I, type(np.ndarray)):
        assert(isinstance(I.dtype, type(I_expected.dtype)))
        assert(I.shape == I_expected.shape)
    assert_allclose(I, I_expected, atol=atol)


@requires_scipy
def test_PVSystem_i_from_v(mocker):
    system = pvsystem.PVSystem()
    m = mocker.patch('pvlib.pvsystem.i_from_v', autospec=True)
    args = (20, 0.1, 0.5, 7.5049875193450521, 6e-7, 7)
    system.i_from_v(*args)
    m.assert_called_once_with(*args)


@requires_scipy
def test_i_from_v_size():
    with pytest.raises(ValueError):
        pvsystem.i_from_v(20, [0.1] * 2, 0.5, [7.5] * 3, 6.0e-7, 7.0)
    with pytest.raises(ValueError):
        pvsystem.i_from_v(20, [0.1] * 2, 0.5, [7.5] * 3, 6.0e-7, 7.0,
                          method='brentq')
    with pytest.raises(ValueError):
        pvsystem.i_from_v(20, 0.1, 0.5, [7.5] * 3, 6.0e-7, np.array([7., 7.]),
                          method='newton')


@requires_scipy
def test_v_from_i_size():
    with pytest.raises(ValueError):
        pvsystem.v_from_i(20, [0.1] * 2, 0.5, [3.0] * 3, 6.0e-7, 7.0)
    with pytest.raises(ValueError):
        pvsystem.v_from_i(20, [0.1] * 2, 0.5, [3.0] * 3, 6.0e-7, 7.0,
                          method='brentq')
    with pytest.raises(ValueError):
        pvsystem.v_from_i(20, [0.1], 0.5, [3.0] * 3, 6.0e-7, np.array([7., 7.]),
                          method='newton')


@requires_scipy
def test_mpp_floats():
    """test max_power_point"""
    IL, I0, Rs, Rsh, nNsVth = (7, 6e-7, .1, 20, .5)
    out = pvsystem.max_power_point(IL, I0, Rs, Rsh, nNsVth, method='brentq')
    expected = {'i_mp': 6.1362673597376753,  # 6.1390251797935704, lambertw
                'v_mp': 6.2243393757884284,  # 6.221535886625464, lambertw
                'p_mp': 38.194210547580511}  # 38.194165464983037} lambertw
    assert isinstance(out, dict)
    for k, v in out.items():
        assert np.isclose(v, expected[k])
    out = pvsystem.max_power_point(IL, I0, Rs, Rsh, nNsVth, method='newton')
    for k, v in out.items():
        assert np.isclose(v, expected[k])


@requires_scipy
def test_mpp_array():
    """test max_power_point"""
    IL, I0, Rs, Rsh, nNsVth = (np.array([7, 7]), 6e-7, .1, 20, .5)
    out = pvsystem.max_power_point(IL, I0, Rs, Rsh, nNsVth, method='brentq')
    expected = {'i_mp': [6.1362673597376753] * 2,
                'v_mp': [6.2243393757884284] * 2,
                'p_mp': [38.194210547580511] * 2}
    assert isinstance(out, dict)
    for k, v in out.items():
        assert np.allclose(v, expected[k])
    out = pvsystem.max_power_point(IL, I0, Rs, Rsh, nNsVth, method='newton')
    for k, v in out.items():
        assert np.allclose(v, expected[k])


@requires_scipy
def test_mpp_series():
    """test max_power_point"""
    idx = ['2008-02-17T11:30:00-0800', '2008-02-17T12:30:00-0800']
    IL, I0, Rs, Rsh, nNsVth = (np.array([7, 7]), 6e-7, .1, 20, .5)
    IL = pd.Series(IL, index=idx)
    out = pvsystem.max_power_point(IL, I0, Rs, Rsh, nNsVth, method='brentq')
    expected = pd.DataFrame({'i_mp': [6.1362673597376753] * 2,
                             'v_mp': [6.2243393757884284] * 2,
                             'p_mp': [38.194210547580511] * 2},
                            index=idx)
    assert isinstance(out, pd.DataFrame)
    for k, v in out.items():
        assert np.allclose(v, expected[k])
    out = pvsystem.max_power_point(IL, I0, Rs, Rsh, nNsVth, method='newton')
    for k, v in out.items():
        assert np.allclose(v, expected[k])


@requires_scipy
def test_singlediode_series(cec_module_params):
    times = pd.date_range(start='2015-01-01', periods=2, freq='12H')
    effective_irradiance = pd.Series([0.0, 800.0], index=times)
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
        effective_irradiance,
        temp_cell=25,
        alpha_sc=cec_module_params['alpha_sc'],
        a_ref=cec_module_params['a_ref'],
        I_L_ref=cec_module_params['I_L_ref'],
        I_o_ref=cec_module_params['I_o_ref'],
        R_sh_ref=cec_module_params['R_sh_ref'],
        R_s=cec_module_params['R_s'],
        EgRef=1.121,
        dEgdT=-0.0002677
    )
    out = pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth)
    assert isinstance(out, pd.DataFrame)


@requires_scipy
def test_singlediode_array():
    # github issue 221
    photocurrent = np.linspace(0, 10, 11)
    resistance_shunt = 16
    resistance_series = 0.094
    nNsVth = 0.473
    saturation_current = 1.943e-09

    sd = pvsystem.singlediode(photocurrent, saturation_current,
                              resistance_series, resistance_shunt, nNsVth,
                              method='lambertw')

    expected = np.array([
        0.        ,  0.54538398,  1.43273966,  2.36328163,  3.29255606,
        4.23101358,  5.16177031,  6.09368251,  7.02197553,  7.96846051,
        8.88220557])

    assert_allclose(sd['i_mp'], expected, atol=0.01)

    sd = pvsystem.singlediode(photocurrent, saturation_current,
                              resistance_series, resistance_shunt, nNsVth)

    expected = pvsystem.i_from_v(resistance_shunt, resistance_series, nNsVth,
                                 sd['v_mp'], saturation_current, photocurrent,
                                 method='lambertw')

    assert_allclose(sd['i_mp'], expected, atol=0.01)


@requires_scipy
def test_singlediode_floats():
    out = pvsystem.singlediode(7, 6e-7, .1, 20, .5, method='lambertw')
    expected = {'i_xx': 4.2498,
                'i_mp': 6.1275,
                'v_oc': 8.1063,
                'p_mp': 38.1937,
                'i_x': 6.7558,
                'i_sc': 6.9651,
                'v_mp': 6.2331,
                'i': None,
                'v': None}
    assert isinstance(out, dict)
    for k, v in out.items():
        if k in ['i', 'v']:
            assert v is None
        else:
            assert_allclose(v, expected[k], atol=1e-3)


@requires_scipy
def test_singlediode_floats_ivcurve():
    out = pvsystem.singlediode(7, 6e-7, .1, 20, .5, ivcurve_pnts=3, method='lambertw')
    expected = {'i_xx': 4.2498,
                'i_mp': 6.1275,
                'v_oc': 8.1063,
                'p_mp': 38.1937,
                'i_x': 6.7558,
                'i_sc': 6.9651,
                'v_mp': 6.2331,
                'i': np.array([6.965172e+00, 6.755882e+00, 2.575717e-14]),
                'v': np.array([0., 4.05315, 8.1063])}
    assert isinstance(out, dict)
    for k, v in out.items():
        assert_allclose(v, expected[k], atol=1e-3)


@requires_scipy
def test_singlediode_series_ivcurve(cec_module_params):
    times = pd.date_range(start='2015-06-01', periods=3, freq='6H')
    effective_irradiance = pd.Series([0.0, 400.0, 800.0], index=times)
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                                  effective_irradiance,
                                  temp_cell=25,
                                  alpha_sc=cec_module_params['alpha_sc'],
                                  a_ref=cec_module_params['a_ref'],
                                  I_L_ref=cec_module_params['I_L_ref'],
                                  I_o_ref=cec_module_params['I_o_ref'],
                                  R_sh_ref=cec_module_params['R_sh_ref'],
                                  R_s=cec_module_params['R_s'],
                                  EgRef=1.121,
                                  dEgdT=-0.0002677)

    out = pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth, ivcurve_pnts=3,
                               method='lambertw')

    expected = OrderedDict([('i_sc', array([0., 3.01054475, 6.00675648])),
                            ('v_oc', array([0., 9.96886962, 10.29530483])),
                            ('i_mp', array([0., 2.65191983, 5.28594672])),
                            ('v_mp', array([0., 8.33392491, 8.4159707])),
                            ('p_mp', array([0., 22.10090078, 44.48637274])),
                            ('i_x', array([0., 2.88414114, 5.74622046])),
                            ('i_xx', array([0., 2.04340914, 3.90007956])),
                            ('v', array([[0., 0., 0.],
                                         [0., 4.98443481, 9.96886962],
                                         [0., 5.14765242, 10.29530483]])),
                            ('i', array([[0., 0., 0.],
                                         [3.01079860e+00, 2.88414114e+00,
                                          3.10862447e-14],
                                         [6.00726296e+00, 5.74622046e+00,
                                          0.00000000e+00]]))])

    for k, v in out.items():
        assert_allclose(v, expected[k], atol=1e-2)

    out = pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth, ivcurve_pnts=3)

    expected['i_mp'] = pvsystem.i_from_v(Rsh, Rs, nNsVth, out['v_mp'], I0, IL,
                                         method='lambertw')
    expected['v_mp'] = pvsystem.v_from_i(Rsh, Rs, nNsVth, out['i_mp'], I0, IL,
                                         method='lambertw')
    expected['i'] = pvsystem.i_from_v(Rsh, Rs, nNsVth, out['v'].T, I0, IL,
                                         method='lambertw').T
    expected['v'] = pvsystem.v_from_i(Rsh, Rs, nNsVth, out['i'].T, I0, IL,
                                         method='lambertw').T

    for k, v in out.items():
        assert_allclose(v, expected[k], atol=1e-2)


def test_scale_voltage_current_power():
    data = pd.DataFrame(
        np.array([[2, 1.5, 10, 8, 12, 0.5, 1.5]]),
        columns=['i_sc', 'i_mp', 'v_oc', 'v_mp', 'p_mp', 'i_x', 'i_xx'],
        index=[0])
    expected = pd.DataFrame(
        np.array([[6, 4.5, 20, 16, 72, 1.5, 4.5]]),
        columns=['i_sc', 'i_mp', 'v_oc', 'v_mp', 'p_mp', 'i_x', 'i_xx'],
        index=[0])
    out = pvsystem.scale_voltage_current_power(data, voltage=2, current=3)
    assert_frame_equal(out, expected, check_less_precise=5)


def test_PVSystem_scale_voltage_current_power(mocker):
    data = None
    system = pvsystem.PVSystem(modules_per_string=2, strings_per_inverter=3)
    m = mocker.patch(
        'pvlib.pvsystem.scale_voltage_current_power', autospec=True)
    system.scale_voltage_current_power(data)
    m.assert_called_once_with(data, voltage=2, current=3)


def test_PVSystem_snlinverter(cec_inverter_parameters):
    system = pvsystem.PVSystem(
        inverter=cec_inverter_parameters['Name'],
        inverter_parameters=cec_inverter_parameters,
    )
    vdcs = pd.Series(np.linspace(0,50,3))
    idcs = pd.Series(np.linspace(0,11,3))
    pdcs = idcs * vdcs

    pacs = system.snlinverter(vdcs, pdcs)
    assert_series_equal(pacs, pd.Series([-0.020000, 132.004308, 250.000000]))


def test_PVSystem_creation():
    pv_system = pvsystem.PVSystem(module='blah', inverter='blarg')
    # ensure that parameter attributes are dict-like. GH 294
    pv_system.module_parameters['pdc0'] = 1
    pv_system.inverter_parameters['Paco'] = 1


def test_PVSystem_get_aoi():
    system = pvsystem.PVSystem(surface_tilt=32, surface_azimuth=135)
    aoi = system.get_aoi(30, 225)
    assert np.round(aoi, 4) == 42.7408


def test_PVSystem_get_irradiance():
    system = pvsystem.PVSystem(surface_tilt=32, surface_azimuth=135)
    times = pd.date_range(start='20160101 1200-0700',
                          end='20160101 1800-0700', freq='6H')
    location = Location(latitude=32, longitude=-111)
    solar_position = location.get_solarposition(times)
    irrads = pd.DataFrame({'dni':[900,0], 'ghi':[600,0], 'dhi':[100,0]},
                          index=times)

    irradiance = system.get_irradiance(solar_position['apparent_zenith'],
                                       solar_position['azimuth'],
                                       irrads['dni'],
                                       irrads['ghi'],
                                       irrads['dhi'])

    expected = pd.DataFrame(data=np.array(
        [[ 883.65494055,  745.86141676,  137.79352379,  126.397131  ,
              11.39639279],
           [   0.        ,   -0.        ,    0.        ,    0.        ,    0.        ]]),
                            columns=['poa_global', 'poa_direct',
                                     'poa_diffuse', 'poa_sky_diffuse',
                                     'poa_ground_diffuse'],
                            index=times)

    assert_frame_equal(irradiance, expected, check_less_precise=2)


def test_PVSystem_localize_with_location():
    system = pvsystem.PVSystem(module='blah', inverter='blarg')
    location = Location(latitude=32, longitude=-111)
    localized_system = system.localize(location=location)

    assert localized_system.module == 'blah'
    assert localized_system.inverter == 'blarg'
    assert localized_system.latitude == 32
    assert localized_system.longitude == -111


def test_PVSystem_localize_with_latlon():
    system = pvsystem.PVSystem(module='blah', inverter='blarg')
    localized_system = system.localize(latitude=32, longitude=-111)

    assert localized_system.module == 'blah'
    assert localized_system.inverter == 'blarg'
    assert localized_system.latitude == 32
    assert localized_system.longitude == -111


def test_PVSystem___repr__():
    system = pvsystem.PVSystem(
        module='blah', inverter='blarg', name='pv ftw',
        temperature_model_parameters={'a': -3.56})

    expected = """PVSystem:
  name: pv ftw
  surface_tilt: 0
  surface_azimuth: 180
  module: blah
  inverter: blarg
  albedo: 0.25
  racking_model: None
  module_type: None
  temperature_model_parameters: {'a': -3.56}"""
    assert system.__repr__() == expected


def test_PVSystem_localize___repr__():
    system = pvsystem.PVSystem(
        module='blah', inverter='blarg', name='pv ftw',
        temperature_model_parameters={'a': -3.56})
    localized_system = system.localize(latitude=32, longitude=-111)
    # apparently name is not preserved when creating a system using localize
    expected = """LocalizedPVSystem:
  name: None
  latitude: 32
  longitude: -111
  altitude: 0
  tz: UTC
  surface_tilt: 0
  surface_azimuth: 180
  module: blah
  inverter: blarg
  albedo: 0.25
  racking_model: None
  module_type: None
  temperature_model_parameters: {'a': -3.56}"""

    assert localized_system.__repr__() == expected


# we could retest each of the models tested above
# when they are attached to LocalizedPVSystem, but
# that's probably not necessary at this point.


def test_LocalizedPVSystem_creation():
    localized_system = pvsystem.LocalizedPVSystem(latitude=32,
                                                  longitude=-111,
                                                  module='blah',
                                                  inverter='blarg')

    assert localized_system.module == 'blah'
    assert localized_system.inverter == 'blarg'
    assert localized_system.latitude == 32
    assert localized_system.longitude == -111


def test_LocalizedPVSystem___repr__():
    localized_system = pvsystem.LocalizedPVSystem(
        latitude=32, longitude=-111, module='blah', inverter='blarg',
        name='my name', temperature_model_parameters={'a': -3.56})

    expected = """LocalizedPVSystem:
  name: my name
  latitude: 32
  longitude: -111
  altitude: 0
  tz: UTC
  surface_tilt: 0
  surface_azimuth: 180
  module: blah
  inverter: blarg
  albedo: 0.25
  racking_model: None
  module_type: None
  temperature_model_parameters: {'a': -3.56}"""

    assert localized_system.__repr__() == expected


def test_pvwatts_dc_scalars():
    expected = 88.65
    out = pvsystem.pvwatts_dc(900, 30, 100, -0.003)
    assert_allclose(out, expected)


@needs_numpy_1_10
def test_pvwatts_dc_arrays():
    irrad_trans = np.array([np.nan, 900, 900])
    temp_cell = np.array([30, np.nan, 30])
    irrad_trans, temp_cell = np.meshgrid(irrad_trans, temp_cell)
    expected = np.array([[nan,  88.65,  88.65],
                         [nan,    nan,    nan],
                         [nan,  88.65,  88.65]])
    out = pvsystem.pvwatts_dc(irrad_trans, temp_cell, 100, -0.003)
    assert_allclose(out, expected, equal_nan=True)


def test_pvwatts_dc_series():
    irrad_trans = pd.Series([np.nan, 900, 900])
    temp_cell = pd.Series([30, np.nan, 30])
    expected = pd.Series(np.array([   nan,    nan,  88.65]))
    out = pvsystem.pvwatts_dc(irrad_trans, temp_cell, 100, -0.003)
    assert_series_equal(expected, out)


def test_pvwatts_losses_default():
    expected = 14.075660688264469
    out = pvsystem.pvwatts_losses()
    assert_allclose(out, expected)


@needs_numpy_1_10
def test_pvwatts_losses_arrays():
    expected = np.array([nan, 14.934904])
    age = np.array([nan, 1])
    out = pvsystem.pvwatts_losses(age=age)
    assert_allclose(out, expected)


def test_pvwatts_losses_series():
    expected = pd.Series([nan, 14.934904])
    age = pd.Series([nan, 1])
    out = pvsystem.pvwatts_losses(age=age)
    assert_series_equal(expected, out)


def make_pvwatts_system_defaults():
    module_parameters = {'pdc0': 100, 'gamma_pdc': -0.003}
    inverter_parameters = {'pdc0': 90}
    system = pvsystem.PVSystem(module_parameters=module_parameters,
                               inverter_parameters=inverter_parameters)
    return system


def make_pvwatts_system_kwargs():
    module_parameters = {'pdc0': 100, 'gamma_pdc': -0.003, 'temp_ref': 20}
    inverter_parameters = {'pdc0': 90, 'eta_inv_nom': 0.95, 'eta_inv_ref': 1.0}
    system = pvsystem.PVSystem(module_parameters=module_parameters,
                               inverter_parameters=inverter_parameters)
    return system


def test_PVSystem_pvwatts_dc(mocker):
    mocker.spy(pvsystem, 'pvwatts_dc')
    system = make_pvwatts_system_defaults()
    irrad = 900
    temp_cell = 30
    expected = 90
    out = system.pvwatts_dc(irrad, temp_cell)
    pvsystem.pvwatts_dc.assert_called_once_with(irrad, temp_cell,
                                                **system.module_parameters)
    assert_allclose(expected, out, atol=10)


def test_PVSystem_pvwatts_dc_kwargs(mocker):
    mocker.spy(pvsystem, 'pvwatts_dc')
    system = make_pvwatts_system_kwargs()
    irrad = 900
    temp_cell = 30
    expected = 90
    out = system.pvwatts_dc(irrad, temp_cell)
    pvsystem.pvwatts_dc.assert_called_once_with(irrad, temp_cell,
                                                **system.module_parameters)
    assert_allclose(expected, out, atol=10)


def test_PVSystem_pvwatts_losses(mocker):
    mocker.spy(pvsystem, 'pvwatts_losses')
    system = make_pvwatts_system_defaults()
    age = 1
    system.losses_parameters = dict(age=age)
    expected = 15
    out = system.pvwatts_losses()
    pvsystem.pvwatts_losses.assert_called_once_with(age=age)
    assert out < expected


def test_PVSystem_pvwatts_ac(mocker):
    mocker.spy(inverter, 'pvwatts')
    system = make_pvwatts_system_defaults()
    pdc = 50
    out = system.pvwatts_ac(pdc)
    inverter.pvwatts.assert_called_once_with(pdc,
                                             **system.inverter_parameters)
    assert out < pdc


def test_PVSystem_pvwatts_ac_kwargs(mocker):
    mocker.spy(inverter, 'pvwatts')
    system = make_pvwatts_system_kwargs()
    pdc = 50
    out = system.pvwatts_ac(pdc)
    inverter.pvwatts.assert_called_once_with(pdc,
                                             **system.inverter_parameters)
    assert out < pdc


@fail_on_pvlib_version('0.9')
def test_deprecated_09(cec_inverter_parameters, adr_inverter_parameters):
    # deprecated function pvsystem.snlinverter
    with pytest.warns(pvlibDeprecationWarning):
        pvsystem.snlinverter(250, 40, cec_inverter_parameters)
    # deprecated function pvsystem.adrinverter
    with pytest.warns(pvlibDeprecationWarning):
        pvsystem.adrinverter(1232, 154, adr_inverter_parameters)
    # deprecated function pvsystem.spvwatts_ac
    with pytest.warns(pvlibDeprecationWarning):
        pvsystem.pvwatts_ac(90, 100, 0.95)
    # for missing temperature_model_parameters
    match = "Reverting to deprecated default: SAPM cell temperature"
    system = pvsystem.PVSystem()
    with pytest.warns(pvlibDeprecationWarning, match=match):
        system.sapm_celltemp(1, 2, 3)
