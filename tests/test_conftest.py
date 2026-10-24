import pytest

from tests import conftest


@pytest.mark.parametrize('function_name', ['assert_index_equal',
                                           'assert_series_equal',
                                           'assert_frame_equal'])
@pytest.mark.parametrize('pd_version', ['1.0.0', '1.1.0'])
@pytest.mark.parametrize('check_less_precise', [True, False])
def test__check_pandas_assert_kwargs(mocker, function_name, pd_version,
                                     check_less_precise):
    # test that conftest._check_pandas_assert_kwargs returns appropriate
    # kwargs for the assert_x_equal functions

    # NOTE: be careful about mixing mocker.patch and pytest.MonkeyPatch!
    # they do not coordinate their cleanups, so it is safest to only
    # use one or the other.  GH #1447

    # patch the pandas assert; not interested in actually calling them,
    # plus we want to spy on how they get called.
    spy = mocker.patch('pandas.testing.' + function_name)
    # patch pd.__version__ to exercise the two branches in
    # conftest._check_pandas_assert_kwargs
    mocker.patch('pandas.__version__', new=pd_version)

    # finally, run the function and check what args got passed to pandas:
    assert_function = getattr(conftest, function_name)
    args = [None, None]
    assert_function(*args, check_less_precise=check_less_precise)
    if pd_version == '1.1.0':
        tol = 1e-3 if check_less_precise else 1e-5
        expected_kwargs = {'atol': tol, 'rtol': tol}
    else:
        expected_kwargs = {'check_less_precise': check_less_precise}
    if function_name == "assert_index_equal":
        # assert_index_equal in Pandas 3.1.0 warns:
        # Pandas4Warning: assert_index_equal will check the 'freq' attribute by default in a future version; these freqs <Minute> and None do not match. Pass check_freq=True or check_freq=False to silence this warning
        # pd.testing.assert_index_equal(left, right, **kwargs)
        expected_kwargs['check_freq'] = False

    spy.assert_called_once_with(*args, **expected_kwargs)
