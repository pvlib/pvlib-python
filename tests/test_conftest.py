import pandas as pd
import pytest

from .conftest import assert_index_equal


@pytest.mark.parametrize(
    "version, expect_check_freq",
    [("1.5.0", False), ("3.1.0", True)],
)
def test_assert_index_equal_check_freq(
        monkeypatch, version, expect_check_freq):
    """check_freq is only passed on pandas >= 3.1.0, where the kwarg exists.

    On older pandas, pandas.testing.assert_index_equal does not accept
    check_freq, so passing it unconditionally would raise TypeError (and
    would break the CI minimum environment, which pins pandas 1.5.0).
    """
    left = pd.Index([1, 2, 3])
    right = pd.Index([1, 2, 3])
    monkeypatch.setattr(pd, "__version__", version)
    calls = []
    monkeypatch.setattr(
        pd.testing, "assert_index_equal",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    assert_index_equal(left, right, atol=1e-5, rtol=1e-5)
    assert len(calls) == 1
    if expect_check_freq:
        assert calls[0].get("check_freq") is False
    else:
        assert "check_freq" not in calls[0]
