"""
Test the _deprecation module.
"""

import pytest

from pvlib import _deprecation


@pytest.fixture
def renamed_kwarg_func():
    """Returns a function decorated by renamed_kwarg_warning."""
    @_deprecation.renamed_kwarg_warning(
        "0.1.0", "old_kwarg", "new_kwarg", "0.2.0"
    )
    def func(new_kwarg):
        return new_kwarg

    return func


def test_renamed_kwarg_warning(renamed_kwarg_func):
    # assert no warning is raised when using the new kwarg
    assert renamed_kwarg_func(new_kwarg=1) == 1  # as keyword argument
    assert renamed_kwarg_func(1) == 1  # as positional argument

    # assert a warning is raised when using the old kwarg
    with pytest.warns(Warning, match="Parameter 'old_kwarg' has been renamed"):
        assert renamed_kwarg_func(old_kwarg=1) == 1

    # assert an error is raised when using both the old and new kwarg
    with pytest.raises(ValueError, match="they refer to the same parameter."):
        renamed_kwarg_func(old_kwarg=1, new_kwarg=2)

    # assert when not providing any of them
    with pytest.raises(
        TypeError, match="missing 1 required positional argument"
    ):
        renamed_kwarg_func()
