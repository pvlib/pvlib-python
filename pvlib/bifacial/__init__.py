"""
The ``bifacial`` submodule contains functions to model bifacial modules.
"""

from pvlib._deprecation import deprecated
from pvlib.bifacial import pvfactors, infinite_sheds, utils  # noqa: F401
from .loss_models import power_mismatch_deline  # noqa: F401

pvfactors_timeseries = deprecated(
    since='0.9.1',
    name='pvlib.bifacial.pvfactors_timeseries',
    alternative='pvlib.bifacial.pvfactors.pvfactors_timeseries'
)(pvfactors.pvfactors_timeseries)
