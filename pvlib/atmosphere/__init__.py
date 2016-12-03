"""atmospheric equations"""

from pvlib.atmosphere.core import (
    absoluteairmass, alt2pres, first_solar_spectral_correction, gueymard94_pw,
    pres2alt, relativeairmass, AIRMASS_MODELS, APPARENT_ZENITH_MODELS,
    TRUE_ZENITH_MODELS
)
from pvlib.atmosphere.linke_turb_forms import kasten_96lt
