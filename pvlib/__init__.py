# First ensure proper Python version.
import sys
if not ((3, 5) <= sys.version_info and sys.version_info < (3, 8)):
    raise RuntimeError("Current Python version is {}.{}.{}, but pvlib-python \
is only compatible with Python 3.5-7.".format(
        sys.version_info[0], sys.version_info[1], sys.version_info[2]))

from pvlib.version import __version__
from pvlib import tools
from pvlib import atmosphere
from pvlib import clearsky
# from pvlib import forecast
from pvlib import irradiance
from pvlib import location
from pvlib import solarposition
from pvlib import iotools
from pvlib import tracking
from pvlib import pvsystem
from pvlib import spa
from pvlib import modelchain
from pvlib import singlediode

# for backwards compatibility for pvlib.tmy module
from pvlib import tmy
