"""
Deprecated version of pvlib.iotools.tmy
"""

from pvlib.iotools import read_tmy2, read_tmy3

from pvlib._deprecation import deprecated

readtmy2 = deprecated('0.6.1', alternative='iotools.read_tmy2',
                      name='readtmy2', removal='0.7')(read_tmy2)

readtmy3 = deprecated('0.6.1', alternative='iotools.read_tmy3',
                      name='readtmy3', removal='0.7')(read_tmy3)
